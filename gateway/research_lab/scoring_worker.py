"""Gateway-owned Research Lab private scoring worker."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import functools
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from gateway.research_lab.bundles import build_research_lab_audit_bundle
from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.code_build import (
    CodeEditBuildError,
    CodeEditCandidateBuilder,
    CodeEditPatchApplyError,
    resolve_source_inspection_requests,
)
from gateway.research_lab.config import (
    DEFAULT_BASELINE_START_UTC_OFFSET_SECONDS,
    DEFAULT_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS,
    ResearchLabGatewayConfig,
)
from gateway.research_lab.icp_window import (
    RollingIcpWindowUnavailable,
    fetch_rolling_icp_window,
    reconstruct_icp_window_from_doc,
    utc_day_start,
    utc_set_id_for_datetime,
)
from gateway.research_lab.logging_utils import (
    compact_ref,
    event_error_diagnostics as _event_error_diagnostics,
    format_worker_block,
    runtime_error_diagnostics as _runtime_error_diagnostics,
    safe_event_error_text as _safe_event_error_text,
)
from gateway.research_lab.maintenance import is_scoring_maintenance_paused
from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest, ResearchLabScoreBundleCreateRequest
from gateway.research_lab.promotion import (
    CONFIRMATION_ATTEMPT_FAILED_REASON,
    CONFIRMATION_CLOSED_REASON,
    CONFIRMATION_HOLD_REASON,
    CONFIRMATION_NON_CLOSING_STATUSES,
    CONFIRMATION_RESULT_REASON,
    CONFIRMATION_STARTED_REASON,
    ResearchLabPromotionController,
    candidate_already_promoted,
    confirmation_attempt_budget,
    load_active_private_model,
    load_confirmation_state,
    private_source_push_retry_seconds,
    promotion_confirmation_rerun_enabled,
    promotion_improvement_metric,
    reconcile_failed_private_source_pushes,
    sync_active_model_to_repo_head,
)
from gateway.research_lab.public_activity import safe_project_public_loop_activity
from gateway.research_lab.public_benchmarks import (
    build_benchmark_visibility_split,
    build_public_benchmark_report,
    sanitize_benchmark_item_summary,
)
from gateway.research_lab.trajectory_projector import execution_trace_id_for_node
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_artifact,
    create_candidate_evaluation_event,
    create_candidate_promotion_event,
    create_private_model_benchmark_bundle,
    create_private_model_benchmark_event,
    create_public_benchmark_report,
    create_receipt_event,
    create_rolling_icp_window,
    create_score_bundle,
    create_scoring_dispatch_event,
    create_signed_audit_bundle,
    create_ticket_event,
    deterministic_uuid,
    insert_row,
    select_all,
    select_many,
    select_one,
)
from research_lab.canonical import canonical_json, sha256_json
from research_lab.code_editing import (
    CodeEditDraft,
    CodeEditSourceInspectionRequest,
    build_code_edit_repair_messages,
    extract_unified_diff_paths,
    parse_code_edit_repair_response,
)
from research_lab.eval import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    SealedBenchmarkSet,
    evaluate_private_model_pair,
    ensure_private_model_outputs,
    private_model_env_passthrough,
    sign_digest_with_kms,
)
from research_lab.eval.miner_report_stats import build_icp_stats
from research_lab.eval.evaluator import (
    INCONTAINER_TRACE_KMS_KEY_ENV,
    INCONTAINER_TRACE_S3_PREFIX_ENV,
    QualificationStyleCompanyScorer,
    _benchmark_style_score as _queue_benchmark_style_score,
    _upload_incontainer_trace as _upload_incontainer_trace_doc,
    build_holdout_gate_result,
    build_score_bundle_from_scored_icps,
    score_private_model_pair_items,
)
from gateway.research_lab import global_icp_queue
from research_lab.eval.private_runtime import (
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
    begin_incontainer_trace_collection,
    end_incontainer_trace_collection,
    incontainer_trace_capture_enabled,
)
from research_lab.observability.langfuse_client import observation, run_trace_id as langfuse_run_trace_id
from research_lab.eval.provider_costs import (
    cost_event_from_trace_entry,
    summarize_provider_cost_trace_entries,
)
from research_lab.observability.redaction import miner_hotkey_hash
from research_lab.observability.tracing import finish_score_bundle_observation


logger = logging.getLogger(__name__)

STALE_PARENT_REBASE_REPAIR_MAX_TOKENS = 32_768
STALE_PARENT_REBASE_REPAIR_REASONING_BODY = {"enabled": True, "effort": "max"}
STALE_PARENT_REBASE_REPAIR_EMPTY_CONTENT_ATTEMPTS = 2
PRIVATE_BASELINE_FAST_EMPTY_ABORT_AFTER = 6
PRIVATE_BASELINE_FAST_EMPTY_ABORT_SECONDS = 90.0
_POSTGREST_TIMESTAMP_RE = re.compile(
    r"^(?P<prefix>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
    r"\.(?P<fraction>\d{1,9})(?P<suffix>Z|[+-]\d{2}(?::?\d{2})?)?$"
)


def _retry_runner_with_provider_cost_scope(
    runner: DockerPrivateModelRunner,
    *,
    retry_round: int,
) -> DockerPrivateModelRunner:
    """Clone a retry runner with a fresh provider-cost budget scope.

    The Docker runtime hashes this evaluation scope together with the actual
    ICP payload, so every ICP still gets an independent ledger. The extra retry
    seed prevents an in-attempt retry from inheriting the first-pass spend.
    """

    extra_env = dict(runner.spec.extra_env or {})
    base_scope = str(extra_env.get(PROVIDER_COST_EVALUATION_SCOPE_ENV) or "").strip()
    retry_scope = sha256_json(
        {
            "schema_version": "research_lab_provider_cost_retry_scope.v1",
            "base_scope": base_scope,
            "retry_round": int(retry_round),
            "retry_round_started_at_ms": int(time.time() * 1000),
        }
    )
    extra_env[PROVIDER_COST_EVALUATION_SCOPE_ENV] = retry_scope
    return DockerPrivateModelRunner(
        replace(
            runner.spec,
            extra_env=extra_env,
            # The first-pass runner already pulled this immutable digest.
            pull_before_run=False,
        )
    )


class StaleParentDuringScoring(RuntimeError):
    """Raised at an ICP boundary when a candidate's parent is no longer current."""

    def __init__(
        self,
        *,
        active_artifact: PrivateModelArtifactManifest,
        candidate_parent: str,
        progress: Mapping[str, Any],
    ) -> None:
        self.active_artifact = active_artifact
        self.candidate_parent = candidate_parent
        self.progress = dict(progress)
        completed = int(self.progress.get("completed_icp_count") or 0)
        super().__init__(
            "candidate parent changed during scoring: "
            f"candidate_parent={compact_ref(candidate_parent)} "
            f"active_parent={compact_ref(active_artifact.model_artifact_hash)} "
            f"completed_icps={completed}"
        )


def _idle_log_seconds() -> float:
    try:
        return max(10.0, float(os.getenv("RESEARCH_LAB_WORKER_IDLE_LOG_SECONDS", "60")))
    except ValueError:
        return 60.0


def _error_backoff_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("RESEARCH_LAB_WORKER_ERROR_BACKOFF_SECONDS", "60")))
    except ValueError:
        return 60.0


def _short_error(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:300]}"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return int(default)


# Event-doc error sanitizers (bug #36) moved to logging_utils so the loop
# engine shares them; imported at the top with underscore aliases to keep
# existing call sites and tests stable.

_PROVIDER_429_RETRY_BACKOFF_SECONDS = 15.0
_PROVIDER_COST_CAP_ERROR_MARKERS = (
    "research_lab_provider_cost_cap_exceeded",
    "cost_cap_reached",
    "provider_cost_cap_blocked",
    "provider cost cap",
    "budget_soft_stop",
)


def _provider_cost_cap_error_text(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    if any(marker in lowered for marker in _PROVIDER_COST_CAP_ERROR_MARKERS):
        return True
    diagnostics = _runtime_error_diagnostics(error_text)
    try:
        return int(diagnostics.get("status") or 0) == 402
    except (TypeError, ValueError):
        return False


def _baseline_error_is_retryable(error_text: str) -> bool:
    """Transient provider/infra failures are retryable; model code bugs and
    auth/request errors are not.

    Must receive the full exception text: ``_short_error`` truncates to 300
    chars and can drop the status marker. Note ``_runtime_error_diagnostics``
    buckets 429 into ``provider_http_4xx`` — a plain 4xx check would mark the
    rate-limit error this classifier exists to catch as permanent, so 429 is
    matched explicitly before the 4xx rejection branch.
    """
    lowered = error_text.lower()
    diagnostics = _runtime_error_diagnostics(error_text)
    status = int(diagnostics.get("status") or 0)
    provider = str(diagnostics.get("provider") or "unknown")
    if _provider_cost_cap_error_text(error_text):
        return False
    if status in (408, 429) or status >= 500:
        return True
    # Scrapingdog's 400 "Something went wrong or profile not found" is
    # transient/data-shaped, not auth or request validation, and can recover on
    # retry. Production evidence showed 410 Gone retries did not produce usable
    # content, so 410 is treated as a terminal provider/data miss.
    if status == 400 and (provider == "scrapingdog" or "something went wrong" in lowered):
        return True
    if status in (400, 401, 402, 403, 404, 409, 410):
        return False
    if any(
        marker in lowered
        for marker in (
            "too many requests",
            "rate limit",
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
        )
    ):
        return True
    if any(
        marker in lowered
        for marker in (
            "exit status 137",
            "killed",
            "docker daemon",
            "no space left on device",
        )
    ):
        # Infra pressure (OOM-kill, daemon wedge) clears when the retry rounds
        # run fewer containers at once.
        return True
    return False


def _baseline_429_retry_backoff_seconds(error_text: str) -> float:
    """Return a short backoff only for explicit provider HTTP 429 errors."""

    diagnostics = _runtime_error_diagnostics(error_text)
    try:
        status = int(diagnostics.get("status") or 0)
    except (TypeError, ValueError):
        status = 0
    if status == 429:
        return _PROVIDER_429_RETRY_BACKOFF_SECONDS
    return 0.0


class BaselineHealthGateFailure(RuntimeError):
    """Raised when an explicitly configured baseline quality guard blocks a run."""

    def __init__(self, message: str, *, baseline_health: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.baseline_health = dict(baseline_health)


class ConfirmationMeasurementUnhealthy(RuntimeError):
    """Raised when a §5.2-2 confirmation measurement is too degraded to decide
    on (too many unresolved provider errors on either side). The attempt is
    recorded as failed and the candidate stays held, bounded by the
    claim-attempt budget — a flaky confirmation must neither reject a good
    candidate nor confirm a phantom improvement."""


# --- §0-N6: scorer-side burst isolation for the parallel baseline ----------
# Only Exa is container-isolated today; the qualification scoring calls
# (score_with_breakdowns — OpenRouter/Scrapingdog) burst N-wide on the prod
# keys during a parallel baseline. Opt-in dedicated benchmark scorer keys,
# applied ONLY within a baseline-batch scoring scope and falling back to prod
# values when unset — mirroring how the benchmark Exa key is plumbed.


def _benchmark_scorer_scrapingdog_api_key() -> str:
    return os.getenv("RESEARCH_LAB_BENCHMARK_SCRAPINGDOG_API_KEY", "").strip()


def _benchmark_scorer_openrouter_api_key() -> str:
    return os.getenv("RESEARCH_LAB_BENCHMARK_OPENROUTER_API_KEY", "").strip()


def _benchmark_scorer_max_concurrency() -> int:
    """Optional cap on concurrent scorer calls inside a baseline batch.
    0 (default) = unlimited (scorer calls fan out at batch concurrency)."""
    try:
        return max(0, int(os.getenv("RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY", "0")))
    except ValueError:
        return 0


# Module attributes that cache provider keys at import time inside the
# qualification scoring stack (the Scrapingdog fetches read os.environ per
# request; the OpenRouter calls read these module constants).
_SCORER_KEY_MODULE_ATTRS: tuple[tuple[str, str, str], ...] = (
    ("gateway.qualification.utils.helpers", "OPENROUTER_API_KEY", "openrouter"),
    ("qualification.scoring.verification_helpers", "OPENROUTER_API_KEY", "openrouter"),
    ("qualification.scoring.verification_helpers", "SCRAPINGDOG_API_KEY", "scrapingdog"),
)


@contextlib.contextmanager
def _benchmark_scorer_isolation():
    """Apply the dedicated benchmark scorer keys for a baseline-batch scope.

    The scoring worker is a dedicated process (worker_process.py), so a
    process-env override scoped to the batch cannot touch live qualification
    traffic; candidate scoring in the same process runs strictly after the
    batch returns and sees the restored prod values. Container runs inside the
    batch keep their prod provider keys: DockerPrivateModelSpec.extra_env was
    captured at runner construction and overrides os.environ passthrough.

    No-op when neither benchmark scorer key is configured.
    """
    scrapingdog = _benchmark_scorer_scrapingdog_api_key()
    openrouter = _benchmark_scorer_openrouter_api_key()
    if not scrapingdog and not openrouter:
        yield
        return
    env_overrides: dict[str, str] = {}
    if scrapingdog:
        env_overrides["SCRAPINGDOG_API_KEY"] = scrapingdog
        env_overrides["QUALIFICATION_SCRAPINGDOG_API_KEY"] = scrapingdog
    if openrouter:
        env_overrides["QUALIFICATION_OPENROUTER_API_KEY"] = openrouter
    saved_env: dict[str, str | None] = {}
    saved_attrs: list[tuple[Any, str, Any]] = []
    try:
        for name, value in env_overrides.items():
            saved_env[name] = os.environ.get(name)
            os.environ[name] = value
        for module_name, attr, provider in _SCORER_KEY_MODULE_ATTRS:
            value = scrapingdog if provider == "scrapingdog" else openrouter
            if not value:
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception:  # pragma: no cover - scorer stack absent in some envs
                continue
            if hasattr(module, attr):
                saved_attrs.append((module, attr, getattr(module, attr)))
                setattr(module, attr, value)
        yield
    finally:
        for module, attr, previous in saved_attrs:
            try:
                setattr(module, attr, previous)
            except Exception:  # pragma: no cover
                logger.warning(
                    "research_lab_benchmark_scorer_attr_restore_failed attr=%s", attr
                )
        for name, previous in saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _confirmation_lease_seconds() -> int:
    """Staleness window for a confirmation_rerun_started claim: a dead worker's
    in-flight confirmation becomes reclaimable after this many seconds."""
    try:
        return max(600, int(os.getenv("RESEARCH_LAB_CONFIRMATION_LEASE_SECONDS", "7200")))
    except ValueError:
        return 7200


def _collect_confirmation_scores(
    summaries: list[dict[str, Any]],
) -> tuple[dict[str, float], list[str], int]:
    """(icp_ref -> score, unresolved-error refs, non-empty output count) from a
    baseline-batch result. Reads the underscore orchestration fields without
    mutating the summaries (confirmation summaries are never stored raw)."""
    scores: dict[str, float] = {}
    unresolved: list[str] = []
    nonempty = 0
    for summary in summaries:
        ref = str(summary.get("icp_ref") or "")
        if summary.get("_nonempty"):
            nonempty += 1
        if not ref:
            continue
        scores[ref] = _safe_float(summary.get("score"), default=0.0)
        if summary.get("_runtime_error"):
            unresolved.append(ref)
    return scores, unresolved, nonempty


def _per_icp_checkpoint_enabled() -> bool:
    """Bug #31: persist per-ICP candidate results so a requeue/rescore resumes
    instead of re-running the whole multi-hour evaluation."""
    return os.getenv(
        "RESEARCH_LAB_SCORING_PER_ICP_CHECKPOINT", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _scoring_progress_s3_prefix(manifest_uri: str, candidate_id: str) -> tuple[str, str] | None:
    uri = str(manifest_uri or "")
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket or not sep or not key:
        return None
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_candidate = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(candidate_id or "candidate")
    )[:96]
    return bucket, f"{base_prefix}/candidates/{safe_candidate}/scoring-progress/"


def _scoring_progress_s3_location(
    manifest_uri: str, candidate_id: str, window_hash: str
) -> tuple[str, str] | None:
    prefix = _scoring_progress_s3_prefix(manifest_uri, candidate_id)
    if prefix is None:
        return None
    bucket, object_prefix = prefix
    window_tag = str(window_hash or "").removeprefix("sha256:")[:16] or "window"
    return bucket, f"{object_prefix}{window_tag}.json"


def _completed_icp_count_from_progress_doc(doc: Mapping[str, Any] | None) -> int:
    if not isinstance(doc, Mapping):
        return 0
    candidates: list[Any] = [
        doc.get("completed_icp_count"),
        doc.get("completed_icps"),
    ]
    for nested_key in ("scoring_progress", "stale_progress", "progress"):
        nested = doc.get(nested_key)
        if isinstance(nested, Mapping):
            candidates.extend(
                [
                    nested.get("completed_icp_count"),
                    nested.get("completed_icps"),
                ]
            )
    for value in candidates:
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    rows = doc.get("per_icp_results")
    return len(rows) if isinstance(rows, list) else 0


def _safe_scoring_progress_summary(
    *,
    source: str,
    completed_icp_count: int,
    rolling_window_hash: str = "",
) -> dict[str, Any]:
    count = max(0, int(completed_icp_count or 0))
    summary: dict[str, Any] = {
        "source": str(source or "unknown")[:80],
        "completed_icp_count": count,
        "checkpoint_found": count > 0,
    }
    if rolling_window_hash:
        summary["rolling_window_hash"] = str(rolling_window_hash)
    return summary


def _latest_scoring_progress_from_events(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = _safe_scoring_progress_summary(
        source="candidate_events",
        completed_icp_count=0,
    )
    for row in rows:
        doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
        completed = _completed_icp_count_from_progress_doc(doc)
        if completed <= best["completed_icp_count"]:
            continue
        rolling_window_hash = ""
        if isinstance(doc, Mapping):
            progress_doc = doc.get("scoring_progress")
            rolling_window_hash = str(doc.get("rolling_window_hash") or "")
            if not rolling_window_hash and isinstance(progress_doc, Mapping):
                rolling_window_hash = str(progress_doc.get("rolling_window_hash") or "")
        best = _safe_scoring_progress_summary(
            source="candidate_events",
            completed_icp_count=completed,
            rolling_window_hash=rolling_window_hash,
        )
    return best


def _load_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    window_hash: str,
    candidate_artifact_hash: str,
) -> list[dict[str, Any]]:
    import boto3  # type: ignore

    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=object_key)["Body"].read()
        doc = json.loads(body.decode("utf-8"))
    except Exception:
        return []
    if not isinstance(doc, Mapping):
        return []
    if str(doc.get("rolling_window_hash") or "") != str(window_hash):
        return []
    if str(doc.get("candidate_artifact_hash") or "") != str(candidate_artifact_hash):
        return []
    rows = doc.get("per_icp_results")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _load_latest_scoring_progress_summary(
    bucket: str,
    object_prefix: str,
    *,
    candidate_artifact_hash: str = "",
) -> dict[str, Any]:
    import boto3  # type: ignore

    try:
        s3 = boto3.client("s3")
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=object_prefix, MaxKeys=25)
    except Exception:
        return _safe_scoring_progress_summary(source="s3", completed_icp_count=0)
    contents = listing.get("Contents") if isinstance(listing, Mapping) else None
    if not isinstance(contents, list):
        return _safe_scoring_progress_summary(source="s3", completed_icp_count=0)
    ordered = sorted(
        (
            item for item in contents
            if isinstance(item, Mapping) and str(item.get("Key") or "").endswith(".json")
        ),
        key=lambda item: item.get("LastModified") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    for item in ordered:
        object_key = str(item.get("Key") or "")
        try:
            body = s3.get_object(Bucket=bucket, Key=object_key)["Body"].read()
            doc = json.loads(body.decode("utf-8"))
        except Exception:
            continue
        if not isinstance(doc, Mapping):
            continue
        if candidate_artifact_hash and str(doc.get("candidate_artifact_hash") or "") != candidate_artifact_hash:
            continue
        completed = _completed_icp_count_from_progress_doc(doc)
        if completed <= 0:
            continue
        return _safe_scoring_progress_summary(
            source="s3",
            completed_icp_count=completed,
            rolling_window_hash=str(doc.get("rolling_window_hash") or ""),
        )
    return _safe_scoring_progress_summary(source="s3", completed_icp_count=0)


def _store_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    candidate_id: str,
    window_hash: str,
    candidate_artifact_hash: str,
    rows: list[dict[str, Any]],
) -> None:
    import boto3  # type: ignore

    doc = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_candidate_scoring_progress",
        "candidate_id": str(candidate_id),
        "rolling_window_hash": str(window_hash),
        "candidate_artifact_hash": str(candidate_artifact_hash),
        "completed_icp_count": len(rows),
        "per_icp_results": rows,
    }
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=object_key,
        Body=json.dumps(doc, sort_keys=True, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def _baseline_progress_s3_location(
    manifest_uri: str,
    *,
    benchmark_date: str,
    window_hash: str,
    private_model_artifact_hash: str,
) -> tuple[str, str] | None:
    uri = str(manifest_uri or "")
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket or not sep or not key:
        return None
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_date = "".join(
        ch if ch.isdigit() or ch == "-" else "-" for ch in str(benchmark_date or "date")
    )[:32]
    window_tag = str(window_hash or "").removeprefix("sha256:")[:16] or "window"
    artifact_tag = str(private_model_artifact_hash or "").removeprefix("sha256:")[:16] or "model"
    return (
        bucket,
        f"{base_prefix}/baselines/{safe_date}/scoring-progress/{window_tag}-{artifact_tag}.json",
    )


def _benchmark_item_ref_for_progress(item: Mapping[str, Any]) -> str:
    return str(item.get("icp_ref") or item.get("icp_hash") or "")


def _progress_rows_by_icp_ref(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        ref = str(row.get("icp_ref") or row.get("icp_hash") or "")
        if ref:
            indexed[ref] = dict(row)
    return indexed


def _baseline_summary_nonempty(row: Mapping[str, Any]) -> bool:
    for key in ("company_count", "sourced_count", "model_output_count"):
        try:
            if int(row.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    breakdowns = row.get("score_breakdowns")
    return isinstance(breakdowns, list) and len(breakdowns) > 0


def _baseline_summary_checkpointable(row: Mapping[str, Any]) -> bool:
    if row.get("_runtime_error"):
        return False
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        if diagnostics.get("runtime_error"):
            return False
        categories = {str(value) for value in diagnostics.get("failure_categories") or []}
        if categories.intersection({"runtime_provider_error", "scorer_provider_error", "provider_cost_tracking_failed"}):
            return False
        if "provider_cost_cap_blocked" in categories and not _baseline_summary_nonempty(row):
            return False
    return bool(str(row.get("icp_ref") or row.get("icp_hash") or ""))


def _baseline_progress_public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if not str(key).startswith("_")}


def _load_baseline_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    benchmark_date: str,
    window_hash: str,
    private_model_artifact_hash: str,
) -> list[dict[str, Any]]:
    import boto3  # type: ignore

    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=object_key)["Body"].read()
        doc = json.loads(body.decode("utf-8"))
    except Exception:
        return []
    if not isinstance(doc, Mapping):
        return []
    if str(doc.get("benchmark_date") or "") != str(benchmark_date):
        return []
    if str(doc.get("rolling_window_hash") or "") != str(window_hash):
        return []
    if str(doc.get("private_model_artifact_hash") or "") != str(private_model_artifact_hash):
        return []
    rows = doc.get("per_icp_results")
    if not isinstance(rows, list):
        return []
    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and _baseline_summary_checkpointable(row)
    ]


def _store_baseline_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    benchmark_date: str,
    window_hash: str,
    private_model_artifact_hash: str,
    rows: list[dict[str, Any]],
) -> None:
    import boto3  # type: ignore

    safe_rows = [
        _baseline_progress_public_row(row)
        for row in rows
        if isinstance(row, Mapping) and _baseline_summary_checkpointable(row)
    ]
    doc = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_private_baseline_scoring_progress",
        "benchmark_date": str(benchmark_date),
        "rolling_window_hash": str(window_hash),
        "private_model_artifact_hash": str(private_model_artifact_hash),
        "completed_icp_count": len(safe_rows),
        "per_icp_results": safe_rows,
    }
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=object_key,
        Body=json.dumps(doc, sort_keys=True, default=str).encode("utf-8"),
        ContentType="application/json",
    )


# --- §5.4 scorer-judge traces + baseline in-container trace collection ------
# The qualification scorer's per-company judgments are dense reward labels for
# a future Sales LLM; the private models' in-container provider traffic is the
# matching trajectory data. Both captures are pure observation: best-effort,
# pointer-only in any event/bundle doc, and inert without S3 configuration.

_SCORER_TRACE_CAPTURE_ENV = "RESEARCH_LAB_SCORER_TRACE_CAPTURE"
_SCORER_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX"
_TRACE_KMS_KEY_ENV = "RESEARCH_LAB_TRACE_KMS_KEY_ID"
_SCORER_TRACE_PUT_CONNECT_TIMEOUT_SECONDS = 5
_SCORER_TRACE_PUT_READ_TIMEOUT_SECONDS = 15
# Model-output fields safe to persist as scorer-INPUT context: company identity
# only — never page content, evidence text, or contact-level material.
_SCORER_TRACE_COMPANY_IDENTITY_FIELDS = (
    "company_name",
    "company_website",
    "company_linkedin",
    "industry",
    "sub_industry",
    "employee_count",
    "company_stage",
    "city",
    "state",
    "country",
)

# The sourcing model emits some identity fields under different key names
# than the canonical ones above ("subindustry" without the underscore,
# "hq_city"/"hq_state"/"hq_country" for the HQ location).  Read those as
# fallbacks so identity capture doesn't silently null those columns out.
_SCORER_TRACE_IDENTITY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sub_industry": ("sub_industry", "subindustry"),
    "city": ("city", "hq_city"),
    "state": ("state", "hq_state"),
    "country": ("country", "hq_country"),
}


def _scorer_trace_capture_enabled() -> bool:
    return os.getenv(_SCORER_TRACE_CAPTURE_ENV, "true").strip().lower() in {"1", "true", "yes", "on"}


def _trace_kms_key_id() -> str:
    """KMS key used for S3 SSE-KMS trace encryption."""
    return str(os.getenv(_TRACE_KMS_KEY_ENV, "") or "").strip()


def _openrouter_include_reasoning_enabled() -> bool:
    return os.getenv("RESEARCH_LAB_LLM_INCLUDE_REASONING", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _trace_path_segment(value: object, *, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or ""))[:96]
    return text.strip("-.") or fallback


def _scorer_trace_company_identity(output: Any) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        return {}
    identity: dict[str, Any] = {}
    for field in _SCORER_TRACE_COMPANY_IDENTITY_FIELDS:
        for key in _SCORER_TRACE_IDENTITY_FIELD_ALIASES.get(field, (field,)):
            value = str(output.get(key) or "").strip()
            if value:
                identity[field] = value
                break
    return identity


_REJECTED_COMPANIES_CAPTURE_ENV = "RESEARCH_LAB_REJECTED_COMPANIES_CAPTURE"
_REJECTED_COMPANIES_TABLE = "research_lab_rejected_companies"


def _rejected_companies_capture_enabled() -> bool:
    return os.getenv(_REJECTED_COMPANIES_CAPTURE_ENV, "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _optional_score(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


async def _persist_rejected_companies(
    *,
    context_ref: str,
    icp_ref: str,
    icp_hash: str,
    is_reference_model: bool,
    outputs: Any,
    breakdowns: Any,
    candidate_id: str | None = None,
    model_manifest_hash: str | None = None,
) -> int:
    """Best-effort: persist model-sourced companies the harness REJECTED
    (final_score 0 / an explicit failure_reason) to
    ``research_lab_rejected_companies`` for later false-rejection analysis.

    Company identity only (name / site / linkedin / industry / size / country) —
    never page content or evidence text. Dedup: each unique rejection (icp_hash |
    company | failure_reason | model) is stored once via ``dedup_key`` + the
    table's UNIQUE constraint (a conflicting insert is swallowed). Never raises
    and never blocks scoring. Returns the number of rows written.
    """
    if not _rejected_companies_capture_enabled() or not breakdowns:
        return 0
    written = 0
    now = datetime.now(timezone.utc).isoformat()
    try:
        for output, breakdown in zip(outputs or (), breakdowns or ()):
            if not isinstance(breakdown, Mapping):
                continue
            final_score = float(breakdown.get("final_score", 0.0) or 0.0)
            failure_reason = str(breakdown.get("failure_reason") or "").strip()
            if final_score > 0.0 and not failure_reason:
                continue  # accepted / scored — not a rejection
            identity = _scorer_trace_company_identity(output)
            # Dedup identity: SAME company + SAME error + SAME icp -> one row
            # (regardless of baseline vs candidate). Key on the company NAME,
            # normalized to alphanumerics so case / punctuation / spacing / a
            # linkedin-slug present one run but not the next never splits it.
            company_key = "".join(
                ch
                for ch in (
                    identity.get("company_name")
                    or identity.get("company_website")
                    or identity.get("company_linkedin")
                    or ""
                ).lower()
                if ch.isalnum()
            )
            dedup_key = sha256_json(
                {
                    "icp_hash": str(icp_hash or ""),
                    "company": company_key,
                    "failure_reason": failure_reason,
                }
            )
            # Model-side claims: what the model itself asserted about this
            # company (nested intent / required_attribute docs + its own score).
            # Deliberately EXCLUDED: required_attribute.text (sealed-ICP
            # content) and evidence_quote (scraped page text).
            intent_doc = output.get("intent") if isinstance(output.get("intent"), Mapping) else {}
            attr_doc = (
                output.get("required_attribute")
                if isinstance(output.get("required_attribute"), Mapping)
                else {}
            )

            def _opt_text(value: Any, limit: int = 500) -> str | None:
                text = str(value or "").strip()
                return text[:limit] if text else None

            row = {
                "context_ref": str(context_ref),
                "is_reference_model": bool(is_reference_model),
                "candidate_id": (str(candidate_id) if candidate_id else None),
                "model_manifest_hash": (str(model_manifest_hash) if model_manifest_hash else None),
                "icp_ref": str(icp_ref),
                "icp_hash": str(icp_hash or ""),
                "company_name": identity.get("company_name") or None,
                "company_website": identity.get("company_website") or None,
                "company_linkedin": identity.get("company_linkedin") or None,
                "industry": identity.get("industry") or None,
                "sub_industry": identity.get("sub_industry") or None,
                "employee_count": identity.get("employee_count") or None,
                "company_stage": identity.get("company_stage") or None,
                "city": identity.get("city") or None,
                "state": identity.get("state") or None,
                "country": identity.get("country") or None,
                # model claims (for model-confidence vs harness-verdict analysis)
                "model_claimed_score": _optional_score(output.get("score")),
                "intent_source": _opt_text(intent_doc.get("source"), 80),
                "intent_claimed_signal": _opt_text(intent_doc.get("signal"), 200),
                "intent_evidence_url": _opt_text(intent_doc.get("url")),
                "intent_evidence_date": _opt_text(intent_doc.get("date"), 40),
                "attribute_evidence_url": _opt_text(attr_doc.get("evidence_url")),
                # harness outcome
                "final_score": final_score,
                "failure_reason": failure_reason or None,
                "failure_stage": (str(breakdown.get("stage_failed") or "").strip() or None),
                "fit_passed": _optional_bool(breakdown.get("fit_passed")),
                "attribute_passed": _optional_bool(breakdown.get("attribute_passed")),
                "intent_passed": _optional_bool(breakdown.get("intent_passed")),
                "icp_fit": _optional_score(breakdown.get("icp_fit")),
                "intent_signal_raw": _optional_score(breakdown.get("intent_signal_raw")),
                "time_decay_multiplier": _optional_score(breakdown.get("time_decay_multiplier")),
                "intent_signal": _optional_score(breakdown.get("intent_signal_final")),
                "captured_at": now,
                "dedup_key": dedup_key,
            }
            try:
                await insert_row(_REJECTED_COMPANIES_TABLE, row)
                written += 1
            except Exception:
                pass  # duplicate (UNIQUE dedup_key) or transient — best-effort
    except Exception:
        pass
    return written


class _ScorerTraceRecorder:
    """Best-effort SSE-KMS S3 capture of the qualification scorer's per-company
    judgments at the ``score_with_breakdowns`` boundary (§5.4).

    One JSON doc per (context, icp) under
    ``{prefix}/scorer-traces/{context_ref}/{icp_ref}.json`` — retried ICPs
    overwrite the same key, so the freshest attempt wins (matching the batch's
    replace-on-retry semantics). ``{prefix}`` comes from
    ``RESEARCH_LAB_SCORER_TRACE_S3_PREFIX`` or falls back to the private model
    manifest's bucket/prefix (like the scoring-progress writer); with neither
    resolvable the recorder counts-and-drops after one log line. Hard rules,
    mirroring ``_OpenRouterRawTraceRecorder``:

      * a capture failure can NEVER affect scoring — S3 writes are
        fire-and-forget on a small background pool, every synchronous step is
        exception-wrapped, and failures log once per context;
      * event/bundle docs receive ONLY ``{s3_ref, sha256}`` pointers — never
        breakdown content (the protected-material scanners reject content
        keys);
      * docs store scorer INPUTS limited to icp context refs plus company
        identity fields, and the scorer's full per-company breakdowns
        (including any reasoning text the scorer returns);
      * the pointer is optimistic: returned while the write is in flight, so a
        failed write leaves a dangling (never wrong) reference.

    Flag: ``RESEARCH_LAB_SCORER_TRACE_CAPTURE`` (default true; inert without
    S3/KMS config). Encryption: SSE-KMS via ``RESEARCH_LAB_TRACE_KMS_KEY_ID``.
    """

    def __init__(self, config: ResearchLabGatewayConfig):
        self.config = config
        self._lock = threading.Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending: set[concurrent.futures.Future[None]] = set()
        self._failure_logged_contexts: set[str] = set()
        self._destinations: dict[str, tuple[str, str] | None] = {}
        self._drop_logged = False
        self._dropped_docs = 0
        self._disabled = False

    def capture(
        self,
        *,
        context_ref: str,
        icp_ref: str,
        icp_hash: str = "",
        outputs: Any = (),
        breakdowns: Any = (),
        is_reference_model: bool = False,
        manifest_uri: str = "",
    ) -> dict[str, str] | None:
        """Queue one scorer-judgment doc for upload.

        Returns the ``{s3_ref, sha256}`` pointer for event/bundle docs, or None
        when capture is disabled/unconfigured or there is nothing to record.
        Never raises."""
        try:
            if self._disabled or not _scorer_trace_capture_enabled() or not breakdowns:
                return None
            destination = self._resolve_destination(manifest_uri)
            if destination is None:
                self._count_drop(reason="missing_s3_prefix", env_name=_SCORER_TRACE_S3_PREFIX_ENV)
                return None
            kms_key_id = _trace_kms_key_id()
            if not kms_key_id:
                self._count_drop(reason="missing_kms_key", env_name=_TRACE_KMS_KEY_ENV)
                return None
            bucket, key_prefix = destination
            safe_context = _trace_path_segment(context_ref, fallback="context")
            safe_icp = _trace_path_segment(icp_ref, fallback="icp")
            object_key = "/".join(
                segment
                for segment in (key_prefix, "scorer-traces", safe_context, f"{safe_icp}.json")
                if segment
            )
            doc = {
                "schema_version": "1.0",
                "artifact_type": "research_lab_scorer_judgment_trace",
                "context_ref": str(context_ref),
                "icp_ref": str(icp_ref),
                "icp_hash": str(icp_hash or ""),
                "is_reference_model": bool(is_reference_model),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "sourced_count": len(outputs) if outputs else 0,
                "scored_count": len(breakdowns),
                "companies": [_scorer_trace_company_identity(output) for output in (outputs or ())],
                "score_breakdowns": [dict(item) for item in breakdowns if isinstance(item, Mapping)],
            }
            body = canonical_json(doc).encode("utf-8")
            digest = sha256_json(doc)
            self._submit(
                context_ref=str(context_ref),
                bucket=bucket,
                object_key=object_key,
                body=body,
                kms_key_id=kms_key_id,
            )
            return {"s3_ref": f"s3://{bucket}/{object_key}", "sha256": digest}
        except Exception as exc:
            self._log_failure_once(str(context_ref or "unknown"), exc)
            return None

    def flush(self, timeout_seconds: float = 10.0) -> None:
        """Wait for in-flight uploads (tests / orderly teardown only)."""
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            with self._lock:
                pending = tuple(self._pending)
            if not pending or time.monotonic() >= deadline:
                return
            for future in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    future.exception(timeout=remaining)
                except Exception:
                    return

    def _resolve_destination(self, manifest_uri: str) -> tuple[str, str] | None:
        prefix_uri = str(os.getenv(_SCORER_TRACE_S3_PREFIX_ENV, "")).strip().rstrip("/")
        cache_key = prefix_uri or str(manifest_uri or "")
        with self._lock:
            if cache_key in self._destinations:
                return self._destinations[cache_key]
        if not prefix_uri:
            uri = str(manifest_uri or "").strip()
            if uri.startswith("s3://"):
                bucket, _sep, key = uri[5:].partition("/")
                base_prefix = key.rsplit("/", 1)[0] if "/" in key else ""
                if bucket:
                    prefix_uri = f"s3://{bucket}/{base_prefix}".rstrip("/")
        destination: tuple[str, str] | None = None
        if prefix_uri.startswith("s3://"):
            bucket, _sep, key_prefix = prefix_uri[5:].partition("/")
            if bucket:
                destination = (bucket, key_prefix.strip("/"))
        with self._lock:
            self._destinations[cache_key] = destination
        return destination

    def _count_drop(self, *, reason: str, env_name: str) -> None:
        with self._lock:
            self._dropped_docs += 1
            if self._drop_logged:
                return
            self._drop_logged = True
        # Local/dev inertness: no S3 destination — one log, then silent drops.
        logger.info(
            "research_lab_scorer_trace_capture_disabled reason=%s env=%s; "
            "scorer judgment capture is skipped for this process",
            reason,
            env_name,
        )

    def _submit(
        self,
        *,
        context_ref: str,
        bucket: str,
        object_key: str,
        body: bytes,
        kms_key_id: str,
    ) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="research-lab-scorer-trace",
                )
            executor = self._executor
        future = executor.submit(
            self._put_object,
            bucket=bucket,
            object_key=object_key,
            body=body,
            kms_key_id=kms_key_id,
        )
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(lambda done: self._consume_put_result(context_ref, done))

    def _put_object(self, *, bucket: str, object_key: str, body: bytes, kms_key_id: str) -> None:
        import boto3  # type: ignore

        client_kwargs: dict[str, Any] = {}
        try:
            from botocore.config import Config as BotoClientConfig  # type: ignore

            client_kwargs["config"] = BotoClientConfig(
                connect_timeout=_SCORER_TRACE_PUT_CONNECT_TIMEOUT_SECONDS,
                read_timeout=_SCORER_TRACE_PUT_READ_TIMEOUT_SECONDS,
                retries={"max_attempts": 2},
            )
        except Exception:  # pragma: no cover - botocore ships with boto3
            client_kwargs = {}
        put_kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": object_key,
            "Body": body,
            "ContentType": "application/json",
            # SSE-KMS at rest via an ENCRYPT_DECRYPT key, matching the
            # raw-trace recorder's §9.1 encryption requirement.
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": kms_key_id,
        }
        boto3.client("s3", **client_kwargs).put_object(**put_kwargs)

    def _consume_put_result(self, context_ref: str, future: "concurrent.futures.Future[None]") -> None:
        with self._lock:
            self._pending.discard(future)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is None:
            return
        if isinstance(exc, ImportError):
            # boto3 absent: local/dev environment — go inert for the process.
            with self._lock:
                self._disabled = True
        self._log_failure_once(context_ref, exc)

    def _log_failure_once(self, context_ref: str, exc: BaseException) -> None:
        context_key = str(context_ref or "unknown")
        with self._lock:
            if context_key in self._failure_logged_contexts:
                return
            self._failure_logged_contexts.add(context_key)
        logger.warning(
            "research_lab_scorer_trace_capture_failed context=%s error=%s; capture is "
            "best-effort and scoring was not affected",
            compact_ref(context_key),
            _short_error(exc),
        )


class _TraceCapturingCompanyScorer:
    """``QualificationStyleCompanyScorer`` delegate that records each judgment
    through a ``_ScorerTraceRecorder`` (§5.4 candidate-eval scorer boundary).

    Pure observation: breakdowns and derived scores are passed through
    untouched, and the capture step is fully exception-contained — with the
    flag off or no S3 config the delegate behaves exactly like the default
    scorer the evaluator would have constructed itself.
    """

    def __init__(
        self,
        *,
        recorder: _ScorerTraceRecorder,
        context_ref: str,
        manifest_uri: str,
        benchmark_items: Any = (),
        pointer_map: dict[str, dict[str, str]] | None = None,
        inner: Any = None,
        candidate_id: str | None = None,
        candidate_model_manifest_hash: str | None = None,
    ):
        self._inner = inner if inner is not None else QualificationStyleCompanyScorer()
        self._recorder = recorder
        self._context_ref = str(context_ref)
        self._manifest_uri = str(manifest_uri or "")
        self._candidate_id = str(candidate_id) if candidate_id else None
        self._candidate_model_manifest_hash = (
            str(candidate_model_manifest_hash) if candidate_model_manifest_hash else None
        )
        self._pointer_map = pointer_map
        # Per-ICP funnel counts (sourced -> fit -> verified -> intent -> scored)
        # for the candidate model, keyed by icp_ref. Read back onto each per-ICP
        # row via scorer_funnel_for.
        self._funnel_map: dict[str, dict[str, Any]] = {}
        self._evidence_types_map: dict[str, dict[str, Any]] = {}
        # The evaluator hands the scorer the raw ICP payload, not the benchmark
        # item, so refs are recovered via the payload's canonical hash.
        self._icp_refs: dict[str, tuple[str, str]] = {}
        for item in benchmark_items or ():
            try:
                icp = item.get("icp") if isinstance(item, Mapping) else None
                if isinstance(icp, Mapping):
                    self._icp_refs[sha256_json(icp)] = (
                        str(item.get("icp_ref") or item.get("icp_hash") or ""),
                        str(item.get("icp_hash") or ""),
                    )
            except Exception:  # noqa: BLE001 - ref recovery is best-effort
                continue

    async def __call__(
        self,
        companies: Any,
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[float]:
        breakdowns = await self.score_with_breakdowns(companies, icp, is_reference_model)
        return [float(item.get("final_score", 0.0) or 0.0) for item in breakdowns]

    async def score_with_breakdowns(
        self,
        companies: Any,
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        breakdowns = await self._inner.score_with_breakdowns(companies, icp, is_reference_model)
        try:
            icp_key = sha256_json(icp if isinstance(icp, Mapping) else {})
            icp_ref, icp_hash = self._icp_refs.get(icp_key, ("", ""))
            if not icp_ref:
                icp_ref = icp_key.removeprefix("sha256:")[:24]
            pointer = self._recorder.capture(
                context_ref=self._context_ref,
                icp_ref=icp_ref,
                icp_hash=icp_hash,
                outputs=list(companies or ()),
                breakdowns=breakdowns,
                is_reference_model=bool(is_reference_model),
                manifest_uri=self._manifest_uri,
            )
            if pointer and self._pointer_map is not None:
                self._pointer_map[icp_ref] = dict(pointer)
            if not is_reference_model and icp_ref:
                # Isolated from the outer trace-capture guard: a stats failure
                # must not also drop the judgment pointer, and it must log with
                # a distinct tag (a silent failure here blanks the per-ICP
                # funnel for the whole candidate — a real regression once went
                # unnoticed because it shared the generic wrapper warning).
                try:
                    # Per-breakdown intent-signal evidence lists feed the
                    # evidence-type (intent pass rate) panel; absent on the
                    # default scorer, so this stays best-effort.
                    signal_details = [
                        (b.get("intent_signals_detail") or [])
                        if isinstance(b, Mapping) else []
                        for b in (breakdowns or [])
                    ]
                    stats = build_icp_stats(
                        sourced_count=len(companies or ()),
                        breakdowns=breakdowns,
                        signal_details=signal_details if any(signal_details) else None,
                    )
                    funnel = stats.get("funnel")
                    if isinstance(funnel, Mapping):
                        self._funnel_map[icp_ref] = dict(funnel)
                    evidence_types = stats.get("evidence_types")
                    if isinstance(evidence_types, Mapping) and evidence_types:
                        self._evidence_types_map[icp_ref] = dict(evidence_types)
                except Exception:  # noqa: BLE001 - stats never affect scoring
                    logger.warning(
                        "research_lab_scorer_funnel_stats_failed context=%s icp_ref=%s",
                        compact_ref(self._context_ref),
                        compact_ref(icp_ref),
                        exc_info=True,
                    )
            # Persist rejected companies for false-rejection analysis (candidate
            # path). Best-effort; never affects scoring.
            await _persist_rejected_companies(
                context_ref=self._context_ref,
                icp_ref=icp_ref,
                icp_hash=icp_hash,
                is_reference_model=bool(is_reference_model),
                outputs=list(companies or ()),
                breakdowns=breakdowns,
                candidate_id=getattr(self, "_candidate_id", None),
                model_manifest_hash=getattr(self, "_candidate_model_manifest_hash", None),
            )
        except Exception:  # noqa: BLE001 - capture can never affect scoring
            logger.warning(
                "research_lab_scorer_trace_wrapper_failed context=%s",
                compact_ref(self._context_ref),
                exc_info=True,
            )
        return breakdowns

    def scorer_trace_pointer_for(self, icp_ref: str) -> dict[str, str] | None:
        """P12: the evaluator's row builder picks up this ICP's judgment-trace
        pointer so ``scorer_trace_ref`` rides the per-ICP row into the bundle
        (previously the pointers reached only the scored event doc)."""
        if not self._pointer_map:
            return None
        pointer = self._pointer_map.get(str(icp_ref))
        return dict(pointer) if isinstance(pointer, Mapping) else None

    def scorer_funnel_for(self, icp_ref: str) -> dict[str, Any] | None:
        """Return this ICP's candidate funnel counts so they ride the per-ICP
        row into the bundle. Duck-typed; the default scorer has no funnel."""
        funnel = self._funnel_map.get(str(icp_ref))
        return dict(funnel) if isinstance(funnel, Mapping) else None

    def scorer_evidence_types_for(self, icp_ref: str) -> dict[str, Any] | None:
        """Return this ICP's per-evidence-type intent stats (intent pass rate
        panel). Duck-typed; the default scorer has none."""
        evidence = self._evidence_types_map.get(str(icp_ref))
        return dict(evidence) if isinstance(evidence, Mapping) else None


def _upload_baseline_incontainer_trace(
    prefix: str,
    context_ref: str,
    icp_ref: str,
    entries: list[dict[str, Any]],
    kms_key_id: str,
) -> str:
    """Local equivalent of the evaluator's default-sink uploader for the
    baseline batch path, writing ``{prefix}/{context}/baseline/{icp_ref}.json``
    under the same ``RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX`` scheme."""
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise PrivateModelRuntimeError("boto3 is required for in-container trace S3 uploads") from exc
    if not prefix.startswith("s3://"):
        raise PrivateModelRuntimeError(f"invalid in-container trace S3 prefix: {prefix}")
    bucket, _, key_prefix = prefix[5:].partition("/")
    if not bucket:
        raise PrivateModelRuntimeError(f"invalid in-container trace S3 prefix: {prefix}")
    key = "/".join(
        part
        for part in (
            key_prefix.strip("/"),
            _trace_path_segment(context_ref, fallback="context"),
            "baseline",
            f"{_trace_path_segment(icp_ref, fallback='icp')}.json",
        )
        if part
    )
    if not kms_key_id:
        # P5/P13: never PUT in-container trace content without aws:kms.
        raise PrivateModelRuntimeError(
            "in-container trace upload requires a KMS key id (unencrypted uploads are refused)"
        )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_incontainer_trace",
        "run_ref": str(context_ref),
        "icp_ref": str(icp_ref),
        "call_count": len(entries),
        "entries": entries,
    }
    put_kwargs: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": canonical_json(payload).encode("utf-8"),
        "ContentType": "application/json",
        "ServerSideEncryption": "aws:kms",
        "SSEKMSKeyId": kms_key_id,
    }
    boto3.client("s3").put_object(**put_kwargs)
    return f"s3://{bucket}/{key}"


async def _publish_baseline_incontainer_trace(
    *,
    context_ref: str,
    icp_ref: str,
    entries: list[dict[str, Any]],
    drop_state: dict[str, Any],
) -> dict[str, Any]:
    """Persist one baseline/champion ICP's collected in-container trace entries
    and build the POINTER-ONLY diagnostics fields (mirroring the evaluator's
    ``_finalize_incontainer_trace`` semantics: upload failures are logged and
    swallowed). P5: a dropped capture must not look populated — no ref means
    ``incontainer_trace_call_count=0`` plus ``incontainer_trace_dropped=true``;
    the sha256 stays as the attestation the entries existed. Returns ``{}``
    when there is nothing to record."""
    if not entries:
        return {}
    ref = ""
    prefix = str(os.getenv(INCONTAINER_TRACE_S3_PREFIX_ENV) or "").strip().rstrip("/")
    kms_key_id = str(os.getenv(INCONTAINER_TRACE_KMS_KEY_ENV) or "").strip()
    if not prefix or not kms_key_id:
        reason = "no_s3_prefix" if not prefix else "missing_kms_key"
        drop_state["dropped_entries"] = int(drop_state.get("dropped_entries") or 0) + len(entries)
        if not drop_state.get("logged"):
            drop_state["logged"] = True
            log = logger.info if not prefix else logger.error
            log(
                "research_lab_incontainer_trace_dropped run_ref=%s reason=%s "
                "(set %s and %s to persist baseline in-container traces encrypted; "
                "further drops this run are silent)",
                context_ref,
                reason,
                INCONTAINER_TRACE_S3_PREFIX_ENV,
                INCONTAINER_TRACE_KMS_KEY_ENV,
            )
    else:
        try:
            ref = await asyncio.to_thread(
                _upload_baseline_incontainer_trace,
                prefix,
                str(context_ref),
                str(icp_ref),
                list(entries),
                kms_key_id,
            )
        except Exception:  # noqa: BLE001 - capture must never fail the run
            logger.warning(
                "research_lab_incontainer_trace_sink_failed icp_ref=%s entry_count=%s",
                icp_ref,
                len(entries),
                exc_info=True,
            )
            ref = ""
    fields = {
        "incontainer_trace_ref": ref,
        "incontainer_trace_sha256": sha256_json(list(entries)),
        "incontainer_trace_call_count": len(entries) if ref else 0,
    }
    # P13: truncation is filterable from the index, not just inside the blob.
    truncated_count = sum(1 for entry in entries if entry.get("truncated"))
    if truncated_count:
        fields["incontainer_trace_truncated_count"] = truncated_count
    provider_cost_summary = summarize_provider_cost_trace_entries(entries)
    if (
        provider_cost_summary.get("paid_call_count")
        or provider_cost_summary.get("cache_hit_count")
        or provider_cost_summary.get("blocked_call_count")
        or provider_cost_summary.get("tracking_failed_count")
    ):
        fields["provider_cost_summary"] = provider_cost_summary
        fields["provider_cost_total_usd"] = provider_cost_summary.get("total_cost_usd", 0.0)
        fields["provider_cost_paid_call_count"] = provider_cost_summary.get("paid_call_count", 0)
        fields["provider_cost_cap_blocked"] = bool(provider_cost_summary.get("cap_blocked"))
        fields["provider_cost_tracking_failed"] = int(provider_cost_summary.get("tracking_failed_count") or 0) > 0
    if not ref:
        fields["incontainer_trace_dropped"] = True
        fields["incontainer_trace_dropped_call_count"] = len(entries)
    return fields


def _apply_provider_cost_baseline_outcome(item_summary: dict[str, Any]) -> None:
    diagnostics = item_summary.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return
    summary = diagnostics.get("provider_cost_summary")
    if not isinstance(summary, Mapping):
        return
    cap_blocked = bool(summary.get("cap_blocked"))
    tracking_failed = int(summary.get("tracking_failed_count") or 0) > 0
    if not cap_blocked and not tracking_failed:
        return
    mutable_diagnostics = dict(diagnostics)
    categories = set(mutable_diagnostics.get("failure_categories") or [])
    if cap_blocked:
        categories.add("provider_cost_cap_blocked")
        mutable_diagnostics["provider_cost_cap_blocked"] = True
    if tracking_failed:
        categories.add("provider_cost_tracking_failed")
        mutable_diagnostics["provider_cost_tracking_failed"] = True
    mutable_diagnostics["failure_categories"] = sorted(categories)
    if tracking_failed or (cap_blocked and not _baseline_summary_nonempty(item_summary)):
        item_summary["score"] = 0.0
        item_summary["company_count"] = 0
    item_summary["diagnostics"] = mutable_diagnostics


def _provider_cost_cap_state(event: Mapping[str, Any]) -> str:
    if event.get("tracking_failed"):
        return "cost_tracking_failed"
    if event.get("cap_blocked"):
        return "blocked_before_call"
    if event.get("cap_exceeded_after_success"):
        return "exceeded_after_success"
    return "under_cap"


def _is_provider_cost_duplicate_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_provider_cost_events" in message
        and (
            "duplicate key" in message
            or "unique constraint" in message
            or "23505" in message
        )
    )


async def _persist_provider_cost_events(
    *,
    entries: list[dict[str, Any]],
    run_type: str,
    icp_ref: str,
    icp_hash: str = "",
    runner_role: str = "unknown",
    candidate_id: str = "",
    benchmark_date: str = "",
    rolling_window_hash: str = "",
) -> None:
    for entry in entries:
        event = cost_event_from_trace_entry(entry)
        if not event:
            continue
        provider = str(event.get("provider") or "unknown")
        if provider not in {"exa", "or", "sd", "deepline"}:
            provider = "unknown"
        anchored_payload = {
            "event": event,
            "icp_ref": str(icp_ref),
            "icp_hash": str(icp_hash or ""),
            "runner_role": str(entry.get("runner_role") or runner_role or "unknown"),
            "trace_seq": entry.get("seq"),
        }
        anchored_hash = sha256_json(anchored_payload)
        row = {
            "event_id": deterministic_uuid("provider_cost_event", anchored_hash),
            "run_scope": str(event.get("scope") or "unscoped"),
            "run_type": str(run_type or "unknown"),
            "candidate_id": str(candidate_id or "") or None,
            "benchmark_date": str(benchmark_date or "") or None,
            "rolling_window_hash": str(rolling_window_hash or "") or None,
            "icp_ref": str(icp_ref or ""),
            "icp_hash": str(icp_hash or ""),
            "runner_role": str(entry.get("runner_role") or runner_role or "unknown"),
            "provider": provider,
            "endpoint": str(event.get("endpoint") or "")[:200],
            "model": str(event.get("model") or "")[:200],
            "request_fingerprint": str(event.get("request_fingerprint") or "")[:64],
            "status_code": int(event.get("status_code") or 0),
            "billable": bool(event.get("billable")),
            "cost_usd": float(event.get("cost_usd") or 0.0),
            "cost_source": str(event.get("cost_source") or "not_billable")[:80],
            "credits": int(event.get("credits") or 0),
            "prompt_tokens": int(event.get("prompt_tokens") or 0),
            "completion_tokens": int(event.get("completion_tokens") or 0),
            "cap_usd": float(event.get("cap_usd") or 0.0),
            "spent_before_usd": float(event.get("spent_before_usd") or 0.0),
            "spent_after_usd": float(event.get("spent_after_usd") or 0.0),
            "cap_state": _provider_cost_cap_state(event),
            "event_doc": {
                "evidence": str(event.get("evidence") or ""),
                "generation_id": str(event.get("generation_id") or "")[:200],
                "tracking_reason": str(event.get("tracking_reason") or "")[:200],
                "trace_seq": entry.get("seq"),
            },
            "anchored_hash": anchored_hash,
        }
        try:
            await insert_row("research_lab_provider_cost_events", row)
        except Exception as exc:  # noqa: BLE001 - telemetry must not affect scoring
            if _is_provider_cost_duplicate_error(exc):
                logger.debug(
                    "research_lab_provider_cost_event_duplicate run_type=%s icp_ref=%s provider=%s",
                    run_type,
                    compact_ref(icp_ref),
                    provider,
                )
                continue
            logger.warning(
                "research_lab_provider_cost_event_insert_failed run_type=%s icp_ref=%s provider=%s",
                run_type,
                compact_ref(icp_ref),
                provider,
                exc_info=True,
            )


def _baseline_max_unresolved_icps() -> int:
    try:
        return max(0, int(os.getenv("RESEARCH_LAB_BASELINE_MAX_UNRESOLVED_ICPS", "2")))
    except ValueError:
        return 2


def _baseline_max_day_jump_points() -> float | None:
    """Day-over-day quarantine threshold; unset means warn-only."""
    raw = os.getenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", "").strip()
    if not raw:
        return None
    try:
        return abs(float(raw))
    except ValueError:
        return None


def _baseline_min_utc_day_delay_seconds() -> int:
    raw = os.getenv(
        "RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS",
        os.getenv(
            "RESEARCH_LAB_BASELINE_MIN_UTC_DAY_DELAY_SECONDS",
            str(DEFAULT_BASELINE_START_UTC_OFFSET_SECONDS),
        ),
    ).strip()
    try:
        return max(0, min(86399, int(raw)))
    except ValueError:
        return DEFAULT_BASELINE_START_UTC_OFFSET_SECONDS


def _candidate_scoring_quiet_start_utc_seconds() -> int:
    raw = os.getenv(
        "RESEARCH_LAB_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS",
        str(DEFAULT_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS),
    ).strip()
    try:
        return max(0, min(86399, int(raw)))
    except ValueError:
        return DEFAULT_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS


def _utc_seconds_since_day_start(value: datetime) -> int:
    dt = value.astimezone(timezone.utc)
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _candidate_baseline_target_date(now: datetime, *, quiet_start_seconds: int) -> str:
    utc_now = now.astimezone(timezone.utc)
    if _utc_seconds_since_day_start(utc_now) >= quiet_start_seconds:
        return (utc_now.date() + timedelta(days=1)).isoformat()
    return utc_now.date().isoformat()


def _summary_has_unresolved_runtime_error(item_summary: Mapping[str, Any]) -> bool:
    diagnostics = item_summary.get("diagnostics")
    return isinstance(diagnostics, Mapping) and bool(diagnostics.get("runtime_error"))


def _build_baseline_health(
    *,
    per_icp_summaries: list[dict[str, Any]],
    retried: int,
    recovered: int,
    max_unresolved_icps: int,
) -> dict[str, Any]:
    """Observe-only health summary for a completed baseline run.

    Retry-exhausted provider errors are scored as zero ICPs and recorded here
    for audit; they must not reject and re-run the whole benchmark.
    """
    unresolved = sum(
        1 for summary in per_icp_summaries if _summary_has_unresolved_runtime_error(summary)
    )
    return {
        "unresolved_provider_errors": unresolved,
        "gate_passed": unresolved <= max_unresolved_icps,
        "decision": "observe_only",
        "retried": int(retried),
        "recovered": int(recovered),
        "max_unresolved_icps": int(max_unresolved_icps),
    }


class CandidateBaselineNotReady(RuntimeError):
    """Raised when candidate scoring must wait for a matching private baseline."""


class CandidateBaselineWindowChanged(CandidateBaselineNotReady):
    """Raised when an in-flight candidate is scoring against an old ICP window."""

    def __init__(
        self,
        *,
        candidate_window_hash: str,
        current_window_hash: str,
        progress: Mapping[str, Any],
    ) -> None:
        self.candidate_window_hash = candidate_window_hash
        self.current_window_hash = current_window_hash
        self.progress = dict(progress)
        completed = int(self.progress.get("completed_icp_count") or 0)
        super().__init__(
            "rolling ICP window changed during candidate scoring: "
            f"candidate_window={compact_ref(candidate_window_hash)} "
            f"current_window={compact_ref(current_window_hash)} "
            f"completed_icps={completed}"
        )


class ClaimLostDuringScoring(RuntimeError):
    """Raised at an ICP boundary when this worker's claim was lost mid-scoring."""


# Known-terminal error classes: validation / malformed-candidate style failures
# that will fail identically on every retry. Everything else defaults to
# retryable (bug #10) — the claim-attempt cap remains the loop guard.
_TERMINAL_CANDIDATE_ERROR_CLASSES = (
    CodeEditBuildError,
    CodeEditPatchApplyError,
    ValueError,
    KeyError,
    TypeError,
)


def _candidate_scoring_failure_class(exc: BaseException) -> tuple[str, bool]:
    text = f"{exc.__class__.__name__}: {str(exc)}"
    lowered = text.lower()
    if isinstance(exc, CandidateBaselineNotReady) or "matching_completed_private_baseline_required" in lowered:
        return "baseline_not_ready", True
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timed out" in lowered or "timeout" in lowered:
        return "adapter_timeout", True
    if any(
        marker in lowered
        for marker in (
            "docker daemon",
            "no space left on device",
            "failed to prepare",
            "failed to solve",
            "exit status 137",
            "killed",
            "manifest unknown",
        )
    ):
        return "infra_docker_error", True
    diagnostics = _runtime_error_diagnostics(text)
    category = str(diagnostics.get("category") or "")
    provider = str(diagnostics.get("provider") or "unknown")
    status = int(diagnostics.get("status") or 0)
    provider_like = provider != "unknown" or status > 0 or any(
        marker in lowered
        for marker in (
            "provider-backed sourcing failed",
            "scrapingdog",
            "openrouter",
            "exa",
            "internal server error",
            "too many requests",
            "rate limit",
        )
    )
    if not provider_like:
        if isinstance(exc, _TERMINAL_CANDIDATE_ERROR_CLASSES):
            return "candidate_scoring_error", False
        if isinstance(exc, PrivateModelRuntimeError):
            # Model code crashed without provider markers: the candidate's own
            # code is at fault and will crash again.
            return "candidate_runtime_error", False
        # Unknown infra/environment failures (window rotation, KMS/S3 signing,
        # PostgREST resets) default retryable; the attempt cap bounds loops.
        return "candidate_scoring_error", True
    if category in {"provider_http_5xx", "runtime_provider_error"}:
        return category, True
    if category == "provider_http_4xx":
        # Same status-level verdicts as the baseline classifier: 408/429 and
        # Scrapingdog 400s retry; genuine auth/request/410 errors stay terminal.
        return category, _baseline_error_is_retryable(text)
    if isinstance(exc, PrivateModelRuntimeError):
        return "candidate_runtime_error", False
    return "candidate_scoring_error", True


def _load_candidate_source_diff(candidate: Mapping[str, Any]) -> str:
    build_doc = candidate.get("candidate_build_doc")
    if not isinstance(build_doc, Mapping):
        raise CodeEditBuildError("stale candidate is missing candidate_build_doc")
    uri = str(build_doc.get("source_diff_artifact_uri") or "")
    if not uri:
        raise CodeEditBuildError("stale candidate has no private source diff artifact")
    expected_source_diff_hash = str(
        candidate.get("candidate_source_diff_hash")
        or build_doc.get("source_diff_hash")
        or ""
    )
    payload = _load_private_json_artifact(uri)
    unified_diff = str(payload.get("unified_diff") or "")
    if not unified_diff.strip():
        raise CodeEditBuildError("private source diff artifact is missing unified_diff")
    actual_source_diff_hash = sha256_json({"unified_diff": unified_diff})
    if expected_source_diff_hash and actual_source_diff_hash != expected_source_diff_hash:
        raise CodeEditBuildError(
            "private source diff artifact hash mismatch: "
            f"expected={compact_ref(expected_source_diff_hash)} actual={compact_ref(actual_source_diff_hash)}"
        )
    return unified_diff


def _stale_parent_rebase_depth(candidate: Mapping[str, Any]) -> int:
    build_doc = candidate.get("candidate_build_doc")
    if not isinstance(build_doc, Mapping):
        return 0
    rebase_doc = build_doc.get("stale_parent_rebase")
    if not isinstance(rebase_doc, Mapping):
        return 0
    try:
        return max(1, int(rebase_doc.get("depth") or 1))
    except (TypeError, ValueError):
        return 1


def _stale_parent_progress_doc(progress: Mapping[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "phase": str(progress.get("phase") or "")[:80],
        "completed_icp_count": int(progress.get("completed_icp_count") or 0),
    }
    for key in ("next_icp_index", "last_icp_index"):
        if key in progress:
            try:
                doc[key] = int(progress.get(key) or 0)
            except (TypeError, ValueError):
                pass
    for key in ("icp_ref", "icp_hash"):
        value = str(progress.get(key) or "").strip()
        if value:
            doc[key] = value[:160]
    return doc


def _candidate_baseline_wait_event_doc(exc: BaseException) -> dict[str, Any]:
    if not isinstance(exc, CandidateBaselineWindowChanged):
        return {}
    return {
        "baseline_wait_reason": "rolling_window_changed_during_candidate_scoring",
        "candidate_window_hash": exc.candidate_window_hash,
        "current_window_hash": exc.current_window_hash,
        "stale_window_progress": _stale_parent_progress_doc(exc.progress),
    }


def _load_private_json_artifact(uri: str) -> dict[str, Any]:
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        try:
            import boto3  # type: ignore
        except Exception as exc:
            raise CodeEditBuildError("boto3 is required to load private source diff artifacts") from exc
        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read().decode("utf-8")
    else:
        raw = Path(uri).expanduser().read_text(encoding="utf-8")
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise CodeEditBuildError("private source diff artifact must be a JSON object")
    text = json.dumps(decoded, sort_keys=True).lower()
    if any(marker in text for marker in ("sk-or-", "service_role", "openrouter_api_key", "raw_secret")):
        raise CodeEditBuildError("private source diff artifact contains forbidden secret-like material")
    return dict(decoded)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    raw = str(uri or "")
    if not raw.startswith("s3://"):
        raise CodeEditBuildError("expected s3:// URI")
    without_scheme = raw[5:]
    bucket, sep, key = without_scheme.partition("/")
    if not bucket or not sep or not key:
        raise CodeEditBuildError("invalid s3 URI")
    return bucket, key


def _extract_diff_paths_safe(unified_diff: str) -> set[str]:
    try:
        return extract_unified_diff_paths(unified_diff)
    except Exception:
        return set()


async def _call_operator_openrouter_json(
    *,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout_seconds: int,
    empty_content_attempts: int = STALE_PARENT_REBASE_REPAIR_EMPTY_CONTENT_ATTEMPTS,
) -> str:
    """Operator stale-parent repair/rebase call, via the shared telemetry
    transport (trajectoryimprovements.md P1: this path used to request
    reasoning, pay for it, then extract only ``message.content`` — the
    reasoning trace is now captured to encrypted S3 instead of discarded)."""
    from research_lab.openrouter_telemetry import (
        OpenRouterTelemetryError,
        call_openrouter_chat_async,
    )

    attempts = max(1, int(empty_content_attempts or 1))
    for attempt in range(1, attempts + 1):
        include_reasoning = attempt == 1
        try:
            result = await call_openrouter_chat_async(
                api_key=api_key,
                model_id=model_id,
                messages=messages,
                channel="research_lab_operator",
                purpose="operator_stale_parent_repair",
                stage="operator_repair" if include_reasoning else "operator_repair_no_reasoning",
                max_tokens=STALE_PARENT_REBASE_REPAIR_MAX_TOKENS,
                temperature=0.1,
                timeout_seconds=max(1, int(timeout_seconds)),
                response_format={"type": "json_object"},
                include_reasoning=include_reasoning,
                extra_body={"reasoning": STALE_PARENT_REBASE_REPAIR_REASONING_BODY}
                if include_reasoning
                else None,
            )
        except OpenRouterTelemetryError as exc:
            raise CodeEditBuildError(f"operator stale-parent repair failed: {exc}") from exc
        content = str(result.content or "").strip()
        if content:
            return content
        if attempt < attempts:
            logger.warning(
                "research_lab_stale_parent_repair_empty_content_retry attempt=%s/%s fallback_no_reasoning=%s",
                attempt,
                attempts,
                attempt == 1,
            )
            continue
    raise CodeEditBuildError(
        f"operator stale-parent repair returned empty content after {attempts} attempt(s)"
    )


def _status_datetime(raw_status_at: object) -> datetime | None:
    if not raw_status_at:
        return None
    text = str(raw_status_at).strip().replace("Z", "+00:00")
    try:
        status_at = datetime.fromisoformat(text)
    except ValueError:
        match = _POSTGREST_TIMESTAMP_RE.match(text)
        if not match:
            return None
        suffix = match.group("suffix") or ""
        if suffix == "Z":
            suffix = "+00:00"
        elif re.fullmatch(r"[+-]\d{2}", suffix):
            suffix = f"{suffix}:00"
        elif re.fullmatch(r"[+-]\d{4}", suffix):
            suffix = f"{suffix[:3]}:{suffix[3:]}"
        fraction = (match.group("fraction") + "000000")[:6]
        try:
            status_at = datetime.fromisoformat(f"{match.group('prefix')}.{fraction}{suffix}")
        except ValueError:
            return None
    if status_at.tzinfo is None:
        status_at = status_at.replace(tzinfo=timezone.utc)
    return status_at.astimezone(timezone.utc)


def _status_age_seconds(raw_status_at: object) -> float | None:
    status_at = _status_datetime(raw_status_at)
    if status_at is None:
        return None
    return (datetime.now(timezone.utc) - status_at).total_seconds()


def _status_is_stale(raw_status_at: object, stale_after_seconds: int) -> bool:
    age_seconds = _status_age_seconds(raw_status_at)
    return age_seconds is not None and age_seconds > max(60, int(stale_after_seconds))


def _claim_predates_worker_boot(
    row: Mapping[str, Any],
    *,
    worker_ref: str,
    worker_started_at: datetime,
    grace_seconds: int,
) -> bool:
    status_at = _status_datetime(row.get("current_status_at"))
    return (
        row.get("current_evaluator_ref") == worker_ref
        and status_at is not None
        and status_at
        < worker_started_at - timedelta(seconds=max(0, int(grace_seconds)))
    )


def _stale_claim_recovery_owner_index(candidate_id: str, total_workers: int) -> int:
    total = max(1, int(total_workers or 1))
    digest = hashlib.sha256(str(candidate_id).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % total


def _candidate_claim_recovery_reason(
    row: Mapping[str, Any],
    *,
    candidate_id: str,
    worker_ref: str,
    worker_index: int,
    total_workers: int,
    worker_started_at: datetime,
    stale_after_seconds: int,
    restart_orphan_grace_seconds: int,
) -> str | None:
    if not candidate_id:
        return None
    if _status_is_stale(row.get("current_status_at"), stale_after_seconds):
        owner_index = _stale_claim_recovery_owner_index(candidate_id, total_workers)
        return "stale_claim" if int(worker_index) == owner_index else None
    if _claim_predates_worker_boot(
        row,
        worker_ref=worker_ref,
        worker_started_at=worker_started_at,
        grace_seconds=restart_orphan_grace_seconds,
    ):
        return "restart_orphan"
    return None


# Requeue reasons that mean the claim cycle only waited (no scoring happened);
# such cycles must not burn the retry budget (bug #6).
_CLAIM_WAIT_REASONS = frozenset({"baseline_not_ready"})


def _count_claim_attempts(rows: list[Mapping[str, Any]]) -> int:
    """Count genuine scoring attempts from seq-ascending assigned/queued events.

    Each ``assigned`` event opens one attempt. A requeue that closes the cycle
    does not count again (the old counter double-charged stale requeues), and a
    cycle that ended waiting (``baseline_not_ready``) is refunded entirely.
    """
    attempts = 0
    open_cycle = False
    for row in rows:
        event_type = str(row.get("event_type") or "")
        if event_type == "assigned":
            attempts += 1
            open_cycle = True
        elif event_type == "queued" and open_cycle:
            if str(row.get("reason") or "") in _CLAIM_WAIT_REASONS:
                attempts -= 1
            open_cycle = False
    return max(0, attempts)


def _worker_can_claim_candidate_slot(worker_index: int, max_active_claims: int) -> bool:
    """Return whether this scoring worker may start new heavy candidate claims.

    ``max_active_claims=0`` disables the slot gate. Otherwise only the first N
    scoring worker slots can claim new candidates, preventing a large worker
    fleet from stampeding one gateway host with simultaneous Docker evaluations.
    """
    try:
        cap = int(max_active_claims)
    except (TypeError, ValueError):
        cap = 0
    if cap <= 0:
        return True
    try:
        index = int(worker_index)
    except (TypeError, ValueError):
        index = 0
    return index < cap


def _active_claim_capacity_available(active_count: int, max_active_claims: int) -> bool:
    try:
        cap = int(max_active_claims)
    except (TypeError, ValueError):
        cap = 0
    if cap <= 0:
        return True
    try:
        count = int(active_count)
    except (TypeError, ValueError):
        count = 0
    return count < cap


def _scoring_host_pressure_capacity(
    *,
    min_available_memory_mb: int,
    max_load_per_cpu: float,
    available_memory_mb: int | None = None,
    load_per_cpu: float | None = None,
) -> dict[str, Any]:
    """Return whether this host is healthy enough to start another heavy score."""
    try:
        memory_floor = max(0, int(min_available_memory_mb or 0))
    except (TypeError, ValueError):
        memory_floor = 0
    try:
        load_ceiling = max(0.0, float(max_load_per_cpu or 0.0))
    except (TypeError, ValueError):
        load_ceiling = 0.0
    result: dict[str, Any] = {
        "available": True,
        "min_available_memory_mb": memory_floor,
        "max_load_per_cpu": load_ceiling,
    }
    if available_memory_mb is not None:
        result["available_memory_mb"] = int(available_memory_mb)
    if load_per_cpu is not None:
        result["load_per_cpu"] = round(float(load_per_cpu), 3)
    if memory_floor > 0 and available_memory_mb is not None and available_memory_mb < memory_floor:
        return {**result, "available": False, "reason": "host_memory_pressure"}
    if load_ceiling > 0 and load_per_cpu is not None and load_per_cpu > load_ceiling:
        return {**result, "available": False, "reason": "host_load_pressure"}
    return result


def _read_mem_available_mb(meminfo_path: str = "/proc/meminfo") -> int | None:
    try:
        with open(meminfo_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                return int(parts[1]) // 1024
    except Exception:
        return None
    return None


def _load_average_per_cpu() -> float | None:
    try:
        cpu_count = max(1, int(os.cpu_count() or 1))
        return float(os.getloadavg()[0]) / float(cpu_count)
    except Exception:
        return None


class ResearchLabGatewayScoringWorker:
    """Scores Research Lab candidates inside the gateway trust boundary."""

    def __init__(self, config: ResearchLabGatewayConfig, *, worker_ref: str | None = None):
        config.validate_public_benchmark_split()
        self.config = config
        self.worker_ref = worker_ref or config.scoring_worker_id or "research-lab-scoring-worker"
        self.proxy_url = config.scoring_worker_proxy_url or os.getenv("RESEARCH_LAB_SCORING_WORKER_PROXY", "")
        self.proxy_ref_hash = canonical_hash({"proxy_ref": self.proxy_url}) if self.proxy_url else None
        self._baseline_skip_logged = False
        self._baseline_already_logged_date: str | None = None
        self._candidate_start_hold_logged_key: str | None = None
        self._last_candidate_start_gate: dict[str, Any] | None = None
        self._private_scoring_env_not_ready_logged = False
        self._resolved_epoch_cache: tuple[int, float] | None = None
        self._scorer_trace_recorder = _ScorerTraceRecorder(config)
        self._confirmation_trace_scope: dict[str, Any] | None = None
        self._worker_started_at = datetime.now(timezone.utc)
        self._last_private_source_push_reconcile_at = 0.0
        self._stale_parent_overdue_warning_keys: set[str] = set()

    async def run_forever(self) -> None:
        # trajectoryimprovements.md P5: one structured capture health block at
        # startup; refuses to start in production when capture is degraded.
        from gateway.research_lab.capture_health import enforce_capture_health

        enforce_capture_health(self.config, worker_kind="scoring_worker")
        last_idle_log = 0.0
        last_error_log = 0.0
        idle_log_seconds = _idle_log_seconds()
        error_backoff_seconds = _error_backoff_seconds()
        while True:
            try:
                outcome = await self.run_once()
            except Exception as exc:
                now = time.monotonic()
                if now - last_error_log >= idle_log_seconds:
                    logger.error(
                        format_worker_block(
                            "RESEARCH LAB SCORING WORKER PASS FAILED",
                            (
                                ("Worker", self.worker_ref),
                                ("Error", _short_error(exc)),
                            ),
                        )
                    )
                    last_error_log = now
                await asyncio.sleep(max(self.config.scoring_worker_poll_seconds, error_backoff_seconds))
                continue
            if outcome.get("processed") or outcome.get("status") != "idle":
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB SCORING WORKER PASS",
                        (
                            ("Worker", self.worker_ref),
                            ("Status", outcome.get("status")),
                            ("Candidates", len(outcome.get("candidate_ids") or [])),
                            (
                                "Baseline status",
                                (outcome.get("baseline") or {}).get("status")
                                if isinstance(outcome.get("baseline"), Mapping)
                                else None,
                            ),
                        ),
                    )
                )
            elif time.monotonic() - last_idle_log >= idle_log_seconds:
                logger.info(
                    "Research Lab scoring worker idle: worker_ref=%s poll_seconds=%s",
                    self.worker_ref,
                    self.config.scoring_worker_poll_seconds,
                )
                last_idle_log = time.monotonic()
            await asyncio.sleep(max(1, self.config.scoring_worker_poll_seconds))

    async def run_once(self) -> dict[str, Any]:
        if not self.config.scoring_worker_enabled:
            return {"processed": False, "status": "disabled"}
        if not self.config.production_writes_enabled or not self.config.evaluation_bundles_enabled:
            return {"processed": False, "status": "writes_or_eval_disabled"}
        if self.config.scoring_worker_require_proxy and not self.proxy_url:
            return {"processed": False, "status": "scoring_worker_proxy_required"}
        if await is_scoring_maintenance_paused():
            return {"processed": False, "status": "maintenance_paused"}

        missing_private_env = self._missing_private_scoring_env()
        if missing_private_env:
            if not self._private_scoring_env_not_ready_logged:
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB SCORING WORKER PRIVATE MODEL ENV NOT READY",
                        (
                            ("Worker", self.worker_ref),
                            ("Missing", ", ".join(missing_private_env)),
                            ("Action", "leaving queued candidates untouched"),
                        ),
                    )
                )
                self._private_scoring_env_not_ready_logged = True
            return {
                "processed": False,
                "status": "idle",
                "private_model_env_ready": False,
                "missing_private_model_env": list(missing_private_env),
            }
        if self._private_scoring_env_not_ready_logged:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB SCORING WORKER PRIVATE MODEL ENV READY",
                    (
                        ("Worker", self.worker_ref),
                        ("Action", "candidate scoring enabled"),
                    ),
                )
            )
            self._private_scoring_env_not_ready_logged = False

        await self._recover_stale_candidate_claims()
        await self._alert_stuck_candidates()

        baseline_result = None
        if self.config.private_baseline_rebenchmark_enabled and self._is_private_baseline_owner():
            baseline_result = await self._maybe_run_private_baseline()
        elif self.config.private_baseline_rebenchmark_enabled and not self._baseline_skip_logged:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE SKIPPED",
                    (
                        ("Worker", self.worker_ref),
                        ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                        ("Owner worker index", 1),
                        ("Proxy ref", self.proxy_ref_hash),
                    ),
                )
            )
            self._baseline_skip_logged = True

        # §5.2-2: run at most one pending confirmation measurement per pass.
        # Any worker may pick these up (leased per candidate via
        # confirmation_rerun_started events). Best-effort: a confirmation
        # failure must never take down the scoring pass.
        confirmation_result: dict[str, Any] | None = None
        if promotion_confirmation_rerun_enabled():
            try:
                confirmation_result = await self._maybe_run_pending_confirmation()
            except Exception as exc:  # noqa: BLE001 - never fail the worker pass
                logger.warning(
                    "research_lab_confirmation_pass_failed worker_ref=%s error=%s",
                    self.worker_ref,
                    _short_error(exc),
                )

        private_source_reconcile_result: dict[str, Any] | None = None
        if self.config.scoring_worker_index == 0 and self.config.auto_promotion_enabled:
            interval = max(1, private_source_push_retry_seconds())
            now = time.monotonic()
            if now - self._last_private_source_push_reconcile_at >= interval:
                self._last_private_source_push_reconcile_at = now
                try:
                    private_source_reconcile_result = await reconcile_failed_private_source_pushes(
                        self.config,
                        worker_ref=self.worker_ref,
                        dry_run=False,
                    )
                except Exception as exc:  # noqa: BLE001 - never fail the scoring pass
                    logger.warning(
                        "research_lab_private_source_push_reconcile_pass_failed worker_ref=%s error=%s",
                        self.worker_ref,
                        _short_error(exc),
                    )
                stale_rebase_ids = [
                    str(item.get("candidate_id") or "")
                    for item in (private_source_reconcile_result or {}).get("results", [])
                    if item.get("stale_parent_rebase_eligible")
                ]
                stale_rebase_ids = [candidate_id for candidate_id in stale_rebase_ids if candidate_id]
                if stale_rebase_ids:
                    try:
                        from .recovery import rebase_stale_parent_candidates as rebase_reconciled_stale_parents

                        rebase_result = await rebase_reconciled_stale_parents(
                            candidate_ids=stale_rebase_ids,
                            dry_run=False,
                            max_batch_size=len(stale_rebase_ids),
                            actor_ref=self.worker_ref,
                        )
                        private_source_reconcile_result["stale_parent_rebase_result"] = rebase_result
                    except Exception as exc:  # noqa: BLE001 - never fail the scoring pass
                        private_source_reconcile_result["stale_parent_rebase_error"] = _short_error(exc)
                        logger.warning(
                            "research_lab_private_source_push_stale_parent_rebase_failed worker_ref=%s candidates=%s error=%s",
                            self.worker_ref,
                            ",".join(compact_ref(candidate_id) for candidate_id in stale_rebase_ids),
                            _short_error(exc),
                        )

        processed: list[str] = []
        claim_capacity: dict[str, Any] = {"available": True}
        self._last_candidate_start_gate = None
        for _ in range(max(1, self.config.scoring_worker_max_candidates)):
            claim_capacity = await self._candidate_claim_capacity()
            if not claim_capacity.get("available"):
                reason = str(claim_capacity.get("reason") or "")
                if reason == "active_claim_capacity_full":
                    logger.info(
                        "research_lab_candidate_claim_capacity_full worker_ref=%s active_claims=%s max_active_claims=%s",
                        self.worker_ref,
                        claim_capacity.get("active_claims"),
                        claim_capacity.get("max_active_claims"),
                    )
                elif reason in {"host_memory_pressure", "host_load_pressure"}:
                    logger.info(
                        "research_lab_candidate_claim_host_pressure worker_ref=%s reason=%s available_memory_mb=%s min_available_memory_mb=%s load_per_cpu=%s max_load_per_cpu=%s",
                        self.worker_ref,
                        reason,
                        claim_capacity.get("available_memory_mb"),
                        claim_capacity.get("min_available_memory_mb"),
                        claim_capacity.get("load_per_cpu"),
                        claim_capacity.get("max_load_per_cpu"),
                    )
                break
            if global_icp_queue.global_icp_queue_enabled():
                # Global (candidate, icp) queue path: enqueue queued candidates'
                # jobs and score from one shared job pool so the pool stays
                # saturated (no per-candidate tail-gap idling). The
                # per-candidate path below is untouched and unused while on.
                queue_processed = await self._run_global_icp_queue_pass()
                processed.extend(queue_processed)
                break
            candidate = await self._claim_next_candidate()
            if not candidate:
                break
            await self._score_candidate(candidate)
            processed.append(str(candidate["candidate_id"]))

        baseline_completed = (
            isinstance(baseline_result, Mapping)
            and str(baseline_result.get("status") or "") == "completed"
        )
        confirmation_processed = (
            isinstance(confirmation_result, Mapping)
            and bool(confirmation_result.get("processed"))
        )
        private_source_reconcile_processed = (
            isinstance(private_source_reconcile_result, Mapping)
            and (
                int(private_source_reconcile_result.get("retried") or 0) > 0
                or int(private_source_reconcile_result.get("finalized") or 0) > 0
            )
        )
        if processed:
            status = "processed"
        elif baseline_completed:
            status = "baseline_completed"
        elif confirmation_processed:
            status = "confirmation_processed"
        elif private_source_reconcile_processed:
            status = "private_source_reconciled"
        elif not claim_capacity.get("available"):
            status = str(claim_capacity.get("reason") or "candidate_claim_capacity_limited")
        elif self._last_candidate_start_gate and not self._last_candidate_start_gate.get("available"):
            status = str(
                self._last_candidate_start_gate.get("reason")
                or "candidate_scoring_daily_baseline_hold"
            )
        else:
            status = "idle"
        return {
            "processed": bool(
                processed
                or baseline_completed
                or confirmation_processed
                or private_source_reconcile_processed
            ),
            "status": status,
            "candidate_ids": processed,
            "baseline": baseline_result,
            "confirmation": confirmation_result,
            "private_source_reconcile": private_source_reconcile_result,
            "candidate_claim_capacity": claim_capacity,
            "candidate_start_gate": self._last_candidate_start_gate,
        }

    async def _candidate_claim_capacity(self) -> dict[str, Any]:
        host_pressure = _scoring_host_pressure_capacity(
            min_available_memory_mb=getattr(self.config, "scoring_worker_min_available_memory_mb", 0),
            max_load_per_cpu=getattr(self.config, "scoring_worker_max_load_per_cpu", 0.0),
            available_memory_mb=_read_mem_available_mb(),
            load_per_cpu=_load_average_per_cpu(),
        )
        if not host_pressure.get("available"):
            return host_pressure
        max_active = int(getattr(self.config, "scoring_worker_max_active_claims", 0) or 0)
        if max_active <= 0:
            return {
                "available": True,
                "max_active_claims": 0,
                "cap_disabled": True,
                "host_pressure": host_pressure,
            }
        if not _worker_can_claim_candidate_slot(self.config.scoring_worker_index, max_active):
            return {
                "available": False,
                "reason": "claim_slot_disabled",
                "worker_index": self.config.scoring_worker_index,
                "max_active_claims": max_active,
                "host_pressure": host_pressure,
            }
        active = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status,current_status_at,current_evaluator_ref",
            filters=(("current_candidate_status", "in", ("assigned", "evaluating")),),
            order_by=(("current_status_at", False),),
            limit=max_active,
        )
        active_count = len(active)
        return {
            "available": _active_claim_capacity_available(active_count, max_active),
            "reason": "" if active_count < max_active else "active_claim_capacity_full",
            "active_claims": active_count,
            "max_active_claims": max_active,
            "worker_index": self.config.scoring_worker_index,
            "host_pressure": host_pressure,
        }

    async def _build_queue_candidate_context(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Per-candidate scoring context for the global-queue path.

        Mirrors the setup block in ``_score_candidate`` (artifact/patch/window/
        gate/runner/run_context) and additionally splits the window into public
        and private ICP items using the gate's public refs. Cached per
        candidate within one queue pass.
        """
        candidate_id = str(candidate["candidate_id"])
        if str(candidate.get("candidate_kind") or "") != "image_build":
            raise RuntimeError("global queue requires image_build candidate_kind")
        evaluation_epoch = await self._resolve_evaluation_epoch()
        artifact = PrivateModelArtifactManifest.from_mapping(candidate["private_model_manifest_doc"])
        patch = candidate["candidate_patch_manifest"]
        candidate_manifest_doc = candidate.get("candidate_model_manifest_doc")
        if not isinstance(candidate_manifest_doc, Mapping):
            raise RuntimeError("image_build candidate missing candidate_model_manifest_doc")
        candidate_artifact = PrivateModelArtifactManifest.from_mapping(candidate_manifest_doc)
        window, gate = await self._daily_candidate_scoring_window_and_gate(artifact=artifact)
        await create_rolling_icp_window(window)
        benchmark = SealedBenchmarkSet(
            benchmark_id=window.benchmark_id,
            icp_set_hash=window.window_hash,
            split_ref=window.split_ref,
            item_refs=window.item_refs,
            scoring_version="qualification-company-scorer:v1",
            hidden_plaintext_available=True,
        )
        runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=candidate_artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_scoring_env(),
                    run_type="candidate_scoring",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=candidate_artifact.model_artifact_hash,
                    candidate_id=candidate_id,
                    run_id=str(candidate.get("run_id") or ""),
                    ticket_id=str(candidate.get("ticket_id") or ""),
                    evaluation_epoch=evaluation_epoch,
                    started_at=time.time(),
                ),
            )
        )
        run_context = self._candidate_run_context(
            candidate, window_hash=window.window_hash, evaluation_epoch=evaluation_epoch
        )
        public_refs = {str(r) for r in (gate.get("public_icp_refs") or ()) if str(r).strip()}

        def _ref(item: Mapping[str, Any]) -> str:
            return str(item.get("icp_ref") or item.get("icp_hash") or "")

        items = list(window.benchmark_items)
        public_items = [it for it in items if _ref(it) in public_refs]
        private_items = [it for it in items if _ref(it) not in public_refs]
        return {
            "candidate": dict(candidate),
            "candidate_id": candidate_id,
            "artifact": artifact,
            "patch": patch,
            "candidate_artifact": candidate_artifact,
            "window": window,
            "gate": gate,
            "benchmark": benchmark,
            "runner": runner,
            "run_context": run_context,
            "scorer": QualificationStyleCompanyScorer(),
            "public_items": public_items,
            "private_items": private_items,
            "items_by_ref": {_ref(it): it for it in items},
            "baseline_public_score": float(gate.get("baseline_public_score") or 0.0),
            "trace_sink": self._candidate_incontainer_trace_sink(candidate_id),
        }

    async def _queue_assemble_candidate(
        self, candidate_id: str, docs: Mapping[str, Any], ctx: Mapping[str, Any]
    ) -> None:
        """Assemble a fully-scored candidate into a signed score bundle and
        write its scored events. Uses the same shared assembler and bundle
        builder as the per-candidate path, so the bundle is byte-identical."""
        _results, gate_result = build_holdout_gate_result(
            public_results=docs.get("public") or (),
            private_results=docs.get("private") or (),
            public_icp_count=len(ctx["public_items"]),
            private_icp_count=len(ctx["private_items"]),
            gate=ctx["gate"],
        )
        score_bundle = build_score_bundle_from_scored_icps(
            artifact_manifest=ctx["artifact"],
            benchmark=ctx["benchmark"],
            patch_manifest=ctx["patch"],
            candidate_artifact_manifest=ctx["candidate_artifact"].to_dict(),
            per_icp_results=_results,
            run_context={**ctx["run_context"], "signature_ref": "pending"},
            policy={},
            extra_bundle_fields={"private_holdout_gate": gate_result},
        )
        scoring_health_gate = self._scoring_health_gate_result(score_bundle)
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=str(score_bundle["score_bundle_hash"]),
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        score_bundle = {**score_bundle, "signature_ref": signature_ref}
        candidate = ctx["candidate"]
        bundle, _bundle_event = await create_score_bundle(
            ResearchLabScoreBundleCreateRequest(
                bundle_status="scored",
                receipt_id=candidate.get("receipt_id") or None,
                score_bundle=score_bundle,
            )
        )
        await self._create_scored_evaluation_event(
            candidate=candidate,
            candidate_id=candidate_id,
            score_bundle_id=str(bundle["score_bundle_id"]),
            event_doc={
                "score_bundle_hash": score_bundle["score_bundle_hash"],
                "rolling_window_hash": ctx["window"].window_hash,
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
                "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                "scoring_health_gate": scoring_health_gate,
                "scored_via": "global_icp_queue",
            },
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="scored",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate.get("run_id") or ""),
            ticket_id=str(candidate.get("ticket_id") or ""),
        )

    async def _run_global_icp_queue_pass(self) -> list[str]:
        """One worker pass over the global (candidate, icp) queue.

        Enqueues any queued candidate's public jobs (and parks its private jobs
        held for the gate), then claims and scores jobs from the shared pool,
        deciding gates and assembling finished candidates. Returns the ids of
        candidates assembled in this pass.
        """
        ctx_cache: dict[str, dict[str, Any]] = {}

        async def _ctx(candidate_id: str) -> dict[str, Any] | None:
            if candidate_id in ctx_cache:
                return ctx_cache[candidate_id]
            row = await select_one(
                "research_lab_candidate_evaluation_current",
                filters=(("candidate_id", candidate_id),),
            )
            if row is None:
                return None
            ctx = await self._build_queue_candidate_context(dict(row))
            ctx_cache[candidate_id] = ctx
            return ctx

        # 1. Enqueue queued candidates not already in the job queue.
        queued = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="*",
            filters=(("current_candidate_status", "queued"),),
            order_by=(("current_status_at", False),),
            limit=50,
        )
        for offset, row in enumerate(queued):
            cid = str(row.get("candidate_id") or "")
            if await select_one(global_icp_queue.CANDIDATE_TABLE, filters=(("candidate_id", cid),)) is not None:
                continue
            try:
                ctx = await self._build_queue_candidate_context(dict(row))
            except Exception:
                logger.warning(
                    "research_lab_queue_enqueue_context_failed candidate_id=%s",
                    compact_ref(cid),
                    exc_info=True,
                )
                continue
            ctx_cache[cid] = ctx
            enqueued = await global_icp_queue.enqueue_candidate(
                candidate_id=cid,
                window_hash=ctx["window"].window_hash,
                public_items=ctx["public_items"],
                private_items=ctx["private_items"],
                baseline_public_score=ctx["baseline_public_score"],
                worker_ref=self.worker_ref,
                seq_base=offset * 10_000,
            )
            if enqueued:
                await create_candidate_evaluation_event(
                    candidate_id=cid,
                    run_id=str(row.get("run_id") or ""),
                    ticket_id=str(row.get("ticket_id") or ""),
                    event_type="evaluating",
                    candidate_status="evaluating",
                    evaluator_ref=self.worker_ref,
                    reason="global_icp_queue_enqueued",
                    event_doc={"worker_ref": self.worker_ref, "scored_via": "global_icp_queue"},
                )

        assembled: list[str] = []

        async def score_icp(job: Mapping[str, Any]) -> dict[str, Any]:
            ctx = await _ctx(str(job.get("candidate_id") or ""))
            if ctx is None:
                return {}
            item = ctx["items_by_ref"].get(str(job.get("icp_ref") or ""))
            if item is None:
                return {}
            results = await score_private_model_pair_items(
                benchmark_items=[item],
                base_runner=None,
                candidate_runner=ctx["runner"],
                company_scorer=ctx["scorer"],
                run_context=ctx["run_context"],
                image_candidate=True,
                runtime_patch=None,
                parent_freshness_check=None,
                trace_sink=ctx["trace_sink"],
            )
            return dict(results[0]) if results else {}

        def compute_public_score(public_docs: Sequence[Mapping[str, Any]]) -> float:
            return float(_queue_benchmark_style_score(public_docs, "candidate_company_scores"))

        async def assemble(candidate_id: str, docs: Mapping[str, Any]) -> None:
            ctx = await _ctx(candidate_id)
            if ctx is None:
                raise RuntimeError("assemble context missing")
            await self._queue_assemble_candidate(candidate_id, docs, ctx)
            assembled.append(candidate_id)

        counters = await global_icp_queue.run_queue_scoring_pass(
            worker_ref=self.worker_ref,
            lease_seconds=self.config.scoring_worker_model_timeout_seconds + 60,
            score_icp=score_icp,
            compute_public_score=compute_public_score,
            assemble_candidate=assemble,
        )
        logger.info(
            "research_lab_global_icp_queue_pass worker_ref=%s counters=%s assembled=%s",
            self.worker_ref,
            counters,
            len(assembled),
        )
        return assembled

    async def _claim_next_candidate(self) -> dict[str, Any] | None:
        rows = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="*",
            filters=(("current_candidate_status", "queued"),),
            order_by=(("current_status_at", False),),
            limit=50,
        )
        candidate: dict[str, Any] | None = None
        for row in rows:
            reason = str(row.get("current_reason") or "")
            status_at = row.get("current_status_at")
            if reason == "baseline_not_ready" and not _status_is_stale(
                status_at,
                self.config.scoring_worker_baseline_not_ready_retry_seconds,
            ):
                continue
            if reason == "candidate_scoring_retryable_failure" and not _status_is_stale(
                status_at,
                self.config.scoring_worker_retryable_failure_retry_seconds,
            ):
                continue
            candidate = dict(row)
            break
        if not candidate:
            if rows:
                logger.info(
                    "research_lab_candidate_claim_deferred worker_ref=%s queued_candidates=%s",
                    self.worker_ref,
                    len(rows),
                )
            return None
        start_gate = await self._candidate_scoring_start_gate()
        self._last_candidate_start_gate = start_gate
        if not start_gate.get("available"):
            hold_key = ":".join(
                str(start_gate.get(key) or "")
                for key in ("reason", "target_benchmark_date", "private_model_manifest_hash")
            )
            if self._candidate_start_hold_logged_key != hold_key:
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE SCORING START HELD",
                        (
                            ("Worker", self.worker_ref),
                            ("Queued candidates", len(rows)),
                            ("Reason", start_gate.get("reason")),
                            ("UTC now", start_gate.get("now_utc")),
                            ("Target baseline date", start_gate.get("target_benchmark_date")),
                            ("Quiet start seconds", start_gate.get("quiet_start_utc_seconds")),
                            ("Private model", compact_ref(start_gate.get("private_model_manifest_hash"))),
                            ("Action", "leaving queued candidates untouched"),
                        ),
                    )
                )
                self._candidate_start_hold_logged_key = hold_key
            return None
        self._candidate_start_hold_logged_key = None
        candidate_id = str(candidate.get("candidate_id") or "")
        fresh = await select_one(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status",
            filters=(("candidate_id", candidate_id),),
        )
        if not fresh or fresh.get("current_candidate_status") != "queued":
            return None
        try:
            assigned_event = await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="assigned",
                candidate_status="assigned",
                evaluator_ref=self.worker_ref,
                reason="assigned_to_gateway_qualification_worker",
                event_doc={
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                },
            )
        except Exception as exc:
            if _is_candidate_claim_race_error(exc):
                logger.info(
                    "research_lab_candidate_claim_race candidate_id=%s worker_ref=%s",
                    compact_ref(candidate_id),
                    self.worker_ref,
                )
                return None
            raise
        assigned_current = await select_one(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status,current_evaluator_ref,current_event_hash",
            filters=(("candidate_id", candidate_id),),
        )
        if (
            not assigned_current
            or assigned_current.get("current_candidate_status") != "assigned"
            or assigned_current.get("current_evaluator_ref") != self.worker_ref
            or assigned_current.get("current_event_hash") != assigned_event.get("anchored_hash")
        ):
            logger.info(
                "research_lab_candidate_claim_lost candidate_id=%s worker_ref=%s",
                compact_ref(candidate_id),
                self.worker_ref,
            )
            return None
        await safe_project_public_loop_activity(
            str(candidate["ticket_id"]),
            source_ref=f"candidate_assigned:{candidate_id}",
            reason="assigned_to_gateway_qualification_worker",
            config=self.config,
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB CANDIDATE ALLOCATED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Ticket", compact_ref(candidate.get("ticket_id"))),
                    ("Proxy ref", self.proxy_ref_hash),
                ),
            )
        )
        return candidate

    # SLA after which a stale-parent candidate with no rebase should be alerted on.
    _STALE_PARENT_REBASE_SLA_SECONDS = 3600

    async def _alert_stuck_candidates(self) -> None:
        """Observability alerts (structured logs) for candidates stuck beyond their
        expected windows. Read-only; never mutates. Best-effort (swallows query errors).
        """
        try:
            try:
                retry_seconds = int(self.config.scoring_worker_baseline_not_ready_retry_seconds or 900)
            except (TypeError, ValueError):
                retry_seconds = 900
            baseline_alert_after = max(300, retry_seconds * 4)
            baseline_rows = await select_many(
                "research_lab_candidate_evaluation_current",
                columns="candidate_id,current_status_at,current_reason,current_candidate_status",
                filters=(
                    ("current_candidate_status", "queued"),
                    ("current_reason", "baseline_not_ready"),
                ),
                limit=200,
            )
            stuck_baseline = [
                str(r.get("candidate_id") or "")
                for r in baseline_rows
                if _status_is_stale(r.get("current_status_at"), baseline_alert_after)
            ]
            if stuck_baseline:
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATES STUCK WAITING FOR BASELINE",
                        (
                            ("Worker", self.worker_ref),
                            ("Count", len(stuck_baseline)),
                            ("Beyond", f"{baseline_alert_after}s"),
                            ("Candidates", ", ".join(compact_ref(c) for c in stuck_baseline[:10])),
                        ),
                    )
                )
                logger.warning(
                    "research_lab_candidates_stuck_baseline_not_ready count=%s threshold_seconds=%s",
                    len(stuck_baseline),
                    baseline_alert_after,
                )

            sla_seconds = int(
                getattr(self.config, "stale_parent_rebase_sla_seconds", None)
                or self._STALE_PARENT_REBASE_SLA_SECONDS
            )
            stale_parent_rows = await select_many(
                "research_lab_candidate_evaluation_current",
                columns="candidate_id,current_status_at",
                filters=(
                    ("current_candidate_status", "rejected"),
                    ("current_reason", "stale_parent_needs_rescore"),
                ),
                limit=200,
            )
            overdue: list[str] = []
            for row in stale_parent_rows:
                if not _status_is_stale(row.get("current_status_at"), sla_seconds):
                    continue
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id:
                    continue
                existing = await select_many(
                    "research_lab_candidate_promotion_events",
                    columns="promotion_event_id",
                    filters=(("candidate_id", candidate_id), ("event_type", "rebase_queued")),
                    limit=1,
                )
                if not existing:
                    overdue.append(candidate_id)
            if overdue:
                overdue_key = sha256_json({"candidates": sorted(overdue), "sla_seconds": sla_seconds})
                if overdue_key not in self._stale_parent_overdue_warning_keys:
                    self._stale_parent_overdue_warning_keys.add(overdue_key)
                    logger.warning(
                        format_worker_block(
                            "RESEARCH LAB STALE-PARENT CANDIDATES NOT REBASED WITHIN SLA",
                            (
                                ("Worker", self.worker_ref),
                                ("Count", len(overdue)),
                                ("SLA", f"{sla_seconds}s"),
                                ("Candidates", ", ".join(compact_ref(c) for c in overdue[:10])),
                            ),
                        )
                    )
                    logger.warning(
                        "research_lab_stale_parent_candidates_overdue count=%s sla_seconds=%s",
                        len(overdue),
                        sla_seconds,
                    )
        except Exception as exc:  # noqa: BLE001 - alerting must never break the worker pass
            logger.warning("research_lab_stuck_candidate_alert_failed error=%s", str(exc)[:200])

    async def _recover_stale_candidate_claims(self) -> int:
        stale_after_seconds = max(120, int(self.config.scoring_worker_model_timeout_seconds or 900) + 60)
        rows: list[dict[str, Any]] = []
        for status in ("assigned", "evaluating"):
            rows.extend(
                await select_many(
                    "research_lab_candidate_evaluation_current",
                    columns=(
                        "candidate_id,run_id,ticket_id,receipt_id,current_candidate_status,current_status_at,"
                        "current_evaluator_ref,current_event_hash,private_model_manifest_doc,"
                        "candidate_model_manifest_doc,candidate_artifact_hash"
                    ),
                    filters=(("current_candidate_status", status),),
                    # Oldest first: under backlog the stalest claims must be
                    # recovered before the limit truncates the scan.
                    order_by=(("current_status_at", False),),
                    limit=50,
                )
            )
        recovered = 0
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            run_id = str(row.get("run_id") or "")
            ticket_id = str(row.get("ticket_id") or "")
            if not candidate_id or not run_id or not ticket_id:
                continue
            recovery_reason = _candidate_claim_recovery_reason(
                row,
                candidate_id=candidate_id,
                worker_ref=self.worker_ref,
                worker_index=self.config.scoring_worker_index,
                total_workers=self.config.scoring_worker_total_workers,
                worker_started_at=self._worker_started_at,
                stale_after_seconds=stale_after_seconds,
                restart_orphan_grace_seconds=_env_int(
                    "RESEARCH_LAB_SCORING_RESTART_ORPHAN_GRACE_SECONDS",
                    5,
                ),
            )
            if recovery_reason is None:
                continue
            claim_attempts = await self._candidate_claim_attempt_count(candidate_id)
            max_attempts = int(self.config.scoring_worker_max_claim_requeues)
            if claim_attempts >= max_attempts:
                progress_summary = await self._candidate_scoring_progress_summary(row)
                completed_icps = _completed_icp_count_from_progress_doc(progress_summary)
                if completed_icps > 0:
                    try:
                        await create_candidate_evaluation_event(
                            candidate_id=candidate_id,
                            run_id=run_id,
                            ticket_id=ticket_id,
                            event_type="queued",
                            candidate_status="queued",
                            evaluator_ref=self.worker_ref,
                            reason="stale_gateway_scoring_progress_requeued",
                            event_doc={
                                "recovering_worker_ref": self.worker_ref,
                                "previous_evaluator_ref": row.get("current_evaluator_ref"),
                                "previous_candidate_status": row.get("current_candidate_status"),
                                "previous_event_hash": row.get("current_event_hash"),
                                "previous_status_at": row.get("current_status_at"),
                                "stale_after_seconds": stale_after_seconds,
                                "worker_started_at": self._worker_started_at.isoformat(),
                                "recovery_reason": recovery_reason,
                                "claim_attempts": claim_attempts,
                                "max_claim_attempts": max_attempts,
                                "retry_budget_preserved": True,
                                "scoring_progress": progress_summary,
                            },
                        )
                        await create_scoring_dispatch_event(
                            dispatch_type="candidate_scoring",
                            dispatch_status="completed",
                            worker_ref=self.worker_ref,
                            proxy_ref_hash=self.proxy_ref_hash,
                            candidate_id=candidate_id,
                            run_id=run_id,
                            ticket_id=ticket_id,
                            event_doc={
                                "reason": "stale_gateway_scoring_progress_requeued",
                                "dispatch_context": "candidate_scoring_recovery",
                                "retry_budget_preserved": True,
                                "recovery_reason": recovery_reason,
                                "claim_attempts": claim_attempts,
                                "max_claim_attempts": max_attempts,
                                "scoring_progress": progress_summary,
                            },
                        )
                        recovered += 1
                    except Exception as exc:
                        logger.warning(
                            "research_lab_stale_candidate_progress_requeue_failed candidate_id=%s error=%s",
                            compact_ref(candidate_id),
                            str(exc)[:240],
                        )
                        continue
                    logger.info(
                        "research_lab_stale_candidate_progress_requeued candidate_id=%s completed_icps=%s claim_attempts=%s/%s",
                        compact_ref(candidate_id),
                        completed_icps,
                        claim_attempts,
                        max_attempts,
                    )
                    continue
                try:
                    await create_candidate_evaluation_event(
                        candidate_id=candidate_id,
                        run_id=run_id,
                        ticket_id=ticket_id,
                        event_type="failed",
                        candidate_status="failed",
                        evaluator_ref=self.worker_ref,
                        reason="stale_gateway_scoring_retry_limit_exceeded",
                        event_doc={
                            "failure_class": "stale_claim_retry_limit_exceeded",
                            "retryable": False,
                            "recovering_worker_ref": self.worker_ref,
                            "previous_evaluator_ref": row.get("current_evaluator_ref"),
                            "previous_candidate_status": row.get("current_candidate_status"),
                            "previous_event_hash": row.get("current_event_hash"),
                            "previous_status_at": row.get("current_status_at"),
                            "stale_after_seconds": stale_after_seconds,
                            "worker_started_at": self._worker_started_at.isoformat(),
                            "recovery_reason": recovery_reason,
                            "claim_attempts": claim_attempts,
                            "max_claim_attempts": max_attempts,
                        },
                    )
                    await create_scoring_dispatch_event(
                        # NOTE: 'candidate_scoring' (not a '_recovery' variant) —
                        # the dispatch_type DB CHECK only allows the base types.
                        dispatch_type="candidate_scoring",
                        dispatch_status="failed",
                        worker_ref=self.worker_ref,
                        proxy_ref_hash=self.proxy_ref_hash,
                        candidate_id=candidate_id,
                        run_id=run_id,
                        ticket_id=ticket_id,
                        event_doc={
                            "reason": "stale_gateway_scoring_retry_limit_exceeded",
                            "dispatch_context": "candidate_scoring_recovery",
                            "failure_class": "stale_claim_retry_limit_exceeded",
                            "retryable": False,
                            "recovery_reason": recovery_reason,
                            "claim_attempts": claim_attempts,
                            "max_claim_attempts": max_attempts,
                        },
                    )
                    recovered += 1
                except Exception as exc:
                    logger.warning(
                        "research_lab_stale_candidate_fail_limit_failed candidate_id=%s error=%s",
                        compact_ref(candidate_id),
                        str(exc)[:240],
                    )
                    continue
                # This terminal path must mirror the normal terminal path:
                # finalize the receipt and project the public card, or the
                # card freezes at `scoring` forever (bug #11).
                try:
                    await self._maybe_finalize_candidate_receipt(row)
                except Exception:
                    logger.exception(
                        "research_lab_stale_candidate_receipt_finalize_failed candidate_id=%s",
                        compact_ref(candidate_id),
                    )
                try:
                    await safe_project_public_loop_activity(
                        ticket_id,
                        source_ref=f"candidate_stale_claim_retry_limit:{candidate_id}",
                        reason="stale_gateway_scoring_retry_limit_exceeded",
                        config=self.config,
                    )
                except Exception:
                    logger.exception(
                        "research_lab_stale_candidate_projection_failed candidate_id=%s",
                        compact_ref(candidate_id),
                    )
                continue
            try:
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=run_id,
                    ticket_id=ticket_id,
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason="stale_gateway_scoring_requeued",
                    event_doc={
                        "recovering_worker_ref": self.worker_ref,
                        "previous_evaluator_ref": row.get("current_evaluator_ref"),
                        "previous_candidate_status": row.get("current_candidate_status"),
                        "previous_event_hash": row.get("current_event_hash"),
                        "previous_status_at": row.get("current_status_at"),
                        "stale_after_seconds": stale_after_seconds,
                        "worker_started_at": self._worker_started_at.isoformat(),
                        "recovery_reason": recovery_reason,
                    },
                )
                recovered += 1
            except Exception as exc:
                logger.warning(
                    "research_lab_stale_candidate_requeue_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:240],
                )
        if recovered:
            logger.info(
                "research_lab_stale_candidates_requeued worker_ref=%s count=%s stale_after_seconds=%s",
                self.worker_ref,
                recovered,
                stale_after_seconds,
            )
        return recovered

    async def _candidate_claim_attempt_count(self, candidate_id: str) -> int:
        rows = await select_many(
            "research_lab_candidate_evaluation_events",
            columns="candidate_id,event_type,candidate_status,reason,seq",
            filters=(
                ("candidate_id", candidate_id),
                # Heartbeats write an `evaluating` event every ~120s; without
                # this filter the seq-ascending page fills with heartbeats and
                # later assigned events fall outside the limit.
                ("event_type", "in", ("assigned", "queued")),
            ),
            order_by=(("seq", False),),
            limit=100,
        )
        return _count_claim_attempts(rows)

    async def _candidate_scoring_progress_summary(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return _safe_scoring_progress_summary(source="none", completed_icp_count=0)
        event_rows = await select_many(
            "research_lab_candidate_evaluation_events",
            columns="candidate_id,event_type,reason,event_doc,seq",
            filters=(
                ("candidate_id", candidate_id),
                ("event_type", "in", ("evaluating", "queued", "failed", "rejected")),
            ),
            order_by=(("seq", True),),
            limit=80,
        )
        from_events = _latest_scoring_progress_from_events(event_rows)
        if _completed_icp_count_from_progress_doc(from_events) > 0:
            return from_events

        private_manifest = candidate.get("private_model_manifest_doc")
        manifest_uri = (
            str(private_manifest.get("manifest_uri") or "")
            if isinstance(private_manifest, Mapping)
            else ""
        )
        location = _scoring_progress_s3_prefix(manifest_uri, candidate_id)
        if location is None:
            return from_events

        candidate_manifest = candidate.get("candidate_model_manifest_doc")
        candidate_artifact_hash = (
            str(candidate_manifest.get("model_artifact_hash") or "")
            if isinstance(candidate_manifest, Mapping)
            else str(candidate.get("candidate_artifact_hash") or "")
        )
        bucket, object_prefix = location
        try:
            from_s3 = await asyncio.to_thread(
                _load_latest_scoring_progress_summary,
                bucket,
                object_prefix,
                candidate_artifact_hash=candidate_artifact_hash,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_scoring_progress_inspect_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:200],
            )
            return from_events
        return from_s3

    async def _candidate_scoring_heartbeat(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        started_at: float,
        claim_lost: asyncio.Event | None = None,
        progress_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        try:
            interval = max(
                60.0,
                float(os.environ.get("RESEARCH_LAB_SCORING_HEARTBEAT_SECONDS", "120")),
            )
        except ValueError:
            interval = 120.0
        while True:
            await asyncio.sleep(interval)
            try:
                current = await select_one(
                    "research_lab_candidate_evaluation_current",
                    columns="candidate_id,current_candidate_status,current_evaluator_ref",
                    filters=(("candidate_id", candidate_id),),
                )
                if (
                    not current
                    or current.get("current_candidate_status") != "evaluating"
                    or current.get("current_evaluator_ref") != self.worker_ref
                ):
                    logger.warning(
                        "research_lab_candidate_heartbeat_claim_lost candidate_id=%s worker_ref=%s",
                        compact_ref(candidate_id),
                        self.worker_ref,
                    )
                    if claim_lost is not None:
                        claim_lost.set()
                    return
                event_doc: dict[str, Any] = {
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                }
                if progress_snapshot is not None:
                    try:
                        progress_doc = dict(progress_snapshot())
                    except Exception:
                        progress_doc = {}
                    completed_icps = _completed_icp_count_from_progress_doc(progress_doc)
                    if completed_icps > 0:
                        event_doc["completed_icp_count"] = completed_icps
                        event_doc["scoring_progress"] = _safe_scoring_progress_summary(
                            source="heartbeat",
                            completed_icp_count=completed_icps,
                            rolling_window_hash=str(progress_doc.get("rolling_window_hash") or ""),
                        )
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="evaluating",
                    candidate_status="evaluating",
                    evaluator_ref=self.worker_ref,
                    reason="gateway_qualification_worker_heartbeat",
                    event_doc=event_doc,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "research_lab_candidate_heartbeat_failed candidate_id=%s worker_ref=%s error=%s",
                    compact_ref(candidate_id),
                    self.worker_ref,
                    str(exc)[:240],
                )

    async def _score_candidate(self, candidate: Mapping[str, Any]) -> None:
        candidate_id = str(candidate["candidate_id"])
        start = time.time()
        scored_event_written = False
        scored_score_bundle_id = ""
        try:
            evaluation_epoch = await self._resolve_evaluation_epoch()
            stale_result = await self._maybe_rebase_stale_candidate_before_scoring(
                candidate,
                evaluation_epoch=evaluation_epoch,
                elapsed_seconds=lambda: round(time.time() - start, 3),
            )
            if stale_result.get("status") in {
                "legacy_patch_candidate_unsupported",
                "stale_parent_needs_rescore",
                "stale_parent_rebase_failed",
            }:
                await self._maybe_finalize_candidate_receipt(candidate)
                await safe_project_public_loop_activity(
                    str(candidate["ticket_id"]),
                    source_ref=f"candidate_stale_parent_or_legacy_rejected:{candidate_id}",
                    reason=str(stale_result["status"]),
                    config=self.config,
                )
                try:
                    await self._write_audit_bundle(evaluation_epoch)
                except Exception:
                    logger.exception("Research Lab audit bundle write failed after candidate rejection")
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE PRE-SCORING REJECTED",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Status", stale_result.get("status")),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                return
            if stale_result.get("status") == "stale_parent_rebased_to_current":
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE REBASED BEFORE SCORING",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Derived candidate", compact_ref(stale_result.get("derived_candidate_id"))),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                return
            assigned_dispatch_event = await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="assigned",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
            )
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORING STARTED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Ticket", compact_ref(candidate.get("ticket_id"))),
                        ("Evaluation epoch", evaluation_epoch),
                        ("Model timeout", f"{self.config.scoring_worker_model_timeout_seconds}s"),
                        ("Proxy ref", self.proxy_ref_hash),
                    ),
                )
            )
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="evaluating",
                candidate_status="evaluating",
                evaluator_ref=self.worker_ref,
                reason="gateway_qualification_worker_started",
                event_doc={"worker_ref": self.worker_ref, "proxy_ref_hash": self.proxy_ref_hash},
            )
            artifact = PrivateModelArtifactManifest.from_mapping(candidate["private_model_manifest_doc"])
            patch = candidate["candidate_patch_manifest"]
            candidate_kind = str(candidate.get("candidate_kind") or "")
            if candidate_kind != "image_build":
                raise RuntimeError("candidate scoring requires image_build candidate_kind")
            candidate_manifest_doc = candidate.get("candidate_model_manifest_doc")
            if not isinstance(candidate_manifest_doc, Mapping):
                raise RuntimeError("image_build candidate is missing candidate_model_manifest_doc")
            candidate_artifact = PrivateModelArtifactManifest.from_mapping(candidate_manifest_doc)
            reused_bundle_row = await self._find_reusable_scored_bundle(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                evaluation_epoch=evaluation_epoch,
            )
            if reused_bundle_row is not None:
                # Bug #12 rescore path: this candidate's signed bundle already
                # landed but its `scored` event write failed; reuse the bundle
                # instead of producing a divergent second evaluation.
                scored_score_bundle_id = str(reused_bundle_row["score_bundle_id"])
                await self._complete_candidate_from_reused_bundle(
                    candidate,
                    candidate_id=candidate_id,
                    bundle_row=reused_bundle_row,
                    evaluation_epoch=evaluation_epoch,
                    start=start,
                )
                scored_event_written = True
                return
            window, private_holdout_gate = await self._daily_candidate_scoring_window_and_gate(
                artifact=artifact,
            )
            await create_rolling_icp_window(window)
            benchmark = SealedBenchmarkSet(
                benchmark_id=window.benchmark_id,
                icp_set_hash=window.window_hash,
                split_ref=window.split_ref,
                item_refs=window.item_refs,
                scoring_version="qualification-company-scorer:v1",
                hidden_plaintext_available=True,
            )
            candidate_runner = DockerPrivateModelRunner(
                DockerPrivateModelSpec(
                    image_digest=candidate_artifact.image_digest,
                    timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                    env_passthrough=self._private_model_env_passthrough(),
                    extra_env=self._with_provider_cost_evaluation_scope(
                        self._private_scoring_env(),
                        run_type="candidate_scoring",
                        rolling_window_hash=window.window_hash,
                        artifact_hash=candidate_artifact.model_artifact_hash,
                        candidate_id=candidate_id,
                        run_id=str(candidate.get("run_id") or ""),
                        ticket_id=str(candidate.get("ticket_id") or ""),
                        dispatch_event_id=str(assigned_dispatch_event.get("dispatch_event_id") or ""),
                        evaluation_epoch=evaluation_epoch,
                        started_at=start,
                    ),
                )
            )
            run_context = self._candidate_run_context(
                candidate,
                window_hash=window.window_hash,
                evaluation_epoch=evaluation_epoch,
            )
            # §5.4 scorer-judge traces: wrap the qualification scorer so every
            # per-company judgment is captured (pointer-only in event docs) at
            # the evaluator's scoring boundary; behavior-identical to the
            # default scorer the evaluator would otherwise construct.
            scorer_trace_pointers: dict[str, dict[str, str]] = {}
            candidate_company_scorer = _TraceCapturingCompanyScorer(
                recorder=self._get_scorer_trace_recorder(),
                context_ref=candidate_id,
                manifest_uri=artifact.manifest_uri,
                benchmark_items=window.benchmark_items,
                pointer_map=scorer_trace_pointers,
                candidate_id=candidate_id,
                candidate_model_manifest_hash=getattr(artifact, "manifest_hash", None),
            )
            last_freshness_check_at = 0.0
            claim_lost_event = asyncio.Event()

            async def parent_freshness_check(progress: Mapping[str, Any]) -> None:
                nonlocal last_freshness_check_at
                if claim_lost_event.is_set() and _env_flag("RESEARCH_LAB_SCORING_ABORT_ON_CLAIM_LOSS"):
                    raise ClaimLostDuringScoring(
                        f"scoring claim lost for candidate {compact_ref(candidate_id)}; "
                        "aborting in-flight evaluation at ICP boundary"
                    )
                now = time.time()
                phase = str(progress.get("phase") or "")
                if (
                    phase != "before_icp"
                    and last_freshness_check_at
                    and now - last_freshness_check_at < self.config.stale_parent_check_interval_seconds
                ):
                    return
                last_freshness_check_at = now
                await self._check_candidate_scoring_freshness(
                    parent_artifact=artifact,
                    candidate_window_hash=window.window_hash,
                    progress=progress,
                )

            # Bug #31: resume already-completed ICPs from the persisted progress
            # artifact and checkpoint each new ICP result, so a requeue at ICP
            # 19/20 no longer re-runs the whole evaluation. Best-effort: any
            # checkpoint failure only loses resumability.
            resume_results: list[dict[str, Any]] | None = None
            icp_checkpoint = None
            progress_rows: list[dict[str, Any]] = []
            progress_location = (
                _scoring_progress_s3_location(
                    artifact.manifest_uri, candidate_id, window.window_hash
                )
                if _per_icp_checkpoint_enabled()
                else None
            )
            if progress_location:
                progress_bucket, progress_key = progress_location
                try:
                    resume_results = await asyncio.to_thread(
                        _load_scoring_progress,
                        progress_bucket,
                        progress_key,
                        window_hash=window.window_hash,
                        candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_scoring_progress_load_failed candidate_id=%s error=%s",
                        compact_ref(candidate_id),
                        str(exc)[:200],
                    )
                    resume_results = []
                if resume_results:
                    logger.info(
                        "research_lab_scoring_progress_resumed candidate_id=%s completed_icps=%s",
                        compact_ref(candidate_id),
                        len(resume_results),
                    )
                progress_rows = [dict(row) for row in resume_results or []]

                async def _checkpoint_completed_icp(row: Mapping[str, Any]) -> None:
                    progress_rows.append(dict(row))
                    try:
                        await asyncio.to_thread(
                            _store_scoring_progress,
                            progress_bucket,
                            progress_key,
                            candidate_id=candidate_id,
                            window_hash=window.window_hash,
                            candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                            rows=progress_rows,
                        )
                    except Exception as exc:
                        logger.warning(
                            "research_lab_scoring_progress_store_failed candidate_id=%s error=%s",
                            compact_ref(candidate_id),
                            str(exc)[:200],
                        )

                icp_checkpoint = _checkpoint_completed_icp

            def _heartbeat_progress_snapshot() -> Mapping[str, Any]:
                return {
                    "completed_icp_count": len(progress_rows),
                    "rolling_window_hash": window.window_hash,
                }

            heartbeat_task = asyncio.create_task(
                self._candidate_scoring_heartbeat(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    started_at=start,
                    claim_lost=claim_lost_event,
                    progress_snapshot=_heartbeat_progress_snapshot,
                )
            )
            langfuse_obs = None
            langfuse_trace_id = ""
            # End-to-end continuity: the eval span joins the loop engine's
            # deterministic run trace (run_trace_id(run_id)), and the sampling
            # decision is seeded by run_id so a sampled run keeps its scoring
            # span too — never a fragment of one without the other.
            langfuse_run_id = str(candidate["run_id"])
            candidate_loop_node_id = (
                str((candidate.get("candidate_build_doc") or {}).get("loop_node_id") or "")
                if isinstance(candidate.get("candidate_build_doc"), Mapping)
                else ""
            )
            try:
                with observation(
                    "research_lab.private_eval_pair",
                    as_type="span",
                    trace_id=langfuse_run_trace_id(langfuse_run_id),
                    sample_seed=langfuse_run_id or None,
                    metadata={
                        "run_id": str(candidate["run_id"]),
                        "ticket_id": str(candidate["ticket_id"]),
                        "candidate_id": candidate_id,
                        **({"loop_node_id": candidate_loop_node_id} if candidate_loop_node_id else {}),
                        "miner_hotkey_hash": miner_hotkey_hash(str(candidate.get("miner_hotkey") or "")),
                        "evaluation_epoch": evaluation_epoch,
                        "parent_artifact_hash": artifact.model_artifact_hash,
                        "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
                        "candidate_patch_hash": str(candidate.get("candidate_patch_hash") or ""),
                        "icp_set_hash": window.window_hash,
                        "benchmark_split_ref": window.split_ref,
                        "worker_ref": self.worker_ref,
                    },
                ) as langfuse_obs:
                    try:
                        score_bundle = await evaluate_private_model_pair(
                            artifact_manifest=artifact,
                            benchmark=benchmark,
                            patch_manifest=patch,
                            candidate_artifact_manifest=candidate_artifact.to_dict(),
                            benchmark_items=window.benchmark_items,
                            base_runner=None,
                            candidate_runner=candidate_runner,
                            company_scorer=candidate_company_scorer,
                            run_context={**run_context, "signature_ref": "pending"},
                            policy=self._evaluation_policy(),
                            private_holdout_gate=private_holdout_gate,
                            parent_freshness_check=parent_freshness_check,
                            icp_checkpoint=icp_checkpoint,
                            resume_results=resume_results,
                            # In-container traces upload keyed by candidate; the
                            # evaluator puts the returned refs into per-ICP rows.
                            trace_sink=self._candidate_incontainer_trace_sink(candidate_id),
                        )
                        langfuse_trace_id = finish_score_bundle_observation(langfuse_obs, score_bundle)
                    finally:
                        heartbeat_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await heartbeat_task
            except StaleParentDuringScoring as stale_exc:
                stale_result = await self._queue_stale_parent_rebase(
                    candidate,
                    active_artifact=stale_exc.active_artifact,
                    candidate_parent=stale_exc.candidate_parent,
                    evaluation_epoch=evaluation_epoch,
                    elapsed_seconds=round(time.time() - start, 3),
                    stage="during_scoring_parent_changed",
                    stale_progress=stale_exc.progress,
                )
                await self._maybe_finalize_candidate_receipt(candidate)
                await safe_project_public_loop_activity(
                    str(candidate["ticket_id"]),
                    source_ref=f"candidate_stale_parent_during_scoring:{candidate_id}",
                    reason=str(stale_result.get("status") or "stale_parent_during_scoring"),
                    config=self.config,
                )
                try:
                    await self._write_audit_bundle(evaluation_epoch)
                except Exception:
                    logger.exception("Research Lab audit bundle write failed after stale parent during scoring")
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE STALE DURING SCORING",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Status", stale_result.get("status")),
                            ("Derived candidate", compact_ref(stale_result.get("derived_candidate_id"))),
                            ("Completed ICPs", stale_exc.progress.get("completed_icp_count", 0)),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                return
            gate_result = score_bundle.get("private_holdout_gate")
            private_holdout_rejected = (
                isinstance(gate_result, Mapping)
                and str(gate_result.get("decision") or "") == "rejected_before_private_holdout"
            )
            scoring_health_gate = self._scoring_health_gate_result(score_bundle)
            unsigned_hash = str(score_bundle["score_bundle_hash"])
            signature_ref = await asyncio.to_thread(
                sign_digest_with_kms,
                key_id=self.config.score_bundle_kms_key_id,
                digest_hash=unsigned_hash,
                signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
            )
            score_bundle = {**score_bundle, "signature_ref": signature_ref}
            score_bundle_request = ResearchLabScoreBundleCreateRequest(
                bundle_status="scored",
                receipt_id=candidate.get("receipt_id") or None,
                score_bundle=score_bundle,
            )
            bundle, _bundle_event = await create_score_bundle(score_bundle_request)
            scored_score_bundle_id = str(bundle["score_bundle_id"])
            if langfuse_trace_id:
                try:
                    await insert_row(
                        "engine_trace_mappings",
                        {
                            "execution_trace_ref": str(score_bundle.get("execution_trace_ref") or f"score_bundle:{score_bundle['score_bundle_hash']}"),
                            "langfuse_trace_id": langfuse_trace_id,
                            "langfuse_project": os.getenv("LANGFUSE_PROJECT", "leadpoet-lab-prod-redacted"),
                            "score_bundle_hash": score_bundle["score_bundle_hash"],
                            "run_id": str(candidate["run_id"]),
                            "ticket_id": str(candidate["ticket_id"]),
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_langfuse_trace_mapping_write_failed candidate_id=%s error=%s",
                        compact_ref(candidate_id),
                        str(exc)[:200],
                    )
            await self._create_scored_evaluation_event(
                candidate=candidate,
                candidate_id=candidate_id,
                score_bundle_id=scored_score_bundle_id,
                event_doc={
                    "score_bundle_hash": score_bundle["score_bundle_hash"],
                    "rolling_window_hash": window.window_hash,
                    "elapsed_seconds": round(time.time() - start, 3),
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                    "scoring_health_gate": scoring_health_gate,
                    # §5.4 pointers only ({icp_ref: {s3_ref, sha256}}) — never
                    # judgment content (audit-scan poison).
                    **(
                        {"scorer_trace_refs": dict(scorer_trace_pointers)}
                        if scorer_trace_pointers
                        else {}
                    ),
                    **({"langfuse_trace_id": langfuse_trace_id} if langfuse_trace_id else {}),
                },
            )
            scored_event_written = True
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="scored",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                rolling_window_hash=window.window_hash,
                score_bundle_id=str(bundle["score_bundle_id"]),
                event_doc={
                    "elapsed_seconds": round(time.time() - start, 3),
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                    "scoring_health_gate": scoring_health_gate,
                },
            )
            if private_holdout_rejected:
                promotion_result = await self._record_public_holdout_rejected(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                    gate_result=gate_result,
                )
            else:
                promotion_result = await self._maybe_promote_scored_candidate(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                )
            if promotion_result.get("status") == "stale_parent_needs_rescore":
                promotion_result = await self._queue_stale_parent_rebase(
                    candidate,
                    active_artifact=(await load_active_private_model(self.config, register_bootstrap=True)).artifact,
                    candidate_parent=str(candidate.get("parent_artifact_hash") or ""),
                    evaluation_epoch=evaluation_epoch,
                    elapsed_seconds=round(time.time() - start, 3),
                    stage="after_scoring_parent_changed",
                )
            await self._maybe_record_score_backfill(
                candidate=candidate,
                score_bundle_row=bundle,
                score_bundle=score_bundle,
                promotion_result=promotion_result,
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_scored:{candidate_id}:{bundle['score_bundle_id']}",
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
            await self._write_audit_bundle(int(run_context["evaluation_epoch"]))
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Score bundle", compact_ref(bundle["score_bundle_id"])),
                        ("Rolling window", compact_ref(window.window_hash)),
                        ("Private holdout gate", (gate_result or {}).get("decision") if isinstance(gate_result, Mapping) else "-"),
                        ("Promotion", promotion_result.get("status")),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
        except Exception as exc:
            if scored_event_written:
                await self._record_scored_candidate_side_effect_failure(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    score_bundle_id=scored_score_bundle_id,
                    error=exc,
                    elapsed_seconds=round(time.time() - start, 3),
                )
                return
            failure_class, retryable = _candidate_scoring_failure_class(exc)
            claim_attempts = await self._candidate_claim_attempt_count(candidate_id)
            max_attempts = int(self.config.scoring_worker_max_claim_requeues)
            if failure_class == "baseline_not_ready":
                retry_after_seconds = int(self.config.scoring_worker_baseline_not_ready_retry_seconds)
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason="baseline_not_ready",
                    event_doc={
                        "failure_class": failure_class,
                        "retryable": True,
                        "retry_after_seconds": retry_after_seconds,
                        "error_diagnostics": _event_error_diagnostics(exc),
                        "elapsed_seconds": round(time.time() - start, 3),
                        "worker_ref": self.worker_ref,
                        "proxy_ref_hash": self.proxy_ref_hash,
                        "claim_attempts": claim_attempts,
                        **_candidate_baseline_wait_event_doc(exc),
                    },
                )
                logger.warning(
                    "research_lab_candidate_baseline_not_ready_requeued candidate_id=%s retry_after_seconds=%s error=%s",
                    compact_ref(candidate_id),
                    retry_after_seconds,
                    str(exc)[:240],
                )
                return
            if retryable and claim_attempts < max_attempts:
                retry_after_seconds = int(self.config.scoring_worker_retryable_failure_retry_seconds)
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason="candidate_scoring_retryable_failure",
                    event_doc={
                        "failure_class": failure_class,
                        "retryable": True,
                        "retry_after_seconds": retry_after_seconds,
                        "error_diagnostics": _event_error_diagnostics(exc),
                        "elapsed_seconds": round(time.time() - start, 3),
                        "worker_ref": self.worker_ref,
                        "proxy_ref_hash": self.proxy_ref_hash,
                        "claim_attempts": claim_attempts,
                        "max_claim_attempts": max_attempts,
                    },
                )
                logger.warning(
                    "research_lab_candidate_retryable_failure_requeued candidate_id=%s failure_class=%s claim_attempts=%s/%s error=%s",
                    compact_ref(candidate_id),
                    failure_class,
                    claim_attempts,
                    max_attempts,
                    str(exc)[:240],
                )
                return
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="failed",
                candidate_status="failed",
                evaluator_ref=self.worker_ref,
                reason=f"candidate_scoring_{failure_class}",
                event_doc={
                    "failure_class": failure_class,
                    "retryable": bool(retryable),
                    "claim_attempts": claim_attempts,
                    "max_claim_attempts": max_attempts,
                    "error_diagnostics": _event_error_diagnostics(exc),
                    "elapsed_seconds": round(time.time() - start, 3),
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                },
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_doc={
                    "failure_class": failure_class,
                    "retryable": bool(retryable),
                    "claim_attempts": claim_attempts,
                    "max_claim_attempts": max_attempts,
                    "error_diagnostics": _event_error_diagnostics(exc),
                },
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_failed:{candidate_id}",
                reason=f"candidate_scoring_{failure_class}",
                config=self.config,
            )
            try:
                await self._write_audit_bundle(await self._resolve_evaluation_epoch())
            except Exception:
                logger.exception("Research Lab audit bundle write failed after candidate failure")
            logger.exception(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORING FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Error", str(exc)[:300]),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )

    async def _create_scored_evaluation_event(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        score_bundle_id: str,
        event_doc: dict[str, Any],
    ) -> None:
        """Write the terminal ``scored`` event with bounded retries (bug #12).

        A signed score bundle already exists when this runs; losing the scored
        event orphans that bundle and re-scores the candidate from scratch on
        the next claim, so transient write failures retry and a persistent
        failure leaves a reconcilable marker before propagating.
        """
        try:
            attempts = max(1, int(os.getenv("RESEARCH_LAB_SCORED_EVENT_WRITE_ATTEMPTS", "3")))
        except ValueError:
            attempts = 3
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="scored",
                    candidate_status="scored",
                    evaluator_ref=self.worker_ref,
                    reason="gateway_qualification_worker_scored_candidate",
                    score_bundle_id=score_bundle_id,
                    event_doc=event_doc,
                )
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "research_lab_scored_event_write_retry candidate_id=%s attempt=%s/%s error=%s",
                    compact_ref(candidate_id),
                    attempt,
                    attempts,
                    _short_error(exc),
                )
                if attempt < attempts:
                    await asyncio.sleep(min(30.0, 2.0 * attempt))
        try:
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                score_bundle_id=score_bundle_id,
                event_doc={
                    "reason": "scored_event_write_failed_for_signed_bundle",
                    "reconcile_marker": "scored_event_missing_for_signed_bundle",
                    "error_diagnostics": _event_error_diagnostics(last_exc),
                },
            )
        except Exception:
            logger.exception("research_lab_scored_event_marker_write_failed")
        logger.error(
            "research_lab_scored_event_write_failed_permanently candidate_id=%s score_bundle_id=%s attempts=%s",
            compact_ref(candidate_id),
            compact_ref(score_bundle_id),
            attempts,
        )
        assert last_exc is not None
        raise last_exc

    async def _find_reusable_scored_bundle(
        self,
        *,
        candidate_id: str,
        run_id: str,
        candidate_artifact_hash: str,
        evaluation_epoch: int,
    ) -> dict[str, Any] | None:
        """Locate a signed score bundle already produced for this candidate+epoch.

        Bug #12 rescore path: a candidate whose bundle landed but whose scored
        event write failed gets re-claimed; without this it re-runs the full
        evaluation and produces a divergent second signed bundle.
        """
        if _env_flag("RESEARCH_LAB_DISABLE_SCORED_BUNDLE_REUSE"):
            return None
        try:
            rows = await select_many(
                "research_evaluation_score_bundle_current",
                columns="*",
                filters=(
                    ("run_id", run_id),
                    ("candidate_artifact_hash", candidate_artifact_hash),
                    ("evaluation_epoch", int(evaluation_epoch)),
                    ("bundle_status", "scored"),
                ),
                order_by=(("created_at", True),),
                limit=5,
            )
        except Exception as exc:
            # Best-effort side check: never fail the scoring pass over it.
            logger.warning(
                "research_lab_reusable_bundle_lookup_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:200],
            )
            return None
        trace_suffix = f":{candidate_id}"
        for row in rows:
            doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else None
            if not doc:
                continue
            if not str(doc.get("execution_trace_ref") or "").endswith(trace_suffix):
                continue
            if not str(row.get("signature_ref") or doc.get("signature_ref") or ""):
                continue
            if not str(doc.get("score_bundle_hash") or ""):
                continue
            return dict(row)
        return None

    async def _complete_candidate_from_reused_bundle(
        self,
        candidate: Mapping[str, Any],
        *,
        candidate_id: str,
        bundle_row: Mapping[str, Any],
        evaluation_epoch: int,
        start: float,
    ) -> None:
        """Finish a candidate from its already-signed bundle (bug #12): write the
        scored event, then mirror the normal post-scored side-effect sequence."""
        score_bundle = dict(bundle_row.get("score_bundle_doc") or {})
        score_bundle_id = str(bundle_row["score_bundle_id"])
        rolling_window_hash = str(score_bundle.get("icp_set_hash") or bundle_row.get("icp_set_hash") or "")
        gate_result = score_bundle.get("private_holdout_gate")
        private_holdout_rejected = (
            isinstance(gate_result, Mapping)
            and str(gate_result.get("decision") or "") == "rejected_before_private_holdout"
        )
        scoring_health_gate = self._scoring_health_gate_result(score_bundle)
        logger.warning(
            format_worker_block(
                "RESEARCH LAB CANDIDATE REUSING SIGNED SCORE BUNDLE",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Reason", "signed bundle exists for this candidate+epoch; skipping re-evaluation"),
                ),
            )
        )
        await self._create_scored_evaluation_event(
            candidate=candidate,
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
            event_doc={
                "score_bundle_hash": str(score_bundle.get("score_bundle_hash") or ""),
                "rolling_window_hash": rolling_window_hash,
                "elapsed_seconds": round(time.time() - start, 3),
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
                "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                "scoring_health_gate": scoring_health_gate,
                "reused_signed_score_bundle": True,
            },
        )
        # From here the candidate is scored: side-effect failures must not undo
        # that, mirroring the main path's post-scored handling.
        try:
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="scored",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                rolling_window_hash=rolling_window_hash or None,
                score_bundle_id=score_bundle_id,
                event_doc={
                    "elapsed_seconds": round(time.time() - start, 3),
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                    "scoring_health_gate": scoring_health_gate,
                    "reused_signed_score_bundle": True,
                },
            )
            if private_holdout_rejected:
                promotion_result = await self._record_public_holdout_rejected(
                    candidate=candidate,
                    score_bundle_row=bundle_row,
                    score_bundle=score_bundle,
                    gate_result=gate_result,
                )
            else:
                promotion_result = await self._maybe_promote_scored_candidate(
                    candidate=candidate,
                    score_bundle_row=bundle_row,
                    score_bundle=score_bundle,
                )
            if promotion_result.get("status") == "stale_parent_needs_rescore":
                promotion_result = await self._queue_stale_parent_rebase(
                    candidate,
                    active_artifact=(await load_active_private_model(self.config, register_bootstrap=True)).artifact,
                    candidate_parent=str(candidate.get("parent_artifact_hash") or ""),
                    evaluation_epoch=evaluation_epoch,
                    elapsed_seconds=round(time.time() - start, 3),
                    stage="after_scoring_parent_changed",
                )
            await self._maybe_record_score_backfill(
                candidate=candidate,
                score_bundle_row=bundle_row,
                score_bundle=score_bundle,
                promotion_result=promotion_result,
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_scored:{candidate_id}:{score_bundle_id}",
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
            await self._write_audit_bundle(evaluation_epoch)
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Score bundle", compact_ref(score_bundle_id)),
                        ("Rolling window", compact_ref(rolling_window_hash)),
                        ("Private holdout gate", (gate_result or {}).get("decision") if isinstance(gate_result, Mapping) else "-"),
                        ("Promotion", promotion_result.get("status")),
                        ("Reused signed bundle", True),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
        except Exception as exc:
            await self._record_scored_candidate_side_effect_failure(
                candidate=candidate,
                candidate_id=candidate_id,
                score_bundle_id=score_bundle_id,
                error=exc,
                elapsed_seconds=round(time.time() - start, 3),
            )

    async def _record_scored_candidate_side_effect_failure(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        score_bundle_id: str,
        error: BaseException,
        elapsed_seconds: float,
    ) -> None:
        event_doc = {
            "error_diagnostics": _event_error_diagnostics(error),
            "elapsed_seconds": elapsed_seconds,
            "worker_ref": self.worker_ref,
            "proxy_ref_hash": self.proxy_ref_hash,
            "score_bundle_id": score_bundle_id,
            "candidate_status_preserved": "scored",
        }
        try:
            # Deliberately NOT promotion_failed/failed: scoring (and possibly
            # promotion) succeeded — only a post-score side effect failed. The
            # failed labels outrank `scored` in the public card derivation and
            # corrupted all 21 historical scored candidates (chain B).
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                source_score_bundle_id=score_bundle_id or None,
                event_type="post_score_side_effect_failed",
                promotion_status="post_score_side_effect_failed",
                active_parent_artifact_hash=str(candidate.get("parent_artifact_hash") or ""),
                candidate_parent_artifact_hash=str(candidate.get("parent_artifact_hash") or ""),
                worker_ref=self.worker_ref,
                event_doc={
                    **event_doc,
                    "reason": "post_score_side_effect_failed",
                },
            )
        except Exception:
            logger.exception("research_lab_post_score_side_effect_event_write_failed")
        try:
            await create_scoring_dispatch_event(
                # 'candidate_scoring' — the dispatch_type DB CHECK only allows
                # the base types, so the old side_effect variant never landed.
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                score_bundle_id=score_bundle_id or None,
                event_doc=event_doc,
            )
        except Exception:
            logger.exception("research_lab_scored_candidate_side_effect_dispatch_failed")
        # The candidate genuinely scored: finalize the receipt and project the
        # true scored state so the public card does not freeze at `scoring`.
        try:
            await self._maybe_finalize_candidate_receipt(candidate)
        except Exception:
            logger.exception("research_lab_scored_candidate_side_effect_receipt_finalize_failed")
        try:
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_scored_side_effect_failed:{candidate_id}",
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
        except Exception:
            logger.exception("research_lab_scored_candidate_side_effect_projection_failed")
        logger.exception(
            format_worker_block(
                "RESEARCH LAB CANDIDATE POST-SCORE SIDE EFFECT FAILED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Error", str(error)[:300]),
                    ("Candidate state", "scored"),
                ),
            )
        )

    async def _record_public_holdout_rejected(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        gate_result: Any,
    ) -> dict[str, Any]:
        aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
        metric = promotion_improvement_metric(score_bundle)
        improvement_points = float(metric.improvement_points)
        delta_lcb = float(aggregates.get("delta_lcb") or 0.0)
        candidate_parent = str(candidate.get("parent_artifact_hash") or score_bundle.get("parent_artifact_hash") or "")
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row.get("score_bundle_id") or ""),
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=candidate_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=str(score_bundle.get("icp_set_hash") or ""),
            improvement_points=improvement_points,
            threshold_points=float(self.config.improvement_threshold_points),
            worker_ref=self.worker_ref,
            event_doc={
                "delta_lcb": round(delta_lcb, 6),
                "auto_commit_enabled": self.config.auto_commit_enabled,
                "candidate_kind": str(candidate.get("candidate_kind") or ""),
                "decision_path": "public_holdout_rejected",
                "promotion_metric": metric.event_doc(),
            },
        )
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row.get("score_bundle_id") or ""),
            event_type="public_holdout_rejected",
            promotion_status="rejected",
            active_parent_artifact_hash=candidate_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=str(score_bundle.get("icp_set_hash") or ""),
            improvement_points=improvement_points,
            threshold_points=float(self.config.improvement_threshold_points),
            worker_ref=self.worker_ref,
            event_doc={
                "reason": "rejected_before_private_holdout",
                "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                "mean_delta": round(improvement_points, 6),
                "delta_lcb": round(delta_lcb, 6),
                "candidate_kind": str(candidate.get("candidate_kind") or ""),
                "promotion_metric": metric.event_doc(),
            },
        )
        return {"status": "rejected_public_holdout_gate"}

    def _scoring_health_gate_result(self, score_bundle: Mapping[str, Any]) -> dict[str, Any]:
        health = score_bundle.get("scoring_health") if isinstance(score_bundle.get("scoring_health"), Mapping) else {}
        thresholds = {
            "reference_runtime_failure_rate": self.config.scoring_health_max_reference_runtime_failure_rate,
            "candidate_runtime_failure_rate": self.config.scoring_health_max_candidate_runtime_failure_rate,
            "reference_zero_company_rate": self.config.scoring_health_max_reference_zero_company_rate,
            "candidate_zero_company_rate": self.config.scoring_health_max_candidate_zero_company_rate,
            "provider_error_rate": self.config.scoring_health_max_provider_error_rate,
            "timeout_rate": self.config.scoring_health_max_timeout_rate,
        }
        observed = {
            "reference_runtime_failure_rate": _failure_rate_from_success(
                health.get("reference_runtime_success_rate")
            ),
            "candidate_runtime_failure_rate": _failure_rate_from_success(
                health.get("candidate_runtime_success_rate")
            ),
            "reference_zero_company_rate": _safe_float(health.get("reference_zero_company_rate")),
            "candidate_zero_company_rate": _safe_float(health.get("candidate_zero_company_rate")),
            "provider_error_rate": _safe_float(health.get("provider_error_rate")),
            "timeout_rate": _safe_float(health.get("timeout_rate")),
        }
        violations: list[dict[str, Any]] = []
        for metric, threshold in thresholds.items():
            value = float(observed.get(metric, 0.0))
            if value <= float(threshold):
                continue
            violations.append(
                {
                    "metric": metric,
                    "observed": round(value, 6),
                    "threshold": round(float(threshold), 6),
                }
            )
        configured_enabled = bool(self.config.scoring_health_gate_enabled)
        return {
            "schema_version": "1.0",
            "enabled": False,
            "configured_enabled": configured_enabled,
            "decision": "observe_only",
            "would_quarantine": bool(violations),
            "violations": violations,
            "thresholds": {key: round(float(value), 6) for key, value in thresholds.items()},
            "observed": {key: round(float(value), 6) for key, value in observed.items()},
        }

    async def _maybe_promote_scored_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.config.auto_promotion_enabled:
            return {"status": "disabled"}
        return await ResearchLabPromotionController(
            self.config,
            worker_ref=self.worker_ref,
        ).process_scored_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
        )

    async def _maybe_record_score_backfill(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        promotion_result: Mapping[str, Any] | None,
    ) -> None:
        """Phase-5 score backfill (flag default OFF): reconcile the realized
        bundle against the hypothesis's predicted_delta + dev score in an
        append-only calibration row. Best-effort; never blocks scoring."""
        candidate_id = str(candidate.get("candidate_id") or "")
        try:
            from gateway.research_lab.score_backfill import record_score_backfill

            result = await record_score_backfill(
                candidate=candidate,
                score_bundle_row=score_bundle_row,
                score_bundle=score_bundle,
                promotion_result=promotion_result,
                created_by=self.worker_ref,
            )
            if result.get("status") == "recorded":
                logger.info(
                    "research_lab_score_backfill_recorded candidate=%s bundle=%s outcome=%s",
                    compact_ref(candidate_id),
                    compact_ref(result.get("score_bundle_id")),
                    str(result.get("outcome") or "")[:80],
                )
        except Exception as exc:
            logger.warning(
                "research_lab_score_backfill_failed candidate=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:200],
            )

    # --- §5.2-2 confirmation re-run: worker-side measurement ---------------
    # A candidate held `held_pending_confirmation` cleared every merge gate on
    # its first measurement. The worker re-measures BOTH sides fresh — a full
    # champion (baseline) run AND a candidate run over the same rolling window
    # — as a side measurement (never recorded as the day's benchmark
    # reference), then re-drives the promotion decision. All state derives
    # from promotion events; a crash at any point resumes from the events.

    async def _maybe_run_pending_confirmation(self) -> dict[str, Any] | None:
        holds = await self._find_pending_confirmation_holds()
        if not holds:
            return None
        lease_seconds = _confirmation_lease_seconds()
        budget = confirmation_attempt_budget(self.config)
        for hold in holds:
            candidate_id = str(hold.get("candidate_id") or "")
            score_bundle_id = str(hold.get("source_score_bundle_id") or "")
            if not candidate_id or not score_bundle_id:
                continue
            try:
                state = await load_confirmation_state(
                    candidate_id=candidate_id,
                    score_bundle_id=score_bundle_id,
                )
            except Exception as exc:  # noqa: BLE001 - skip unreadable state, next pass retries
                logger.warning(
                    "research_lab_confirmation_state_load_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:200],
                )
                continue
            latest_reason = str(state.get("latest_reason") or "")
            if latest_reason == CONFIRMATION_CLOSED_REASON:
                # A terminal promotion decision already landed for this
                # candidate+bundle; nothing left to drive.
                continue
            try:
                if await candidate_already_promoted(candidate_id):
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "research_lab_confirmation_promoted_check_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:200],
                )
                continue
            if latest_reason == CONFIRMATION_RESULT_REASON:
                # Measurement recorded but the decision never landed (crash
                # between record and re-drive): finish the state machine
                # without re-measuring.
                result = await self._redrive_confirmation_decision(
                    candidate_id=candidate_id,
                    score_bundle_id=score_bundle_id,
                    context="recorded_confirmation_pending_decision",
                )
                if result is not None:
                    return {
                        "processed": True,
                        "status": "confirmation_decision_redriven",
                        "candidate_id": candidate_id,
                        "promotion_status": str(result.get("status") or ""),
                    }
                continue
            if latest_reason == CONFIRMATION_STARTED_REASON:
                started = state.get("latest_event") or {}
                age = _status_age_seconds(started.get("created_at"))
                if age is not None and age <= lease_seconds:
                    # In flight elsewhere (or our own crashed predecessor —
                    # wait out the lease either way).
                    continue
            if int(state.get("attempts") or 0) >= budget:
                # Budget exhausted: re-drive so the gate records the terminal
                # rejected_confirmation_failed decision.
                result = await self._redrive_confirmation_decision(
                    candidate_id=candidate_id,
                    score_bundle_id=score_bundle_id,
                    context="confirmation_attempts_exhausted",
                )
                if result is not None:
                    return {
                        "processed": True,
                        "status": "confirmation_decision_redriven",
                        "candidate_id": candidate_id,
                        "promotion_status": str(result.get("status") or ""),
                    }
                continue
            return await self._run_confirmation_attempt(hold=hold, state=state)
        return None

    async def _find_pending_confirmation_holds(self) -> list[dict[str, Any]]:
        """Newest-first held_pending_confirmation events, deduped per
        candidate+bundle. Primary query filters on event_doc->>reason (a plain
        PostgREST JSON filter); falls back to a bounded promotion_checked scan
        filtered in Python if the JSON filter is rejected."""
        columns = (
            "promotion_event_id,candidate_id,source_score_bundle_id,rolling_window_hash,"
            "improvement_points,threshold_points,worker_ref,event_doc,created_at"
        )
        try:
            rows = await select_many(
                "research_lab_candidate_promotion_events",
                columns=columns,
                filters=(
                    ("event_type", "promotion_checked"),
                    ("event_doc->>reason", CONFIRMATION_HOLD_REASON),
                ),
                order_by=(("created_at", True),),
                limit=50,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to a plain-column scan
            logger.warning(
                "research_lab_confirmation_hold_scan_json_filter_failed error=%s",
                str(exc)[:200],
            )
            try:
                rows = await select_many(
                    "research_lab_candidate_promotion_events",
                    columns=columns,
                    filters=(("event_type", "promotion_checked"),),
                    order_by=(("created_at", True),),
                    limit=200,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                logger.warning(
                    "research_lab_confirmation_hold_scan_failed error=%s",
                    str(fallback_exc)[:200],
                )
                return []
            rows = [
                row
                for row in rows
                if isinstance(row.get("event_doc"), Mapping)
                and str(row["event_doc"].get("reason") or "") == CONFIRMATION_HOLD_REASON
            ]
        seen: set[tuple[str, str]] = set()
        holds: list[dict[str, Any]] = []
        for row in rows:
            key = (
                str(row.get("candidate_id") or ""),
                str(row.get("source_score_bundle_id") or ""),
            )
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            holds.append(dict(row))
        return holds

    async def _redrive_confirmation_decision(
        self,
        *,
        candidate_id: str,
        score_bundle_id: str,
        context: str,
    ) -> dict[str, Any] | None:
        """Re-run the promotion decision for a held candidate. The promotion
        controller re-checks every gate (including the recorded confirmation)
        from events, so this is idempotent and crash-safe."""
        candidate = await select_one(
            "research_lab_candidate_evaluation_current",
            filters=(("candidate_id", candidate_id),),
        )
        if not candidate:
            logger.warning(
                "research_lab_confirmation_redrive_candidate_missing candidate_id=%s",
                compact_ref(candidate_id),
            )
            return None
        bundle_row = await select_one(
            "research_evaluation_score_bundle_current",
            filters=(("score_bundle_id", score_bundle_id),),
        )
        score_bundle = (
            bundle_row.get("score_bundle_doc") if isinstance(bundle_row, Mapping) else None
        )
        if not isinstance(score_bundle, Mapping):
            logger.warning(
                "research_lab_confirmation_redrive_bundle_missing candidate_id=%s score_bundle_id=%s",
                compact_ref(candidate_id),
                compact_ref(score_bundle_id),
            )
            return None
        result = await self._maybe_promote_scored_candidate(
            candidate=candidate,
            score_bundle_row=bundle_row,
            score_bundle=dict(score_bundle),
        )
        if str(result.get("status") or "") == "stale_parent_needs_rescore":
            # Mirror _score_candidate's post-promotion handling: the champion
            # moved while the candidate was held, so queue the rebase (deduped)
            # instead of stranding a scored candidate at rebase_required.
            rebase_result = await self._queue_confirmation_stale_parent_rebase(candidate)
            if rebase_result is not None:
                result = rebase_result
        logger.info(
            format_worker_block(
                "RESEARCH LAB CONFIRMATION DECISION",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Context", context),
                    ("Promotion", result.get("status")),
                ),
            )
        )
        status = str(result.get("status") or "")
        if status not in CONFIRMATION_NON_CLOSING_STATUSES:
            # A terminal decision landed: close the confirmation so future
            # passes stop re-driving this candidate+bundle. Best-effort — a
            # lost close only costs one redundant re-drive next pass.
            try:
                await create_candidate_promotion_event(
                    candidate_id=candidate_id,
                    source_score_bundle_id=score_bundle_id,
                    event_type="promotion_checked",
                    promotion_status="checked",
                    worker_ref=self.worker_ref,
                    event_doc={
                        "reason": CONFIRMATION_CLOSED_REASON,
                        "decision_path": "confirmation_rerun_closed",
                        "decision_status": status,
                        "context": context,
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("research_lab_confirmation_close_event_write_failed")
        return result

    async def _queue_confirmation_stale_parent_rebase(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Queue the stale-parent rebase for a held candidate whose champion
        moved. Deduped on an existing rebase_queued event (the confirmation
        re-drive can repeat after a crash); best-effort — a failure leaves the
        recorded rebase_required promotion event for the operator."""
        candidate_id = str(candidate.get("candidate_id") or "")
        try:
            existing = await select_many(
                "research_lab_candidate_promotion_events",
                columns="promotion_event_id",
                filters=(("candidate_id", candidate_id), ("event_type", "rebase_queued")),
                limit=1,
            )
            if existing:
                return {"status": "stale_parent_rebase_already_queued"}
            active = await load_active_private_model(self.config, register_bootstrap=True)
            evaluation_epoch = await self._resolve_evaluation_epoch()
            return await self._queue_stale_parent_rebase(
                candidate,
                active_artifact=active.artifact,
                candidate_parent=str(candidate.get("parent_artifact_hash") or ""),
                evaluation_epoch=evaluation_epoch,
                elapsed_seconds=0.0,
                stage="confirmation_redrive_parent_changed",
            )
        except Exception as exc:  # noqa: BLE001 - best-effort side effect
            logger.warning(
                "research_lab_confirmation_stale_parent_rebase_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:240],
            )
            return None

    async def _run_confirmation_attempt(
        self,
        *,
        hold: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        candidate_id = str(hold.get("candidate_id") or "")
        score_bundle_id = str(hold.get("source_score_bundle_id") or "")
        candidate = await select_one(
            "research_lab_candidate_evaluation_current",
            filters=(("candidate_id", candidate_id),),
        )
        if not candidate:
            return None
        bundle_row = await select_one(
            "research_evaluation_score_bundle_current",
            filters=(("score_bundle_id", score_bundle_id),),
        )
        score_bundle = (
            bundle_row.get("score_bundle_doc") if isinstance(bundle_row, Mapping) else None
        )
        if not isinstance(score_bundle, Mapping):
            return None
        attempt = int(state.get("attempts") or 0) + 1
        candidate_parent = str(candidate.get("parent_artifact_hash") or "")
        first_pass_points = _safe_float(hold.get("improvement_points"), default=0.0)
        threshold_points = _safe_float(
            hold.get("threshold_points"),
            default=float(self.config.improvement_threshold_points),
        )
        started_event = await create_candidate_promotion_event(
            candidate_id=candidate_id,
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=candidate_parent or None,
            candidate_parent_artifact_hash=candidate_parent or None,
            rolling_window_hash=str(hold.get("rolling_window_hash") or "") or None,
            improvement_points=first_pass_points,
            threshold_points=threshold_points,
            worker_ref=self.worker_ref,
            event_doc={
                "reason": CONFIRMATION_STARTED_REASON,
                "decision_path": "confirmation_rerun_attempt",
                "attempt": attempt,
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
            },
        )
        # Post-write claim confirm (write-then-verify, like
        # _claim_next_candidate): events are append-only with no unique claim
        # constraint, so two workers racing past the pre-check both land a
        # started event. The OLDEST unexpired claim in the contiguous head run
        # of started events wins (ties break on promotion_event_id); the later
        # writer backs off before measuring. Expired claims (dead workers)
        # are ignored, so a stale claim can always be reclaimed.
        fresh = await load_confirmation_state(
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
        )
        if str(fresh.get("latest_reason") or "") != CONFIRMATION_STARTED_REASON:
            # Someone recorded/failed an attempt between our write and this
            # read — the state moved on without us.
            logger.info(
                "research_lab_confirmation_claim_lost candidate_id=%s worker_ref=%s (state advanced)",
                compact_ref(candidate_id),
                self.worker_ref,
            )
            return {"processed": False, "status": "confirmation_claim_lost", "candidate_id": candidate_id}
        lease_seconds = _confirmation_lease_seconds()
        open_claims = []
        for claim in fresh.get("open_claim_events") or []:
            age = _status_age_seconds(claim.get("created_at"))
            if age is not None and age > lease_seconds:
                continue
            open_claims.append(claim)
        my_event_id = str(started_event.get("promotion_event_id") or "")
        winner = min(
            open_claims,
            key=lambda row: (str(row.get("created_at") or ""), str(row.get("promotion_event_id") or "")),
            default=None,
        )
        if winner is None or str(winner.get("promotion_event_id") or "") != my_event_id:
            logger.info(
                "research_lab_confirmation_claim_lost candidate_id=%s worker_ref=%s winner_worker=%s",
                compact_ref(candidate_id),
                self.worker_ref,
                (winner or {}).get("worker_ref"),
            )
            return {"processed": False, "status": "confirmation_claim_lost", "candidate_id": candidate_id}
        start = time.time()
        logger.info(
            format_worker_block(
                "RESEARCH LAB CONFIRMATION RERUN STARTED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Attempt", f"{attempt}/{confirmation_attempt_budget(self.config)}"),
                    ("First-pass delta", f"{first_pass_points:.4f}"),
                ),
            )
        )
        try:
            measurement = await self._run_confirmation_measurement(
                candidate=candidate,
                score_bundle=score_bundle,
                hold=hold,
                attempt=attempt,
                start=start,
            )
        except Exception as exc:  # noqa: BLE001 - infra failure re-holds, bounded by budget
            try:
                await create_candidate_promotion_event(
                    candidate_id=candidate_id,
                    source_score_bundle_id=score_bundle_id,
                    event_type="promotion_checked",
                    promotion_status="checked",
                    active_parent_artifact_hash=candidate_parent or None,
                    candidate_parent_artifact_hash=candidate_parent or None,
                    rolling_window_hash=str(hold.get("rolling_window_hash") or "") or None,
                    improvement_points=first_pass_points,
                    threshold_points=threshold_points,
                    worker_ref=self.worker_ref,
                    event_doc={
                        "reason": CONFIRMATION_ATTEMPT_FAILED_REASON,
                        "decision_path": "confirmation_rerun_attempt",
                        "attempt": attempt,
                        "retryable": True,
                        "unhealthy_measurement": isinstance(exc, ConfirmationMeasurementUnhealthy),
                        "error_diagnostics": _event_error_diagnostics(exc),
                        "elapsed_seconds": round(time.time() - start, 3),
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("research_lab_confirmation_attempt_failed_event_write_failed")
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB CONFIRMATION RERUN ATTEMPT FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Attempt", attempt),
                        ("Error", str(exc)[:300]),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
            return {
                "processed": True,
                "status": "confirmation_attempt_failed",
                "candidate_id": candidate_id,
                "attempt": attempt,
            }
        if measurement.get("status") == "skipped_stale_parent":
            # The champion moved while the candidate was held: the fresh delta
            # would be against the wrong parent. Re-drive so the stale-parent
            # path records the proper rebase decision.
            result = await self._redrive_confirmation_decision(
                candidate_id=candidate_id,
                score_bundle_id=score_bundle_id,
                context="confirmation_skipped_stale_parent",
            )
            return {
                "processed": True,
                "status": "confirmation_skipped_stale_parent",
                "candidate_id": candidate_id,
                "promotion_status": str((result or {}).get("status") or ""),
            }
        confirmation_doc = dict(measurement.get("confirmation") or {})
        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=candidate_parent or None,
            candidate_parent_artifact_hash=candidate_parent or None,
            rolling_window_hash=str(confirmation_doc.get("rolling_window_hash") or "") or None,
            improvement_points=first_pass_points,
            threshold_points=threshold_points,
            worker_ref=self.worker_ref,
            event_doc={
                "reason": CONFIRMATION_RESULT_REASON,
                "decision_path": "confirmation_rerun_result",
                "attempt": attempt,
                "confirmation": confirmation_doc,
                # P18: the confirmation pair is calibration gold — two
                # measurements of the SAME artifact. Surface both aggregates
                # and their disagreement as flat, queryable fields (the
                # empirical same-model noise any reward-model training needs).
                "measurement_pair": {
                    "first_pass_delta": round(float(first_pass_points), 6),
                    "confirmation_delta": round(
                        _safe_float(confirmation_doc.get("confirmation_delta"), default=0.0), 6
                    ),
                    "delta_of_deltas": round(
                        _safe_float(confirmation_doc.get("confirmation_delta"), default=0.0)
                        - float(first_pass_points),
                        6,
                    ),
                },
            },
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB CONFIRMATION RERUN RECORDED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Attempt", attempt),
                    ("First-pass delta", f"{first_pass_points:.4f}"),
                    ("Confirmation delta", f"{_safe_float(confirmation_doc.get('confirmation_delta'), default=0.0):.4f}"),
                    ("Window match", confirmation_doc.get("window_match")),
                    ("Excluded ICPs", len(confirmation_doc.get("provider_excluded_icp_refs") or [])),
                    ("Elapsed", f"{time.time() - start:.1f}s"),
                ),
            )
        )
        result = await self._redrive_confirmation_decision(
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
            context="confirmation_recorded",
        )
        return {
            "processed": True,
            "status": "confirmation_recorded",
            "candidate_id": candidate_id,
            "attempt": attempt,
            "confirmation_delta": confirmation_doc.get("confirmation_delta"),
            "promotion_status": str((result or {}).get("status") or ""),
        }

    async def _run_confirmation_measurement(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        hold: Mapping[str, Any],
        attempt: int,
        start: float,
    ) -> dict[str, Any]:
        """Fresh baseline + fresh candidate evaluation over the same window.

        A side measurement only: nothing here writes a benchmark bundle or a
        public report, so the day's promotion reference is never replaced.
        Provider-errored ICPs (after the batch retry rounds) are excluded
        SYMMETRICALLY from both sides; too many exclusions make the attempt
        unhealthy (raise) instead of producing a skewed verdict.
        """
        active = await load_active_private_model(self.config, register_bootstrap=True)
        candidate_parent = str(candidate.get("parent_artifact_hash") or "")
        if active.artifact.model_artifact_hash != candidate_parent:
            return {"status": "skipped_stale_parent"}
        candidate_manifest_doc = candidate.get("candidate_model_manifest_doc")
        if not isinstance(candidate_manifest_doc, Mapping):
            raise RuntimeError("confirmation requires candidate_model_manifest_doc")
        candidate_artifact = PrivateModelArtifactManifest.from_mapping(candidate_manifest_doc)

        original_window_hash = str(
            score_bundle.get("icp_set_hash") or hold.get("rolling_window_hash") or ""
        )
        window = await fetch_rolling_icp_window(
            days=self.config.lab_champion_eval_days,
            icps_per_day=self.config.lab_champion_icps_per_day,
            **_rolling_window_fetch_kwargs(self.config),
            allow_partial=self.config.scoring_worker_allow_partial_icp_window,
        )
        window_match = bool(original_window_hash) and window.window_hash == original_window_hash
        if original_window_hash and not window_match:
            reconstructed = await self._reconstruct_rolling_window(original_window_hash)
            if reconstructed is not None:
                window = reconstructed
                window_match = True
        # Idempotent; also satisfies the promotion-event FK when the
        # confirmation ran on a newly rotated window.
        await create_rolling_icp_window(window)

        # Trace scope (§5.4 + in-container capture): keyed per candidate,
        # attempt, and side so confirmation docs never collide across attempts
        # or candidates. Carried via an instance attribute (not a new call
        # kwarg) so the side runner's call signature stays stable.
        self._confirmation_trace_scope = {
            "candidate_id": str(candidate.get("candidate_id") or "candidate"),
            "attempt": int(attempt),
            "manifest_uri": str(active.artifact.manifest_uri or ""),
        }
        try:
            champion_summaries, champion_stats = await self._run_confirmation_side(
                artifact=active.artifact,
                window=window,
                mode_label="confirmation_baseline",
                run_start=start,
            )
            candidate_summaries, candidate_stats = await self._run_confirmation_side(
                artifact=candidate_artifact,
                window=window,
                mode_label="confirmation_candidate",
                run_start=start,
            )
        finally:
            self._confirmation_trace_scope = None
        baseline_scores, baseline_unresolved, baseline_nonempty = _collect_confirmation_scores(
            champion_summaries
        )
        candidate_scores, candidate_unresolved, candidate_nonempty = _collect_confirmation_scores(
            candidate_summaries
        )
        if baseline_nonempty <= 0:
            raise PrivateModelRuntimeError(
                f"confirmation baseline returned zero companies across all {len(window.benchmark_items)} ICPs"
            )
        if candidate_nonempty <= 0:
            raise PrivateModelRuntimeError(
                f"confirmation candidate returned zero companies across all {len(window.benchmark_items)} ICPs"
            )
        excluded = sorted(set(baseline_unresolved) | set(candidate_unresolved))
        max_unresolved = _baseline_max_unresolved_icps()
        if len(excluded) > max_unresolved:
            raise ConfirmationMeasurementUnhealthy(
                "confirmation_measurement_unresolved_provider_errors: "
                f"unresolved={len(excluded)} max={max_unresolved} "
                f"baseline_unresolved={len(baseline_unresolved)} candidate_unresolved={len(candidate_unresolved)}"
            )
        excluded_set = set(excluded)
        included_refs = [
            ref
            for ref in baseline_scores
            if ref in candidate_scores and ref not in excluded_set
        ]
        if not included_refs:
            raise ConfirmationMeasurementUnhealthy(
                "confirmation_measurement_no_icps_left_after_exclusions"
            )
        baseline_aggregate = _average([baseline_scores[ref] for ref in included_refs])
        candidate_total = _average([candidate_scores[ref] for ref in included_refs])
        confirmation_delta = candidate_total - baseline_aggregate
        confirmation = {
            "schema_version": "1.0",
            "measurement_type": "confirmation_rerun_side_measurement",
            "rolling_window_hash": window.window_hash,
            "original_window_hash": original_window_hash,
            "window_match": window_match,
            "selected_icp_count": len(window.item_refs),
            "included_icp_count": len(included_refs),
            "provider_excluded_icp_refs": excluded,
            "baseline_aggregate_score": round(baseline_aggregate, 6),
            "candidate_total_score": round(candidate_total, 6),
            "confirmation_delta": round(confirmation_delta, 6),
            "per_icp_baseline_scores": {
                ref: round(baseline_scores[ref], 6) for ref in sorted(baseline_scores)
            },
            "per_icp_candidate_scores": {
                ref: round(candidate_scores[ref], 6) for ref in sorted(candidate_scores)
            },
            "baseline_retry": champion_stats,
            "candidate_retry": candidate_stats,
            "champion_artifact_hash": active.artifact.model_artifact_hash,
            "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
            "attempt": attempt,
            "worker_ref": self.worker_ref,
            "elapsed_seconds": round(time.time() - start, 3),
        }
        return {"status": "measured", "confirmation": confirmation}

    async def _run_confirmation_side(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        window: Any,
        mode_label: str,
        run_start: float,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """One side (champion or candidate) of a confirmation measurement,
        through the same batch machinery as the parallel baseline — provider
        retry rounds, benchmark Exa isolation, and §0-N6 scorer isolation
        included, so both sides run under the identical execution regime."""
        confirmation_scope = getattr(self, "_confirmation_trace_scope", None)
        confirmation_scope = confirmation_scope if isinstance(confirmation_scope, Mapping) else {}
        confirmation_attempt = None
        try:
            if confirmation_scope.get("attempt") is not None:
                confirmation_attempt = int(confirmation_scope.get("attempt"))
        except (TypeError, ValueError):
            confirmation_attempt = None
        confirmation_candidate_id = str(confirmation_scope.get("candidate_id") or "")
        runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_baseline_scoring_env(),
                    run_type="promotion_confirmation",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=artifact.model_artifact_hash,
                    candidate_id=confirmation_candidate_id,
                    confirmation_attempt=confirmation_attempt,
                    side=mode_label,
                    started_at=run_start,
                ),
            )
        )
        retry_runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_baseline_retry_scoring_env(),
                    run_type="promotion_confirmation",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=artifact.model_artifact_hash,
                    candidate_id=confirmation_candidate_id,
                    confirmation_attempt=confirmation_attempt,
                    side=mode_label,
                    started_at=run_start,
                ),
                # The first-pass runner already pulled this digest.
                pull_before_run=False,
            )
        )
        scorer = QualificationStyleCompanyScorer()
        summaries, stats = await self._run_baseline_batch(
            runner=runner,
            retry_runner=retry_runner,
            scorer=scorer,
            window=window,
            run_start=run_start,
            mode_label=mode_label,
            trace_context=self._confirmation_side_trace_context(mode_label=mode_label),
        )
        return summaries, {
            "retried": int(stats.get("retried") or 0),
            "recovered": int(stats.get("recovered") or 0),
            "unresolved": int(stats.get("unresolved") or 0),
        }

    def _confirmation_side_trace_context(self, *, mode_label: str) -> dict[str, Any] | None:
        """Trace scope for one confirmation side, derived from the pending
        confirmation's candidate/attempt (set by _run_confirmation_measurement).
        None outside a confirmation measurement — capture then stays off."""
        scope = getattr(self, "_confirmation_trace_scope", None)
        if not isinstance(scope, Mapping):
            return None
        side = "champion" if mode_label == "confirmation_baseline" else "candidate"
        candidate_ref = _trace_path_segment(scope.get("candidate_id"), fallback="candidate")
        return self._baseline_trace_context(
            context_ref=f"confirmation-{candidate_ref}-a{int(scope.get('attempt') or 0)}-{side}",
            manifest_uri=str(scope.get("manifest_uri") or ""),
            run_type=f"promotion_confirmation_{side}",
        )

    async def _reconstruct_rolling_window(self, window_hash: str) -> Any | None:
        """Best-effort re-materialization of a stored rolling window.

        The window selection is deterministic over its set rows, so re-fetching
        the original set_ids reproduces the identical window unless the set
        contents changed — verified by requiring the reconstructed hash to
        match. Any failure falls back to the caller's current window (the
        confirmation stays a same-window paired comparison either way)."""
        try:
            row = await select_one(
                "research_lab_rolling_icp_windows",
                filters=(("rolling_window_hash", window_hash),),
            )
            doc = row.get("window_doc") if isinstance(row, Mapping) else None
            if not isinstance(doc, Mapping):
                return None
            set_ids = [
                int(entry.get("set_id"))
                for entry in (doc.get("sets") or [])
                if isinstance(entry, Mapping) and entry.get("set_id") is not None
            ]
            if not set_ids:
                return None
            rows = await select_many(
                "qualification_private_icp_sets",
                columns="set_id,icps,icp_set_hash,active_from,active_until,is_active",
                filters=(("set_id", "in", tuple(set_ids)),),
                limit=max(len(set_ids), 10),
            )
            if len(rows) < len(set_ids):
                return None
            window = reconstruct_icp_window_from_doc(rows, doc)
            if window.window_hash != window_hash:
                logger.info(
                    "research_lab_confirmation_window_reconstruct_hash_mismatch expected=%s got=%s",
                    compact_ref(window_hash),
                    compact_ref(window.window_hash),
                )
                return None
            return window
        except Exception as exc:  # noqa: BLE001 - best-effort; caller keeps the current window
            logger.warning(
                "research_lab_confirmation_window_reconstruct_failed error=%s",
                str(exc)[:200],
            )
            return None

    async def _maybe_rebase_stale_candidate_before_scoring(
        self,
        candidate: Mapping[str, Any],
        *,
        evaluation_epoch: int,
        elapsed_seconds: Any,
    ) -> dict[str, Any]:
        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash
        candidate_parent = str(candidate.get("parent_artifact_hash") or "")
        candidate_id = str(candidate["candidate_id"])
        candidate_kind = str(candidate.get("candidate_kind") or "patch")
        if candidate_kind != "image_build":
            base_event_doc = {
                "action": "legacy_patch_candidate_rejected_before_scoring",
                "candidate_kind": candidate_kind,
                "active_parent_artifact_hash": active_parent,
                "candidate_parent_artifact_hash": candidate_parent,
                "evaluation_epoch": int(evaluation_epoch),
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
            }
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                event_type="unsupported_candidate_kind",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                worker_ref=self.worker_ref,
                event_doc={**base_event_doc, "stage": "before_scoring"},
            )
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="rejected",
                candidate_status="rejected",
                evaluator_ref=self.worker_ref,
                reason="legacy_patch_candidate_unsupported",
                event_doc={**base_event_doc, "elapsed_seconds": elapsed_seconds()},
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="rejected",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_doc={**base_event_doc, "reason": "legacy_patch_candidate_unsupported"},
            )
            return {"status": "legacy_patch_candidate_unsupported"}

        if candidate_parent == active_parent:
            return {"status": "current_parent"}

        return await self._queue_stale_parent_rebase(
            candidate,
            active_artifact=active.artifact,
            candidate_parent=candidate_parent,
            evaluation_epoch=evaluation_epoch,
            elapsed_seconds=elapsed_seconds(),
            stage="before_scoring_parent_changed",
        )

    async def _queue_stale_parent_rebase(
        self,
        candidate: Mapping[str, Any],
        *,
        active_artifact: PrivateModelArtifactManifest,
        candidate_parent: str,
        evaluation_epoch: int,
        elapsed_seconds: float,
        stage: str,
        stale_progress: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_parent = active_artifact.model_artifact_hash
        candidate_id = str(candidate["candidate_id"])
        rebase_depth = _stale_parent_rebase_depth(candidate)
        next_rebase_depth = rebase_depth + 1
        base_event_doc = {
            "action": "image_build_candidate_parent_changed",
            "active_parent_artifact_hash": active_parent,
            "candidate_parent_artifact_hash": candidate_parent,
            "evaluation_epoch": int(evaluation_epoch),
            "worker_ref": self.worker_ref,
            "proxy_ref_hash": self.proxy_ref_hash,
            "stage": stage,
            "rebase_depth": rebase_depth,
            "next_rebase_depth": next_rebase_depth,
            "reimbursement_preserved": True,
            "reimbursement_source": "hosted_loop_completion",
        }
        if stale_progress:
            base_event_doc["stale_progress"] = _stale_parent_progress_doc(stale_progress)
        if not self.config.stale_parent_rebase_enabled:
            await self._reject_stale_parent_candidate(
                candidate,
                base_event_doc=base_event_doc,
                reason="stale_parent_needs_rescore",
                elapsed_seconds=elapsed_seconds,
            )
            return {"status": "stale_parent_needs_rescore"}
        if rebase_depth >= self.config.stale_parent_rebase_max_depth:
            await self._reject_stale_parent_candidate(
                candidate,
                base_event_doc={
                    **base_event_doc,
                    "failure_class": "stale_parent_rebase_depth_exceeded",
                    "max_rebase_depth": self.config.stale_parent_rebase_max_depth,
                },
                reason="stale_parent_rebase_failed",
                elapsed_seconds=elapsed_seconds,
            )
            recovery_result = await self._recover_stale_parent_rebase_failed_candidate(candidate)
            return {
                "status": "stale_parent_rebase_failed",
                "error": "stale_parent_rebase_depth_exceeded",
                "recovery_result": recovery_result,
            }

        try:
            draft = await asyncio.to_thread(self._draft_from_stale_candidate, candidate)
            build = await asyncio.to_thread(
                CodeEditCandidateBuilder(self.config).build,
                draft=draft,
                parent_artifact=active_artifact,
                run_id=str(candidate["run_id"]),
                candidate_index=await self._next_rebase_candidate_index(str(candidate["run_id"])),
            )
            repair_used = False
        except CodeEditPatchApplyError as exc:
            try:
                draft, build = await self._repair_and_build_stale_candidate(
                    candidate,
                    active_artifact=active_artifact,
                    original_error=exc,
                    run_id=str(candidate["run_id"]),
                )
                repair_used = True
            except Exception as repair_exc:
                await self._reject_stale_parent_candidate(
                    candidate,
                    base_event_doc={
                        **base_event_doc,
                        "failure_class": "stale_parent_rebase_repair_failed",
                        "error_diagnostics": _event_error_diagnostics(repair_exc),
                        "error_hash": sha256_json({"error": str(repair_exc)}),
                    },
                    reason="stale_parent_rebase_failed",
                    elapsed_seconds=elapsed_seconds,
                )
                recovery_result = await self._recover_stale_parent_rebase_failed_candidate(candidate)
                return {
                    "status": "stale_parent_rebase_failed",
                    "error": str(repair_exc)[:300],
                    "recovery_result": recovery_result,
                }
        except Exception as exc:
            await self._reject_stale_parent_candidate(
                candidate,
                base_event_doc={
                    **base_event_doc,
                    "failure_class": "stale_parent_rebase_failed",
                    "error_diagnostics": _event_error_diagnostics(exc),
                    "error_hash": sha256_json({"error": str(exc)}),
                },
                reason="stale_parent_rebase_failed",
                elapsed_seconds=elapsed_seconds,
            )
            recovery_result = await self._recover_stale_parent_rebase_failed_candidate(candidate)
            return {
                "status": "stale_parent_rebase_failed",
                "error": str(exc)[:300],
                "recovery_result": recovery_result,
            }

        rebase_build_doc = {
            **build.build_doc,
            # Preserve the source candidate's loop-node linkage so the rebased
            # candidate's score bundle keeps the deterministic
            # execution_trace:<uuid5> ref (bundle→trace join survives rebases).
            **(
                {"loop_node_id": str((candidate.get("candidate_build_doc") or {}).get("loop_node_id") or "")}
                if isinstance(candidate.get("candidate_build_doc"), Mapping)
                and (candidate.get("candidate_build_doc") or {}).get("loop_node_id")
                else {}
            ),
            "stale_parent_rebase": {
                "schema_version": "1.0",
                "source_candidate_id": candidate_id,
                "source_parent_artifact_hash": candidate_parent,
                "rebased_parent_artifact_hash": active_parent,
                "repair_used": repair_used,
                "stage": stage,
                "depth": next_rebase_depth,
                "max_depth": self.config.stale_parent_rebase_max_depth,
            },
        }
        request = ResearchLabCandidateArtifactCreateRequest(
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            receipt_id=str(candidate.get("receipt_id") or "") or None,
            miner_hotkey=str(candidate["miner_hotkey"]),
            island=str(candidate.get("island") or "generalist"),
            candidate_kind="image_build",
            private_model_manifest=active_artifact.to_dict(),
            candidate_patch_manifest=build.code_edit_manifest,
            candidate_model_manifest=build.candidate_model_manifest.to_dict(),
            candidate_source_diff_hash=build.source_diff_hash,
            candidate_build_doc=rebase_build_doc,
            hypothesis_doc=dict(candidate.get("hypothesis_doc") or {}),
            redacted_public_summary=str(candidate.get("redacted_public_summary") or draft.redacted_summary or ""),
        )
        derived_candidate, _event = await create_candidate_artifact(request)
        derived_candidate_id = str(derived_candidate["candidate_id"])
        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            derived_candidate_id=derived_candidate_id,
            event_type="rebase_queued",
            promotion_status="rebenchmarking",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            worker_ref=self.worker_ref,
            event_doc={
                **base_event_doc,
                "derived_candidate_id": derived_candidate_id,
                "derived_candidate_artifact_hash": build.candidate_model_manifest.model_artifact_hash,
                "derived_source_diff_hash": build.source_diff_hash,
                "repair_used": repair_used,
            },
        )
        await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type="rejected",
            candidate_status="rejected",
            evaluator_ref=self.worker_ref,
            reason="stale_parent_rebased_to_current",
            event_doc={
                **base_event_doc,
                "derived_candidate_id": derived_candidate_id,
                "derived_source_diff_hash": build.source_diff_hash,
                "elapsed_seconds": elapsed_seconds,
            },
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="rejected",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_doc={
                **base_event_doc,
                "derived_candidate_id": derived_candidate_id,
                "reason": "stale_parent_rebased_to_current",
            },
        )
        return {
            "status": "stale_parent_rebased_to_current",
            "derived_candidate_id": derived_candidate_id,
            "repair_used": repair_used,
        }

    async def _recover_stale_parent_rebase_failed_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return {"ok": False, "status": "candidate_missing_id"}
        try:
            from .recovery import recover_rebase_failed_candidates

            result = await recover_rebase_failed_candidates(
                candidate_ids=[candidate_id],
                dry_run=False,
                max_batch_size=1,
                regenerate=True,
                actor_ref=self.worker_ref,
            )
        except Exception as exc:  # noqa: BLE001 - recovery must not hide the original rebase failure
            logger.warning(
                "research_lab_stale_parent_rebase_failed_auto_recovery_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                _short_error(exc),
            )
            return {"ok": False, "status": "auto_recovery_exception", "error": _short_error(exc)}
        if int(result.get("regenerated") or 0):
            logger.info(
                "research_lab_stale_parent_rebase_failed_auto_regenerated candidate_id=%s regenerated=%s",
                compact_ref(candidate_id),
                result.get("regenerated"),
            )
        return result

    async def _reject_stale_parent_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        base_event_doc: Mapping[str, Any],
        reason: str,
        elapsed_seconds: float,
    ) -> None:
        candidate_id = str(candidate["candidate_id"])
        active_parent = str(base_event_doc.get("active_parent_artifact_hash") or "")
        candidate_parent = str(base_event_doc.get("candidate_parent_artifact_hash") or "")
        event_doc = {
            "reimbursement_preserved": True,
            "reimbursement_source": "hosted_loop_completion",
            **dict(base_event_doc),
        }
        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            event_type="stale_parent_detected",
            promotion_status="rejected" if reason == "stale_parent_rebase_failed" else "rebase_required",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            worker_ref=self.worker_ref,
            event_doc=event_doc,
        )
        await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type="rejected",
            candidate_status="rejected",
            evaluator_ref=self.worker_ref,
            reason=reason,
            event_doc={**event_doc, "elapsed_seconds": elapsed_seconds},
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="rejected",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_doc={**event_doc, "reason": reason},
        )

    def _draft_from_stale_candidate(self, candidate: Mapping[str, Any]) -> CodeEditDraft:
        unified_diff = _load_candidate_source_diff(candidate)
        patch = candidate.get("candidate_patch_manifest")
        patch_doc = patch.get("patch_doc") if isinstance(patch, Mapping) else {}
        if not isinstance(patch_doc, Mapping):
            patch_doc = {}
        hypothesis = candidate.get("hypothesis_doc") if isinstance(candidate.get("hypothesis_doc"), Mapping) else {}
        patch_summary = str(patch.get("redacted_summary") or "") if isinstance(patch, Mapping) else ""
        target_files = patch_doc.get("target_files")
        if not isinstance(target_files, list) or not target_files:
            target_files = sorted(_extract_diff_paths_safe(unified_diff))
        return CodeEditDraft(
            failure_mode=str(hypothesis.get("failure_mode") or "Previously generated miner code edit")[:700],
            mechanism=str(hypothesis.get("mechanism") or patch_doc.get("expected_improvement") or "")[:1000],
            expected_improvement=str(hypothesis.get("expected_improvement") or patch_doc.get("expected_improvement") or "")[:1000],
            risk=str(hypothesis.get("risk") or patch_doc.get("risk") or "")[:700],
            lane=str(patch_doc.get("lane") or "stale_parent_rebase")[:80],
            target_files=tuple(str(path) for path in target_files),
            unified_diff=unified_diff,
            redacted_summary=str(candidate.get("redacted_public_summary") or patch_summary)[:1200],
            test_plan=str(patch_doc.get("test_plan") or "Run the standard Research Lab private test command.")[:1200],
            rollback_plan=str(patch_doc.get("rollback_plan") or "Discard the rebased candidate image.")[:1200],
            predicted_delta=float(hypothesis.get("predicted_delta") or 1.0),
        )

    async def _repair_and_build_stale_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        active_artifact: PrivateModelArtifactManifest,
        original_error: CodeEditPatchApplyError,
        run_id: str,
    ) -> tuple[CodeEditDraft, Any]:
        if not self.config.stale_parent_rebase_repair_enabled:
            raise original_error
        api_key = os.getenv("RESEARCH_LAB_STALE_PARENT_REBASE_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise CodeEditBuildError("stale parent repair OpenRouter operator key is not configured")
        model_id = str(self.config.stale_parent_rebase_repair_model or "").strip()
        if not model_id:
            raise CodeEditBuildError("stale parent repair model is not configured")

        original_draft = await asyncio.to_thread(self._draft_from_stale_candidate, candidate)
        builder = CodeEditCandidateBuilder(self.config)
        with tempfile.TemporaryDirectory(prefix="research-lab-stale-rebase-") as tmp:
            source_context = await asyncio.to_thread(
                builder.prepare_parent_source_context,
                parent_artifact=active_artifact,
                workspace_dir=Path(tmp),
            )
            read_batch = resolve_source_inspection_requests(
                source_context,
                [
                    CodeEditSourceInspectionRequest(
                        operation="read_file",
                        path=path,
                        rationale="repair stale parent code-edit diff against current model source",
                    )
                    for path in original_draft.target_files
                ],
                already_read_paths=(),
                max_files=max(len(original_draft.target_files), self.config.code_edit_source_inspection_max_files),
                max_file_bytes=self.config.code_edit_source_inspection_file_bytes,
                max_total_bytes=self.config.code_edit_source_inspection_total_bytes,
                max_search_matches=self.config.code_edit_source_inspection_search_matches,
            )
            raw = await _call_operator_openrouter_json(
                api_key=api_key,
                model_id=model_id,
                messages=build_code_edit_repair_messages(
                    draft=original_draft,
                    apply_error=str(original_error),
                    source_inspection_context=read_batch.model_context,
                    runtime_source_context=source_context.prompt_context(),
                    budget_context={
                        "repair_context": "stale_parent_rebase",
                        "operator_funded": True,
                    },
                    repair_attempt=1,
                    max_candidates=1,
                ),
                timeout_seconds=self.config.stale_parent_rebase_repair_timeout_seconds,
            )
            repaired = parse_code_edit_repair_response(raw, original_draft=original_draft)[0]
            source_errors = builder.validate_draft_against_source_context(
                repaired,
                source_context,
                read_paths=read_batch.read_paths,
                require_read=True,
            )
            if source_errors:
                raise CodeEditBuildError("; ".join(source_errors))
            candidate_index = await self._next_rebase_candidate_index(run_id)
            build = await asyncio.to_thread(
                builder.build,
                draft=repaired,
                parent_artifact=active_artifact,
                run_id=run_id,
                candidate_index=candidate_index,
                source_context=source_context,
            )
            return repaired, build

    async def _next_rebase_candidate_index(self, run_id: str) -> int:
        rows = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id",
            filters=(("run_id", str(run_id)),),
            limit=1000,
        )
        return 1000 + len(rows)

    async def _candidate_private_holdout_gate(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        window_hash: str,
    ) -> dict[str, Any]:
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,private_model_manifest_hash,rolling_window_hash,"
                "benchmark_quality,evaluation_epoch,score_summary_doc,current_benchmark_status,created_at"
            ),
            filters=(
                ("private_model_manifest_hash", artifact.manifest_hash),
                ("rolling_window_hash", window_hash),
                ("current_benchmark_status", "completed"),
            ),
            order_by=(("created_at", True),),
            limit=10,
        )
        for row in rows:
            if not _private_benchmark_row_is_valid(row):
                continue
            gate = _private_holdout_gate_from_baseline_row(row)
            if gate:
                return gate
        raise CandidateBaselineNotReady(
            "matching_completed_private_baseline_required_before_candidate_private_holdout: "
            f"manifest={compact_ref(artifact.manifest_hash)} window={compact_ref(window_hash)}"
        )

    async def _daily_candidate_scoring_window_and_gate(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        now: datetime | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return the UTC day's frozen baseline window and private holdout gate.

        Candidate scoring must compete against the private baseline that owns
        the UTC day, not a freshly reselected "latest" ICP window. That keeps the
        daily baseline stable even if ICP rows are inserted or repaired after
        midnight.
        """
        today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,private_model_manifest_hash,rolling_window_hash,"
                "benchmark_quality,evaluation_epoch,score_summary_doc,current_benchmark_status,created_at"
            ),
            filters=(
                ("benchmark_date", today),
                ("private_model_manifest_hash", artifact.manifest_hash),
                ("current_benchmark_status", "completed"),
            ),
            order_by=(("created_at", True),),
            limit=25,
        )
        for row in rows:
            if not _private_benchmark_row_is_valid(row):
                continue
            gate = _private_holdout_gate_from_baseline_row(row)
            window_hash = str(row.get("rolling_window_hash") or "")
            if not gate or not window_hash:
                continue
            window = await self._reconstruct_rolling_window(window_hash)
            if window is None:
                raise CandidateBaselineNotReady(
                    "same_day_private_baseline_window_reconstruction_required_before_candidate_scoring: "
                    f"manifest={compact_ref(artifact.manifest_hash)} window={compact_ref(window_hash)}"
                )
            return window, gate
        raise CandidateBaselineNotReady(
            "same_day_completed_private_baseline_required_before_candidate_scoring: "
            f"date={today} manifest={compact_ref(artifact.manifest_hash)}"
        )

    async def _candidate_scoring_start_gate(self, *, now: datetime | None = None) -> dict[str, Any]:
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        quiet_start = int(
            getattr(
                self.config,
                "candidate_scoring_quiet_start_utc_seconds",
                _candidate_scoring_quiet_start_utc_seconds(),
            )
            or 0
        )
        quiet_start = max(0, min(86399, quiet_start))
        target_date = _candidate_baseline_target_date(now_utc, quiet_start_seconds=quiet_start)
        target_now = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        seconds_since_midnight = _utc_seconds_since_day_start(now_utc)
        quiet_window_active = seconds_since_midnight >= quiet_start
        try:
            active = await load_active_private_model(self.config, register_bootstrap=True)
            artifact = active.artifact
        except Exception as exc:  # noqa: BLE001 - fail closed before assigning a candidate
            return {
                "available": False,
                "reason": "candidate_scoring_active_model_unavailable",
                "now_utc": now_utc.isoformat(),
                "target_benchmark_date": target_date,
                "quiet_start_utc_seconds": quiet_start,
                "quiet_window_active": quiet_window_active,
                "error": _short_error(exc),
            }
        try:
            window, gate = await self._daily_candidate_scoring_window_and_gate(
                artifact=artifact,
                now=target_now,
            )
        except CandidateBaselineNotReady as exc:
            return {
                "available": False,
                "reason": (
                    "candidate_scoring_next_daily_baseline_not_ready"
                    if quiet_window_active
                    else "candidate_scoring_daily_baseline_not_ready"
                ),
                "now_utc": now_utc.isoformat(),
                "target_benchmark_date": target_date,
                "quiet_start_utc_seconds": quiet_start,
                "quiet_window_active": quiet_window_active,
                "private_model_manifest_hash": artifact.manifest_hash,
                "private_model_artifact_hash": artifact.model_artifact_hash,
                "error": _short_error(exc),
            }
        except Exception as exc:  # noqa: BLE001 - fail closed before assigning a candidate
            return {
                "available": False,
                "reason": "candidate_scoring_daily_baseline_gate_unavailable",
                "now_utc": now_utc.isoformat(),
                "target_benchmark_date": target_date,
                "quiet_start_utc_seconds": quiet_start,
                "quiet_window_active": quiet_window_active,
                "private_model_manifest_hash": artifact.manifest_hash,
                "private_model_artifact_hash": artifact.model_artifact_hash,
                "error": _short_error(exc),
            }
        return {
            "available": True,
            "reason": "",
            "now_utc": now_utc.isoformat(),
            "target_benchmark_date": target_date,
            "quiet_start_utc_seconds": quiet_start,
            "quiet_window_active": quiet_window_active,
            "private_model_manifest_hash": artifact.manifest_hash,
            "private_model_artifact_hash": artifact.model_artifact_hash,
            "rolling_window_hash": getattr(window, "window_hash", ""),
            "baseline_benchmark_bundle_id": str(gate.get("baseline_benchmark_bundle_id") or ""),
        }

    async def _check_candidate_scoring_freshness(
        self,
        *,
        parent_artifact: PrivateModelArtifactManifest,
        candidate_window_hash: str,
        progress: Mapping[str, Any],
    ) -> None:
        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash
        if active_parent != parent_artifact.model_artifact_hash:
            raise StaleParentDuringScoring(
                active_artifact=active.artifact,
                candidate_parent=parent_artifact.model_artifact_hash,
                progress=progress,
            )
        try:
            current_window, _gate = await self._daily_candidate_scoring_window_and_gate(
                artifact=parent_artifact,
            )
        except Exception as exc:
            if isinstance(exc, CandidateBaselineNotReady):
                raise
            raise CandidateBaselineNotReady(
                "candidate_daily_baseline_unavailable_during_candidate_scoring: "
                f"candidate_window={compact_ref(candidate_window_hash)}"
            ) from exc
        current_window_hash = str(getattr(current_window, "window_hash", "") or "")
        if current_window_hash and current_window_hash != candidate_window_hash:
            raise CandidateBaselineWindowChanged(
                candidate_window_hash=candidate_window_hash,
                current_window_hash=current_window_hash,
                progress=progress,
            )

    async def _maybe_run_private_baseline(self) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        baseline_start_offset = int(
            getattr(
                self.config,
                "baseline_start_utc_offset_seconds",
                _baseline_min_utc_day_delay_seconds(),
            )
            or 0
        )
        baseline_start_offset = max(0, min(86399, baseline_start_offset))
        min_start_at = utc_day_start(now) + timedelta(seconds=baseline_start_offset)
        if now < min_start_at:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE WAITING FOR DAILY ICP ACTIVATION",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Scheduled start", min_start_at.isoformat()),
                        ("Start offset seconds", baseline_start_offset),
                        ("Action", "deferring baseline so the UTC day's ICP set can activate"),
                    ),
                )
            )
            return {
                "status": "waiting_for_daily_icp_activation",
                "benchmark_date": today,
                "scheduled_start_at": min_start_at.isoformat(),
                "earliest_start_at": min_start_at.isoformat(),
                "start_offset_seconds": baseline_start_offset,
            }
        start = time.time()
        evaluation_epoch = await self._resolve_evaluation_epoch()
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE ALLOCATED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Proxy ref", self.proxy_ref_hash),
                    ("Benchmark date", today),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Eval days", self.config.lab_champion_eval_days),
                    ("ICPs per day", self.config.lab_champion_icps_per_day),
                    ("Expected ICPs", self.config.lab_champion_eval_days * self.config.lab_champion_icps_per_day),
                ),
            )
        )
        repo_head_sync = await sync_active_model_to_repo_head(
            self.config,
            actor_ref=self.worker_ref,
            dry_run=False,
            wait_for_repo_head=False,
        )
        if not bool(repo_head_sync.get("ok")):
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE BLOCKED: REPO HEAD MANIFEST NOT READY",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Status", repo_head_sync.get("status")),
                        ("Repo main SHA", str(repo_head_sync.get("repo_main_sha") or "")[:12] or "-"),
                        (
                            "current.json SHA",
                            str(repo_head_sync.get("current_json_git_sha") or "")[:12] or "-",
                        ),
                        (
                            "Active model SHA",
                            str(repo_head_sync.get("active_model_git_sha") or "")[:12] or "-",
                        ),
                        ("Action", "deferring daily benchmark; stale active lineage will not be benchmarked"),
                    ),
                )
            )
            return {
                "status": str(repo_head_sync.get("status") or "repo_head_sync_failed"),
                "benchmark_date": today,
                "repo_head_sync": repo_head_sync,
            }
        expected_fresh_set_id = utc_set_id_for_datetime(now)
        try:
            window = await fetch_rolling_icp_window(
                days=self.config.lab_champion_eval_days,
                icps_per_day=self.config.lab_champion_icps_per_day,
                **_rolling_window_fetch_kwargs(self.config),
                allow_partial=self.config.scoring_worker_allow_partial_icp_window,
                required_fresh_set_id=expected_fresh_set_id,
                require_fresh_set_active_at=now,
            )
        except RollingIcpWindowUnavailable as exc:
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE BLOCKED: DAILY ICP WINDOW NOT READY",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Expected fresh set", expected_fresh_set_id),
                        ("Reason", str(exc)[:240]),
                        ("Action", "retrying later; not falling back to a prior UTC day's ICP window"),
                    ),
                )
            )
            return {
                "status": "daily_icp_window_not_ready",
                "benchmark_date": today,
                "expected_fresh_set_id": expected_fresh_set_id,
                "error": str(exc),
            }
        active = await load_active_private_model(self.config, register_bootstrap=True)
        artifact = active.artifact
        existing = await select_many(
            "research_lab_private_model_benchmark_current",
            columns="*",
            filters=(
                ("benchmark_date", today),
                ("rolling_window_hash", window.window_hash),
                ("private_model_manifest_hash", artifact.manifest_hash),
            ),
            order_by=(("created_at", True),),
            limit=25,
        )
        valid_existing = [row for row in existing if _private_benchmark_row_is_valid(row)]
        if valid_existing:
            already_key = f"{today}:{window.window_hash}:{artifact.manifest_hash}"
            if self._baseline_already_logged_date != already_key:
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ALREADY BENCHMARKED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Rolling window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            ("Selected ICPs", len(window.item_refs)),
                            ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                        ),
                    )
                )
                self._baseline_already_logged_date = already_key
            return {
                "status": "already_benchmarked",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "private_model_manifest_hash": artifact.manifest_hash,
            }
        same_day_reference = await self._reusable_same_day_benchmark(
            today=today,
            manifest_hash=artifact.manifest_hash,
        )
        if same_day_reference is not None:
            reuse_key = f"{today}:reuse:{artifact.manifest_hash}"
            if self._baseline_already_logged_date != reuse_key:
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE SAME-DAY REFERENCE REUSED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Existing bundle", compact_ref(same_day_reference.get("benchmark_bundle_id"))),
                            ("Existing window", compact_ref(same_day_reference.get("rolling_window_hash"))),
                            ("Requested window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            (
                                "Action",
                                "NOT re-running: a same-day re-run silently replaces the day's promotion "
                                "reference (same-day deltas of 3+ points observed). Set "
                                "RESEARCH_LAB_BASELINE_ALLOW_SAMEDAY_REPLACE=true to replace deliberately.",
                            ),
                        ),
                    )
                )
                self._baseline_already_logged_date = reuse_key
            return {
                "status": "reused_same_day_benchmark",
                "benchmark_date": today,
                "benchmark_bundle_id": str(same_day_reference.get("benchmark_bundle_id") or ""),
                "rolling_window_hash": str(same_day_reference.get("rolling_window_hash") or ""),
                "private_model_manifest_hash": artifact.manifest_hash,
            }
        if _env_flag("RESEARCH_LAB_BASELINE_ANY_WORKER") and await self._baseline_leased_by_other_worker(today):
            return {"status": "baseline_leased_elsewhere", "benchmark_date": today}
        benchmark_attempt = _next_benchmark_attempt(existing)
        await create_rolling_icp_window(window)
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE STARTED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Proxy ref", self.proxy_ref_hash),
                    ("Rolling window", compact_ref(window.window_hash)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Selected sets", len(window.set_ids)),
                    ("Selected ICPs", len(window.item_refs)),
                    ("Private model", compact_ref(artifact.model_artifact_hash)),
                    ("Active model SHA", str(repo_head_sync.get("active_model_git_sha") or "")[:12] or "-"),
                    ("Repo main SHA", str(repo_head_sync.get("repo_main_sha") or "")[:12] or "-"),
                    ("current.json SHA", str(repo_head_sync.get("current_json_git_sha") or "")[:12] or "-"),
                    ("Active is repo head", repo_head_sync.get("active_is_repo_head")),
                    ("Concurrency", self.config.private_baseline_concurrency),
                    ("Benchmark Exa key", "dedicated" if self.config.benchmark_exa_api_key else "inherited"),
                    ("Exa RPS per container", self.config.benchmark_exa_max_rps or "inherited"),
                ),
            )
        )
        runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_baseline_scoring_env(),
                    run_type="private_baseline_rebenchmark",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=artifact.model_artifact_hash,
                    benchmark_date=today,
                    benchmark_attempt=benchmark_attempt,
                    evaluation_epoch=evaluation_epoch,
                    started_at=start,
                ),
            )
        )
        scorer = QualificationStyleCompanyScorer()
        # Trace scope (§5.4 + in-container capture): keyed per day, attempt,
        # and window so a deliberate same-day replacement never overwrites the
        # prior attempt's docs.
        baseline_window_tag = str(window.window_hash or "").removeprefix("sha256:")[:16] or "window"
        baseline_trace_context = self._baseline_trace_context(
            context_ref=f"daily-{today}-a{benchmark_attempt}-{baseline_window_tag}",
            manifest_uri=artifact.manifest_uri,
            run_type="private_baseline_rebenchmark",
            benchmark_date=today,
            rolling_window_hash=window.window_hash,
        )
        per_icp_summaries: list[dict[str, Any]] = []
        nonempty_output_count = 0
        retried_total = 0
        recovered_total = 0
        baseline_progress_rows: list[dict[str, Any]] = []
        baseline_progress_location = (
            _baseline_progress_s3_location(
                artifact.manifest_uri,
                benchmark_date=today,
                window_hash=window.window_hash,
                private_model_artifact_hash=artifact.model_artifact_hash,
            )
            if _per_icp_checkpoint_enabled()
            else None
        )
        if baseline_progress_location is not None:
            progress_bucket, progress_key = baseline_progress_location
            baseline_progress_rows = await asyncio.to_thread(
                _load_baseline_scoring_progress,
                progress_bucket,
                progress_key,
                benchmark_date=today,
                window_hash=window.window_hash,
                private_model_artifact_hash=artifact.model_artifact_hash,
            )
            if baseline_progress_rows:
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE PROGRESS RESUMED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Rolling window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            ("Completed ICPs", len(baseline_progress_rows)),
                            ("Action", "skipping already-checkpointed ICPs after restart"),
                        ),
                    )
                )

        baseline_progress_lock = asyncio.Lock()

        async def _checkpoint_completed_baseline_icp(row: Mapping[str, Any]) -> None:
            if baseline_progress_location is None or not _baseline_summary_checkpointable(row):
                return
            async with baseline_progress_lock:
                progress_bucket, progress_key = baseline_progress_location
                public_row = _baseline_progress_public_row(row)
                by_ref = _progress_rows_by_icp_ref(baseline_progress_rows)
                ref = str(public_row.get("icp_ref") or public_row.get("icp_hash") or "")
                if not ref:
                    return
                by_ref[ref] = public_row
                baseline_progress_rows[:] = list(by_ref.values())
                await asyncio.to_thread(
                    _store_baseline_scoring_progress,
                    progress_bucket,
                    progress_key,
                    benchmark_date=today,
                    window_hash=window.window_hash,
                    private_model_artifact_hash=artifact.model_artifact_hash,
                    rows=baseline_progress_rows,
                )

        try:
            lease_event = await create_scoring_dispatch_event(
                dispatch_type="private_baseline_rebenchmark",
                dispatch_status="assigned",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                rolling_window_hash=window.window_hash,
                event_doc={
                    "benchmark_date": today,
                    "benchmark_attempt": benchmark_attempt,
                    "selected_icp_count": len(window.item_refs),
                },
            )
            # Post-write lease confirm (claim-guard pattern, like
            # _claim_next_candidate's write-then-verify): two workers can both
            # pass the pre-check before either's `assigned` event lands. The
            # earliest unexpired open lease wins; the loser backs off silently
            # (writing a failed/completed release here would clear the
            # winner's lease for third-party observers). Only meaningful when
            # any-worker baselines are enabled — worker 0 has no peers.
            if _env_flag("RESEARCH_LAB_BASELINE_ANY_WORKER") and not await self._confirm_baseline_lease(
                today=today,
                lease_event=lease_event,
            ):
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE LEASE LOST",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Action", "another worker holds the earlier lease; backing off"),
                        ),
                    )
                )
                return {"status": "baseline_leased_elsewhere", "benchmark_date": today}
            total_icps = len(window.benchmark_items)
            parallel_mode = self.config.private_baseline_concurrency > 1
            if parallel_mode:
                retry_runner = DockerPrivateModelRunner(
                    DockerPrivateModelSpec(
                        image_digest=artifact.image_digest,
                        timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                        env_passthrough=self._private_model_env_passthrough(),
                        extra_env=self._with_provider_cost_evaluation_scope(
                            self._private_baseline_retry_scoring_env(),
                            run_type="private_baseline_rebenchmark",
                            rolling_window_hash=window.window_hash,
                            artifact_hash=artifact.model_artifact_hash,
                            benchmark_date=today,
                            benchmark_attempt=benchmark_attempt,
                            evaluation_epoch=evaluation_epoch,
                            started_at=start,
                        ),
                        # The first-pass runner already pulled this digest.
                        pull_before_run=False,
                    )
                )
                batch_summaries, retry_stats = await self._run_baseline_batch(
                    runner=runner,
                    retry_runner=retry_runner,
                    scorer=scorer,
                    window=window,
                    run_start=start,
                    trace_context=baseline_trace_context,
                    resume_results=baseline_progress_rows,
                    icp_checkpoint=_checkpoint_completed_baseline_icp,
                )
                retried_total = int(retry_stats.get("retried") or 0)
                recovered_total = int(retry_stats.get("recovered") or 0)
                for item_summary in batch_summaries:
                    if item_summary.pop("_nonempty", False):
                        nonempty_output_count += 1
                    item_summary.pop("_item_index", None)
                    item_summary.pop("_retryable", None)
                    item_summary.pop("_runtime_error", None)
                    item_summary.pop("_retry_backoff_seconds", None)
                    per_icp_summaries.append(item_summary)
            baseline_resume_by_ref = _progress_rows_by_icp_ref(baseline_progress_rows)
            for item_index, item in enumerate([] if parallel_mode else window.benchmark_items, start=1):
                item_start = time.time()
                label = str(item.get("icp_ref") or item.get("icp_hash") or "unknown_icp")
                resumed_summary = baseline_resume_by_ref.get(label)
                if resumed_summary is not None:
                    logger.info(
                        format_worker_block(
                            "RESEARCH LAB PRIVATE BASELINE ICP RESUMED FROM CHECKPOINT",
                            (
                                ("Worker", self.worker_ref),
                                ("ICP", f"{item_index}/{total_icps}"),
                                ("ICP ref", compact_ref(label)),
                                ("Rolling window", compact_ref(window.window_hash)),
                            ),
                        )
                    )
                    if _baseline_summary_nonempty(resumed_summary):
                        nonempty_output_count += 1
                    per_icp_summaries.append(dict(resumed_summary))
                    continue
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP STARTED",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Set", item.get("set_id")),
                            ("Day", item.get("day_index")),
                            ("Day rank", item.get("day_rank")),
                        ),
                    )
                )
                runtime_error = ""
                # In-container trace collection (§9.1): asyncio.to_thread copies
                # the contextvars context, so the runner thread sees the
                # collector installed here.
                serial_trace_entries: list[dict[str, Any]] | None = None
                serial_trace_token: contextvars.Token | None = None
                if incontainer_trace_capture_enabled():
                    serial_trace_entries, serial_trace_token = begin_incontainer_trace_collection()
                try:
                    outputs = ensure_private_model_outputs(
                        await asyncio.to_thread(runner, item["icp"], {"mode": "private_baseline"}),
                        context_label=f"private baseline for {label}",
                        require_non_empty=False,
                    )
                except PrivateModelRuntimeError as exc:
                    outputs = []
                    runtime_error = _short_error(exc)
                    logger.warning(
                        format_worker_block(
                            "RESEARCH LAB PRIVATE BASELINE ICP RUNTIME ERROR",
                            (
                                ("Worker", self.worker_ref),
                                ("ICP", f"{item_index}/{total_icps}"),
                                ("ICP ref", compact_ref(label)),
                                ("ICP hash", compact_ref(item.get("icp_hash"))),
                                ("Error", runtime_error),
                            ),
                        )
                    )
                finally:
                    if serial_trace_token is not None:
                        end_incontainer_trace_collection(serial_trace_token)
                item_elapsed = time.time() - item_start
                if outputs:
                    nonempty_output_count += 1
                score_breakdowns: list[dict[str, Any]] = []
                scorer_error = ""
                if outputs:
                    try:
                        score_breakdowns = await self._score_baseline_outputs(
                            scorer=scorer,
                            outputs=outputs,
                            icp=item["icp"],
                            apply_isolation=True,
                        )
                    except Exception as scorer_exc:  # noqa: BLE001 - §0-N6: non-fatal per ICP
                        scorer_error = _short_error(scorer_exc)
                        logger.warning(
                            format_worker_block(
                                "RESEARCH LAB PRIVATE BASELINE ICP SCORER ERROR",
                                (
                                    ("Worker", self.worker_ref),
                                    ("ICP", f"{item_index}/{total_icps}"),
                                    ("ICP ref", compact_ref(label)),
                                    ("ICP hash", compact_ref(item.get("icp_hash"))),
                                    ("Error", scorer_error),
                                ),
                            )
                        )
                scores = [float(row.get("final_score", 0.0) or 0.0) for row in score_breakdowns]
                icp_score = _average(scores)
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP SCORED",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Set", item.get("set_id")),
                            ("Day", item.get("day_index")),
                            ("Day rank", item.get("day_rank")),
                            ("Score", f"{icp_score:.4f}"),
                            ("Companies", len(scores)),
                            ("Non-empty output", bool(outputs)),
                            ("Runtime error", runtime_error or "-"),
                            ("Scorer error", scorer_error or "-"),
                            ("ICP runtime", f"{item_elapsed:.1f}s"),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                if (
                    item_index >= PRIVATE_BASELINE_FAST_EMPTY_ABORT_AFTER
                    and nonempty_output_count <= 0
                    and time.time() - start < PRIVATE_BASELINE_FAST_EMPTY_ABORT_SECONDS
                ):
                    raise PrivateModelRuntimeError(
                        "private baseline fast-empty guard tripped: "
                        f"first {item_index} ICPs returned zero companies in {time.time() - start:.1f}s. "
                        "The private model is not executing the full provider-backed sourcing path; "
                        "check Docker env passthrough, provider keys, proxy connectivity, and ICP canonicalization."
                    )
                item_summary = sanitize_benchmark_item_summary(
                    item=item,
                    score=icp_score,
                    company_count=len(scores),
                    score_breakdowns=score_breakdowns,
                    # Model output count BEFORE the scorer's employee-bucket
                    # pre-filter, so the funnel's first stage is the true
                    # "companies discovered" number.
                    sourced_count=len(outputs),
                )
                if runtime_error or scorer_error:
                    diagnostics = dict(item_summary.get("diagnostics") or {})
                    runtime_diagnostics = _runtime_error_diagnostics(runtime_error or scorer_error)
                    categories = set(diagnostics.get("failure_categories") or [])
                    if runtime_error:
                        categories.add("runtime_provider_error")
                    if scorer_error:
                        categories.add("scorer_provider_error")
                    categories.add(str(runtime_diagnostics["category"]))
                    diagnostics["failure_categories"] = sorted(categories)
                    diagnostics["runtime_error"] = runtime_diagnostics
                    item_summary["diagnostics"] = diagnostics
                await self._record_baseline_icp_traces(
                    item=item,
                    item_summary=item_summary,
                    outputs=outputs,
                    score_breakdowns=score_breakdowns,
                    trace_entries=serial_trace_entries,
                    trace_context=baseline_trace_context,
                )
                _apply_provider_cost_baseline_outcome(item_summary)
                await _checkpoint_completed_baseline_icp(item_summary)
                per_icp_summaries.append(item_summary)
            if nonempty_output_count <= 0:
                raise PrivateModelRuntimeError(
                    f"private baseline returned zero companies across all {len(window.benchmark_items)} ICPs"
                )
            baseline_health = _build_baseline_health(
                per_icp_summaries=per_icp_summaries,
                retried=retried_total,
                recovered=recovered_total,
                max_unresolved_icps=_baseline_max_unresolved_icps(),
            )
            aggregate_score = _average([summary["score"] for summary in per_icp_summaries])
            day_jump_points = await self._baseline_day_jump_points(
                today=today,
                aggregate_score=aggregate_score,
            )
            if day_jump_points is not None:
                baseline_health["day_jump_points"] = round(day_jump_points, 4)
            max_day_jump = _baseline_max_day_jump_points()
            if (
                max_day_jump is not None
                and day_jump_points is not None
                and abs(day_jump_points) > max_day_jump
            ):
                raise BaselineHealthGateFailure(
                    "baseline_day_over_day_jump_gate_failed: "
                    f"jump={day_jump_points:+.4f} max=±{max_day_jump} "
                    f"aggregate={aggregate_score:.4f}; "
                    "refusing to record this run as the day's benchmark reference",
                    baseline_health={**baseline_health, "gate_passed": False},
                )
        except Exception as exc:
            await create_scoring_dispatch_event(
                dispatch_type="private_baseline_rebenchmark",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                rolling_window_hash=window.window_hash,
                event_doc={
                    "benchmark_date": today,
                    "benchmark_attempt": benchmark_attempt,
                    "selected_icp_count": len(window.item_refs),
                    "error_diagnostics": _event_error_diagnostics(exc),
                    "baseline_health": (
                        dict(exc.baseline_health)
                        if isinstance(exc, BaselineHealthGateFailure)
                        else None
                    ),
                    "elapsed_seconds": round(time.time() - start, 3),
                },
            )
            logger.exception(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Rolling window", compact_ref(window.window_hash)),
                        ("Evaluation epoch", evaluation_epoch),
                        ("Attempt", benchmark_attempt),
                        ("Error", str(exc)[:300]),
                    ),
                )
            )
            return {
                "status": "failed",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "error": str(exc)[:300],
            }
        pre_record_conflict = await self._same_day_reference_recorded_while_running(
            today=today,
            window_hash=window.window_hash,
            manifest_hash=artifact.manifest_hash,
        )
        if pre_record_conflict is not None:
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE COMPLETED BUT REFERENCE ALREADY RECORDED",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Existing bundle", compact_ref(pre_record_conflict.get("benchmark_bundle_id"))),
                        ("Action", "discarding this run's recording to keep the day's reference stable"),
                    ),
                )
            )
            return {
                "status": "already_benchmarked",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "private_model_manifest_hash": artifact.manifest_hash,
            }
        visibility_split = build_benchmark_visibility_split(
            rolling_window_hash=window.window_hash,
            benchmark_items=window.benchmark_items,
            per_icp_summaries=per_icp_summaries,
            public_icps_per_day=self.config.public_benchmark_public_icps_per_day,
            public_weak_per_day=self.config.public_benchmark_public_weak_per_day,
            public_total_icps=self.config.public_benchmark_public_total_icps,
            public_weak_total=self.config.public_benchmark_public_weak_total,
        )
        score_summary_doc = {
            "schema_version": "1.0",
            "benchmark_quality": "passed",
            "benchmark_attempt": benchmark_attempt,
            "rolling_window_hash": window.window_hash,
            "per_icp_summaries": per_icp_summaries,
            "visibility_split": visibility_split,
            "aggregate_score": aggregate_score,
            "baseline_health": baseline_health,
            "elapsed_seconds": round(time.time() - start, 3),
        }
        bundle_hash = canonical_hash(score_summary_doc)
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=bundle_hash,
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        bundle, _event = await create_private_model_benchmark_bundle(
            benchmark_date=today,
            private_model_artifact_hash=artifact.model_artifact_hash,
            private_model_manifest_hash=artifact.manifest_hash,
            rolling_window_hash=window.window_hash,
            evaluation_epoch=evaluation_epoch,
            benchmark_attempt=benchmark_attempt,
            benchmark_quality="passed",
            aggregate_score=aggregate_score,
            scoring_worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            signature_ref=signature_ref,
            score_summary_doc=score_summary_doc,
        )
        await create_scoring_dispatch_event(
            dispatch_type="private_baseline_rebenchmark",
            dispatch_status="completed",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            rolling_window_hash=window.window_hash,
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            event_doc={
                "benchmark_date": today,
                "elapsed_seconds": round(time.time() - start, 3),
                "selected_icp_count": len(window.item_refs),
                "public_icp_count": int(visibility_split.get("public_count") or 0),
                "private_holdout_icp_count": int(visibility_split.get("private_count") or 0),
            },
        )
        public_report_doc = build_public_benchmark_report(
            benchmark_date=today,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            per_icp_summaries=per_icp_summaries,
            benchmark_items=window.benchmark_items,
            public_icps_per_day=self.config.public_benchmark_public_icps_per_day,
            public_weak_per_day=self.config.public_benchmark_public_weak_per_day,
            public_total_icps=self.config.public_benchmark_public_total_icps,
            public_weak_total=self.config.public_benchmark_public_weak_total,
        )
        public_report, _report_event = await create_public_benchmark_report(
            benchmark_date=today,
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            private_model_artifact_hash=artifact.model_artifact_hash,
            private_model_manifest_hash=artifact.manifest_hash,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            benchmark_attempt=benchmark_attempt,
            benchmark_quality="passed",
            report_doc=public_report_doc,
        )
        await self._write_audit_bundle(evaluation_epoch)
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE COMPLETED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Benchmark bundle", compact_ref(bundle["benchmark_bundle_id"])),
                    ("Public report", compact_ref(public_report["report_id"])),
                    ("Rolling window", compact_ref(window.window_hash)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Selected ICPs", len(window.item_refs)),
                    ("Attempt", benchmark_attempt),
                    ("Public ICPs", visibility_split.get("public_count")),
                    ("Private holdout ICPs", visibility_split.get("private_count")),
                    ("Public strength", visibility_split.get("public_strength_counts")),
                    ("Private strength", visibility_split.get("private_strength_counts")),
                    ("Aggregate score", f"{aggregate_score:.4f}"),
                    ("Unresolved provider errors", baseline_health.get("unresolved_provider_errors")),
                    ("Baseline health", "healthy" if baseline_health.get("gate_passed") else "degraded"),
                    ("Health decision", baseline_health.get("decision") or "observe_only"),
                    ("Day jump", f"{day_jump_points:+.4f}" if day_jump_points is not None else "-"),
                    ("Elapsed", f"{time.time() - start:.1f}s"),
                ),
            )
        )
        return {
            "status": "completed",
            "benchmark_date": today,
            "benchmark_bundle_id": str(bundle["benchmark_bundle_id"]),
            "public_report_id": str(public_report["report_id"]),
            "rolling_window_hash": window.window_hash,
        }

    def _get_scorer_trace_recorder(self) -> _ScorerTraceRecorder:
        recorder = getattr(self, "_scorer_trace_recorder", None)
        if recorder is None:
            recorder = _ScorerTraceRecorder(self.config)
            self._scorer_trace_recorder = recorder
        return recorder

    def _baseline_trace_context(
        self,
        *,
        context_ref: str,
        manifest_uri: str,
        run_type: str = "private_baseline_rebenchmark",
        benchmark_date: str = "",
        rolling_window_hash: str = "",
    ) -> dict[str, Any]:
        """Shared trace scope for one baseline/champion batch: keys the §5.4
        scorer-trace docs and the in-container trace docs, and carries the
        batch's log-once drop state for unconfigured S3."""
        return {
            "context_ref": str(context_ref),
            "manifest_uri": str(manifest_uri or ""),
            "run_type": str(run_type or "private_baseline_rebenchmark"),
            "benchmark_date": str(benchmark_date or ""),
            "rolling_window_hash": str(rolling_window_hash or ""),
            "incontainer_drop_state": {"logged": False, "dropped_entries": 0},
        }

    async def _record_baseline_icp_traces(
        self,
        *,
        item: Mapping[str, Any],
        item_summary: dict[str, Any],
        outputs: list[Any],
        score_breakdowns: list[dict[str, Any]],
        trace_entries: list[dict[str, Any]] | None,
        trace_context: Mapping[str, Any] | None,
    ) -> None:
        """Best-effort §5.4 scorer-trace capture plus in-container trace
        publication for one baseline/champion ICP.

        Mutates ONLY ``item_summary['diagnostics']`` with pointer fields
        (``scorer_trace_ref``/``scorer_trace_sha256`` and the
        ``incontainer_trace_*`` trio) — never content — and never raises.
        With capture flags off (or ``trace_context`` unset) the summary stays
        byte-identical to the pre-capture shape.
        """
        if trace_context is None:
            return
        try:
            label = str(item.get("icp_ref") or item.get("icp_hash") or "unknown_icp")
            context_ref = str(trace_context.get("context_ref") or "baseline")
            diagnostics_updates: dict[str, Any] = {}
            if score_breakdowns:
                pointer = self._get_scorer_trace_recorder().capture(
                    context_ref=context_ref,
                    icp_ref=label,
                    icp_hash=str(item.get("icp_hash") or ""),
                    outputs=outputs,
                    breakdowns=score_breakdowns,
                    is_reference_model=True,
                    manifest_uri=str(trace_context.get("manifest_uri") or ""),
                )
                if pointer:
                    diagnostics_updates["scorer_trace_ref"] = pointer["s3_ref"]
                    diagnostics_updates["scorer_trace_sha256"] = pointer["sha256"]
                # Persist model-sourced companies the harness rejected, for later
                # false-rejection analysis (best-effort; never blocks scoring).
                await _persist_rejected_companies(
                    context_ref=context_ref,
                    icp_ref=label,
                    icp_hash=str(item.get("icp_hash") or ""),
                    is_reference_model=True,
                    outputs=outputs,
                    breakdowns=score_breakdowns,
                )
            if trace_entries:
                drop_state = trace_context.get("incontainer_drop_state")
                await _persist_provider_cost_events(
                    entries=trace_entries,
                    run_type=str(trace_context.get("run_type") or "private_baseline_rebenchmark"),
                    icp_ref=label,
                    icp_hash=str(item.get("icp_hash") or ""),
                    runner_role="baseline",
                    benchmark_date=str(trace_context.get("benchmark_date") or ""),
                    rolling_window_hash=str(trace_context.get("rolling_window_hash") or ""),
                )
                diagnostics_updates.update(
                    await _publish_baseline_incontainer_trace(
                        context_ref=context_ref,
                        icp_ref=label,
                        entries=trace_entries,
                        drop_state=drop_state if isinstance(drop_state, dict) else {},
                    )
                )
            if diagnostics_updates:
                diagnostics = dict(item_summary.get("diagnostics") or {})
                diagnostics.update(diagnostics_updates)
                item_summary["diagnostics"] = diagnostics
        except Exception:  # noqa: BLE001 - capture can never affect the batch
            logger.warning(
                "research_lab_baseline_trace_record_failed icp_ref=%s",
                compact_ref(str(item.get("icp_ref") or "")),
                exc_info=True,
            )

    def _candidate_incontainer_trace_sink(self, candidate_id: str) -> Any:
        """Candidate-scoped in-container trace/cost sink for the evaluator.

        Provider-cost events must be persisted even when full in-container
        trace upload is disabled. When an S3 prefix is configured, uploads key
        by candidate (``{prefix}/{candidate_id}/{icp_ref}.json``) instead of
        the default run-scoped ref, so one candidate's traces stay enumerable
        under a single prefix; the evaluator places the returned uri into the
        row's ``incontainer_trace_ref``. When no encrypted S3 sink is
        available, this still persists cost events and returns an empty ref so
        the evaluator records dropped trace counts plus cost summaries. The
        ``RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE`` kill switch still overrides
        injected sinks inside the evaluator.
        """
        prefix = str(os.getenv(INCONTAINER_TRACE_S3_PREFIX_ENV) or "").strip().rstrip("/")
        kms_key_id = str(os.getenv(INCONTAINER_TRACE_KMS_KEY_ENV) or "").strip()
        upload_enabled = prefix.startswith("s3://") and bool(kms_key_id)
        if prefix.startswith("s3://") and not kms_key_id:
            # P5/P13: prefix-on/key-off must never write unencrypted; fall back
            # to dropped trace refs while still writing provider-cost events.
            logger.error(
                "research_lab_incontainer_trace_sink_refused candidate=%s reason=missing_kms_key "
                "(set %s so candidate in-container traces are written SSE-KMS encrypted)",
                compact_ref(candidate_id),
                INCONTAINER_TRACE_KMS_KEY_ENV,
            )
        safe_candidate = _trace_path_segment(candidate_id, fallback="candidate")

        async def _sink(icp_ref: str, entries: list[dict[str, Any]]) -> str:
            await _persist_provider_cost_events(
                entries=entries,
                run_type="candidate_scoring",
                icp_ref=str(icp_ref),
                runner_role="candidate",
                candidate_id=candidate_id,
            )
            if not upload_enabled:
                return ""
            return await asyncio.to_thread(
                _upload_incontainer_trace_doc,
                prefix,
                safe_candidate,
                str(icp_ref),
                list(entries),
                kms_key_id,
            )

        return _sink

    async def _run_baseline_icp(
        self,
        *,
        runner: DockerPrivateModelRunner,
        scorer: QualificationStyleCompanyScorer,
        item: Mapping[str, Any],
        item_index: int,
        total_icps: int,
        run_start: float,
        executor: concurrent.futures.Executor,
        scorer_semaphore: asyncio.Semaphore | None = None,
        mode_label: str = "private_baseline",
        trace_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one benchmark ICP (docker sourcing + scoring) and build its summary.

        Mirrors the serial baseline loop body, minus the fast-empty guard (its
        sequential-completion assumption does not hold under concurrency). The
        blocking docker wait runs on the dedicated baseline executor instead of
        asyncio.to_thread so many concurrent ~10-minute subprocess waits cannot
        starve the loop's default executor (which the KMS signing call uses).

        A scorer-side provider failure is NOT fatal for the batch (§0-N6): the
        scorer call is retried once inline, then the ICP is marked unresolved
        (retryable per the runner-error classifier) so retry rounds and
        observe-only baseline health diagnostics see it instead of the whole
        batch dying.

        The returned summary carries underscore-prefixed orchestration fields
        (_item_index, _retryable, _nonempty, _runtime_error,
        _retry_backoff_seconds) that MUST be popped before the summary enters
        score_summary_doc.
        """
        loop = asyncio.get_running_loop()
        item_start = time.time()
        label = str(item.get("icp_ref") or item.get("icp_hash") or "unknown_icp")
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE ICP STARTED",
                (
                    ("Worker", self.worker_ref),
                    ("ICP", f"{item_index}/{total_icps}"),
                    ("ICP ref", compact_ref(label)),
                    ("ICP hash", compact_ref(item.get("icp_hash"))),
                    ("Set", item.get("set_id")),
                    ("Day", item.get("day_index")),
                    ("Day rank", item.get("day_rank")),
                ),
            )
        )
        runtime_error = ""
        scorer_error = ""
        retryable = False
        retry_backoff_seconds = 0.0
        # In-container trace collection (§9.1): install a per-task collector so
        # the runner's stderr trace markers are published instead of dropped.
        # run_in_executor does NOT copy contextvars (asyncio.to_thread does), so
        # the context is copied explicitly AFTER the collector is installed —
        # the executor thread then appends to the same list object.
        trace_entries: list[dict[str, Any]] | None = None
        trace_token: contextvars.Token | None = None
        if trace_context is not None and incontainer_trace_capture_enabled():
            trace_entries, trace_token = begin_incontainer_trace_collection()
        try:
            runner_call: Any = functools.partial(runner, item["icp"], {"mode": mode_label})
            if trace_token is not None:
                runner_call = functools.partial(contextvars.copy_context().run, runner_call)
            outputs = ensure_private_model_outputs(
                await loop.run_in_executor(executor, runner_call),
                context_label=f"{mode_label} for {label}",
                require_non_empty=False,
            )
        except PrivateModelRuntimeError as exc:
            outputs = []
            runtime_error = _short_error(exc)
            # Classify from the full exception text: _short_error truncates to
            # 300 chars and can drop the status marker the classifier needs.
            retryable = _baseline_error_is_retryable(str(exc))
            retry_backoff_seconds = max(
                retry_backoff_seconds,
                _baseline_429_retry_backoff_seconds(str(exc)),
            )
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE ICP RUNTIME ERROR",
                    (
                        ("Worker", self.worker_ref),
                        ("ICP", f"{item_index}/{total_icps}"),
                        ("ICP ref", compact_ref(label)),
                        ("ICP hash", compact_ref(item.get("icp_hash"))),
                        ("Retryable", retryable),
                        ("Error", runtime_error),
                    ),
                )
            )
        finally:
            if trace_token is not None:
                end_incontainer_trace_collection(trace_token)
        item_elapsed = time.time() - item_start
        score_breakdowns: list[dict[str, Any]] = []
        if outputs:
            try:
                score_breakdowns = await self._score_baseline_outputs(
                    scorer=scorer,
                    outputs=outputs,
                    icp=item["icp"],
                    scorer_semaphore=scorer_semaphore,
                )
            except Exception as scorer_exc:  # noqa: BLE001 - §0-N6: non-fatal for the batch
                scorer_error = _short_error(scorer_exc)
                retryable = retryable or _baseline_error_is_retryable(str(scorer_exc))
                retry_backoff_seconds = max(
                    retry_backoff_seconds,
                    _baseline_429_retry_backoff_seconds(str(scorer_exc)),
                )
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP SCORER ERROR",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Retryable", retryable),
                            ("Error", scorer_error),
                        ),
                    )
                )
        scores = [float(row.get("final_score", 0.0) or 0.0) for row in score_breakdowns]
        icp_score = _average(scores)
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE ICP SCORED",
                (
                    ("Worker", self.worker_ref),
                    ("ICP", f"{item_index}/{total_icps}"),
                    ("ICP ref", compact_ref(label)),
                    ("ICP hash", compact_ref(item.get("icp_hash"))),
                    ("Set", item.get("set_id")),
                    ("Day", item.get("day_index")),
                    ("Day rank", item.get("day_rank")),
                    ("Score", f"{icp_score:.4f}"),
                    ("Companies", len(scores)),
                    ("Non-empty output", bool(outputs)),
                    ("Runtime error", runtime_error or "-"),
                    ("Scorer error", scorer_error or "-"),
                    ("ICP runtime", f"{item_elapsed:.1f}s"),
                    ("Elapsed", f"{time.time() - run_start:.1f}s"),
                ),
            )
        )
        item_summary = sanitize_benchmark_item_summary(
            item=item,
            score=icp_score,
            company_count=len(scores),
            score_breakdowns=score_breakdowns,
            # Model output count BEFORE the scorer's employee-bucket
            # pre-filter, so the funnel's first stage is the true
            # "companies discovered" number.
            sourced_count=len(outputs),
        )
        if runtime_error or scorer_error:
            diagnostics = dict(item_summary.get("diagnostics") or {})
            runtime_diagnostics = _runtime_error_diagnostics(runtime_error or scorer_error)
            categories = set(diagnostics.get("failure_categories") or [])
            if runtime_error:
                categories.add("runtime_provider_error")
            if scorer_error:
                categories.add("scorer_provider_error")
            categories.add(str(runtime_diagnostics["category"]))
            diagnostics["failure_categories"] = sorted(categories)
            diagnostics["runtime_error"] = runtime_diagnostics
            item_summary["diagnostics"] = diagnostics
        await self._record_baseline_icp_traces(
            item=item,
            item_summary=item_summary,
            outputs=outputs,
            score_breakdowns=score_breakdowns,
            trace_entries=trace_entries,
            trace_context=trace_context,
        )
        _apply_provider_cost_baseline_outcome(item_summary)
        diagnostics = item_summary.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            categories = {str(value) for value in diagnostics.get("failure_categories") or []}
            if categories.intersection({"provider_cost_cap_blocked", "provider_cost_tracking_failed"}):
                retryable = False
        item_summary["_item_index"] = item_index
        item_summary["_retryable"] = retryable
        item_summary["_nonempty"] = bool(outputs)
        item_summary["_runtime_error"] = runtime_error or scorer_error
        item_summary["_retry_backoff_seconds"] = retry_backoff_seconds
        return item_summary

    async def _score_baseline_outputs(
        self,
        *,
        scorer: QualificationStyleCompanyScorer,
        outputs: list[Any],
        icp: Mapping[str, Any],
        scorer_semaphore: asyncio.Semaphore | None = None,
        apply_isolation: bool = False,
    ) -> list[dict[str, Any]]:
        """Baseline-batch scorer call with the §0-N6 semantics.

        * optional benchmark scorer-key isolation (``apply_isolation`` — the
          serial loop applies it per call; the parallel batch applies ONE
          batch-wide scope instead, because per-call enter/exit under
          concurrency would restore prod keys mid-flight for sibling tasks);
        * optional burst semaphore (RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY);
        * one inline retry for transient scorer provider errors (classified by
          the same rules as runner errors).
        """

        async def _call() -> list[dict[str, Any]]:
            if scorer_semaphore is None:
                return await scorer.score_with_breakdowns(outputs, icp, True)
            async with scorer_semaphore:
                return await scorer.score_with_breakdowns(outputs, icp, True)

        async def _guarded() -> list[dict[str, Any]]:
            try:
                return await _call()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _baseline_error_is_retryable(str(exc)):
                    raise
                logger.warning(
                    "research_lab_baseline_scorer_retry worker_ref=%s error=%s",
                    self.worker_ref,
                    _short_error(exc),
                )
                backoff_seconds = _baseline_429_retry_backoff_seconds(str(exc))
                if backoff_seconds > 0:
                    logger.warning(
                        "research_lab_baseline_scorer_rate_limit_backoff worker_ref=%s backoff_seconds=%.1f error=%s",
                        self.worker_ref,
                        backoff_seconds,
                        _short_error(exc),
                    )
                    await asyncio.sleep(backoff_seconds)
                else:
                    await asyncio.sleep(2.0)
                return await _call()

        if not apply_isolation:
            return await _guarded()
        with _benchmark_scorer_isolation():
            return await _guarded()

    async def _run_baseline_batch(
        self,
        *,
        runner: DockerPrivateModelRunner,
        retry_runner: DockerPrivateModelRunner,
        scorer: QualificationStyleCompanyScorer,
        window: Any,
        run_start: float,
        mode_label: str = "private_baseline",
        trace_context: Mapping[str, Any] | None = None,
        resume_results: list[dict[str, Any]] | None = None,
        icp_checkpoint: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Run all benchmark ICPs concurrently, then retry transient failures.

        The whole batch (first pass + retry rounds) runs inside ONE benchmark
        scorer-key isolation scope (§0-N6): scorer calls burst on the dedicated
        benchmark Scrapingdog/OpenRouter keys when configured, container runs
        keep their construction-time prod keys via extra_env.
        """
        with _benchmark_scorer_isolation():
            return await self._run_baseline_batch_inner(
                runner=runner,
                retry_runner=retry_runner,
                scorer=scorer,
                window=window,
                run_start=run_start,
                mode_label=mode_label,
                trace_context=trace_context,
                resume_results=resume_results,
                icp_checkpoint=icp_checkpoint,
            )

    async def _run_baseline_batch_inner(
        self,
        *,
        runner: DockerPrivateModelRunner,
        retry_runner: DockerPrivateModelRunner,
        scorer: QualificationStyleCompanyScorer,
        window: Any,
        run_start: float,
        mode_label: str = "private_baseline",
        trace_context: Mapping[str, Any] | None = None,
        resume_results: list[dict[str, Any]] | None = None,
        icp_checkpoint: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Run all benchmark ICPs concurrently, then retry transient failures.

        Returns the ordered per-ICP summaries plus retry stats
        (``retried``/``recovered``/``unresolved``) for baseline health diagnostics.

        First pass fans out at private_baseline_concurrency. Provider/infra
        errors classified retryable are re-run for up to
        private_baseline_provider_retry_rounds at the lower retry concurrency
        (with the aggregate Exa budget re-spread by the retry runner env), so a
        transient 429 never leaves a valid ICP scored 0. ICPs that still fail
        keep their error and score 0 — aggregate semantics are unchanged.

        Results are reassembled in benchmark_items order: score_summary_doc is
        canonically hashed and the promotion gate consumes that hash, so
        ordering must be deterministic regardless of completion order.
        """
        concurrency = self.config.private_baseline_concurrency
        items = list(enumerate(window.benchmark_items, start=1))
        total_icps = len(items)
        resumed_by_ref = _progress_rows_by_icp_ref(resume_results)
        results: dict[int, dict[str, Any]] = {}
        run_items: list[tuple[int, Mapping[str, Any]]] = []
        for item_index, item in items:
            ref = _benchmark_item_ref_for_progress(item)
            resumed = resumed_by_ref.get(ref)
            if resumed is None:
                run_items.append((item_index, item))
                continue
            restored = dict(resumed)
            restored["_item_index"] = item_index
            restored["_retryable"] = False
            restored["_nonempty"] = _baseline_summary_nonempty(restored)
            restored["_runtime_error"] = ""
            restored["_retry_backoff_seconds"] = 0.0
            results[item_index] = restored
        if resumed_by_ref:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE PARALLEL RESUME",
                    (
                        ("Worker", self.worker_ref),
                        ("Resumed ICPs", len(results)),
                        ("Remaining ICPs", len(run_items)),
                        ("Total ICPs", total_icps),
                    ),
                )
            )
        scorer_limit = _benchmark_scorer_max_concurrency()
        scorer_semaphore = asyncio.Semaphore(scorer_limit) if scorer_limit > 0 else None
        executor = concurrent.futures.ThreadPoolExecutor(
            # Retry rounds reuse this pool — size for whichever phase is wider.
            max_workers=max(concurrency, self.config.private_baseline_retry_concurrency),
            thread_name_prefix="baseline-icp",
        )
        try:
            semaphore = asyncio.Semaphore(concurrency)

            async def run_one(item_index: int, item: Mapping[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    entry = await self._run_baseline_icp(
                        runner=runner,
                        scorer=scorer,
                        item=item,
                        item_index=item_index,
                        total_icps=total_icps,
                        run_start=run_start,
                        executor=executor,
                        scorer_semaphore=scorer_semaphore,
                        mode_label=mode_label,
                        trace_context=trace_context,
                    )
                    if icp_checkpoint is not None and _baseline_summary_checkpointable(entry):
                        await icp_checkpoint(entry)
                    return entry

            # Settle-then-raise: blocking docker waits cannot be cancelled, so a
            # fail-fast gather would leave containers racing the failure path.
            # Wait for every task to finish (bounded by the per-container
            # timeout), then surface the first fatal error.
            settled = await asyncio.gather(
                *(run_one(item_index, item) for item_index, item in run_items),
                return_exceptions=True,
            )
            fatal = [entry for entry in settled if isinstance(entry, BaseException)]
            if fatal:
                raise fatal[0]
            for entry in settled:
                results[entry["_item_index"]] = entry

            retried_total = 0
            recovered_total = 0
            for round_no in range(1, self.config.private_baseline_provider_retry_rounds + 1):
                pending = sorted(
                    item_index for item_index, entry in results.items() if entry.get("_retryable")
                )
                if not pending:
                    break
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE RETRY ROUND",
                        (
                            ("Worker", self.worker_ref),
                            ("Round", f"{round_no}/{self.config.private_baseline_provider_retry_rounds}"),
                            ("Retrying ICPs", len(pending)),
                            ("ICP indexes", ", ".join(str(item_index) for item_index in pending)),
                            ("Retry concurrency", self.config.private_baseline_retry_concurrency),
                        ),
                    )
                )
                retry_semaphore = asyncio.Semaphore(self.config.private_baseline_retry_concurrency)
                round_retry_runner = _retry_runner_with_provider_cost_scope(
                    retry_runner,
                    retry_round=round_no,
                )

                async def retry_one(item_index: int) -> dict[str, Any]:
                    async with retry_semaphore:
                        backoff_seconds = float(
                            results.get(item_index, {}).get("_retry_backoff_seconds") or 0.0
                        )
                        if backoff_seconds > 0:
                            logger.warning(
                                "research_lab_baseline_icp_rate_limit_backoff worker_ref=%s round=%s icp_index=%s backoff_seconds=%.1f",
                                self.worker_ref,
                                round_no,
                                item_index,
                                backoff_seconds,
                            )
                            await asyncio.sleep(backoff_seconds)
                        entry = await self._run_baseline_icp(
                            runner=round_retry_runner,
                            scorer=scorer,
                            item=window.benchmark_items[item_index - 1],
                            item_index=item_index,
                            total_icps=total_icps,
                            run_start=run_start,
                            executor=executor,
                            scorer_semaphore=scorer_semaphore,
                            mode_label=mode_label,
                            trace_context=trace_context,
                        )
                        if icp_checkpoint is not None and _baseline_summary_checkpointable(entry):
                            await icp_checkpoint(entry)
                        return entry

                retried = await asyncio.gather(
                    *(retry_one(item_index) for item_index in pending),
                    return_exceptions=True,
                )
                fatal = [entry for entry in retried if isinstance(entry, BaseException)]
                if fatal:
                    raise fatal[0]
                recovered = 0
                for entry in retried:
                    if not entry.get("_runtime_error"):
                        recovered += 1
                    # Replace on success AND on repeat failure: the fresher
                    # attempt carries the more current diagnostics.
                    results[entry["_item_index"]] = entry
                retried_total += len(pending)
                recovered_total += recovered
            unresolved = sorted(
                item_index for item_index, entry in results.items() if entry.get("_runtime_error")
            )
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE PARALLEL SUMMARY",
                    (
                        ("Worker", self.worker_ref),
                        ("ICPs", total_icps),
                        ("Concurrency", concurrency),
                        ("Retried", retried_total),
                        ("Recovered", recovered_total),
                        ("Unresolved provider errors", len(unresolved)),
                        ("Unresolved ICP indexes", ", ".join(str(item_index) for item_index in unresolved) or "-"),
                        ("Elapsed", f"{time.time() - run_start:.1f}s"),
                    ),
                )
            )
            return (
                [results[item_index] for item_index in sorted(results)],
                {
                    "retried": retried_total,
                    "recovered": recovered_total,
                    "unresolved": len(unresolved),
                },
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _reusable_same_day_benchmark(
        self,
        *,
        today: str,
        manifest_hash: str,
    ) -> dict[str, Any] | None:
        """Any valid benchmark already recorded today for this model (any window).

        A restart used to silently re-run the baseline and replace the day's
        promotion reference mid-day (§0.3); replacement must be deliberate via
        RESEARCH_LAB_BASELINE_ALLOW_SAMEDAY_REPLACE.
        """
        if _env_flag("RESEARCH_LAB_BASELINE_ALLOW_SAMEDAY_REPLACE"):
            return None
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns="*",
            filters=(
                ("benchmark_date", today),
                ("private_model_manifest_hash", manifest_hash),
            ),
            order_by=(("created_at", True),),
            limit=25,
        )
        for row in rows:
            if _private_benchmark_row_is_valid(row):
                return dict(row)
        return None

    async def _same_day_reference_recorded_while_running(
        self,
        *,
        today: str,
        window_hash: str,
        manifest_hash: str,
    ) -> dict[str, Any] | None:
        """Re-check just before recording: another worker (or a restart) may have
        recorded the day's reference while this multi-hour run was in flight."""
        if _env_flag("RESEARCH_LAB_BASELINE_ALLOW_SAMEDAY_REPLACE"):
            return None
        try:
            rows = await select_many(
                "research_lab_private_model_benchmark_current",
                columns="*",
                filters=(
                    ("benchmark_date", today),
                    ("private_model_manifest_hash", manifest_hash),
                ),
                order_by=(("created_at", True),),
                limit=25,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_pre_record_check_failed error=%s", str(exc)[:200])
            return None
        for row in rows:
            if _private_benchmark_row_is_valid(row):
                return dict(row)
        return None

    async def _baseline_day_jump_points(
        self,
        *,
        today: str,
        aggregate_score: float,
    ) -> float | None:
        """Aggregate jump vs the most recent valid pre-today benchmark.

        Best-effort: a lookup failure never fails the baseline. Always logs the
        jump loudly — day-over-day drift of 3-5 points has been observed live,
        several times the 1.0-point promotion threshold (§0-N1).
        """
        try:
            rows = await select_many(
                "research_lab_private_model_benchmark_current",
                columns="*",
                filters=(("benchmark_date", "lt", today),),
                order_by=(("benchmark_date", True), ("created_at", True)),
                limit=10,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_day_jump_lookup_failed error=%s", str(exc)[:200])
            return None
        previous = next((row for row in rows if _private_benchmark_row_is_valid(row)), None)
        if previous is None:
            return None
        previous_aggregate = _safe_float(previous.get("aggregate_score"), default=0.0)
        day_jump = float(aggregate_score) - previous_aggregate
        logger.warning(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE DAY-OVER-DAY JUMP",
                (
                    ("Worker", self.worker_ref),
                    ("Benchmark date", today),
                    ("Aggregate score", f"{aggregate_score:.4f}"),
                    ("Previous date", previous.get("benchmark_date")),
                    ("Previous aggregate", f"{previous_aggregate:.4f}"),
                    ("Previous model", compact_ref(previous.get("private_model_manifest_hash"))),
                    ("Jump", f"{day_jump:+.4f}"),
                    ("Quarantine threshold", _baseline_max_day_jump_points() or "warn-only"),
                ),
            )
        )
        return day_jump

    async def _baseline_leased_by_other_worker(self, today: str) -> bool:
        """Best-effort lease via dispatch events so any worker can run baselines
        (RESEARCH_LAB_BASELINE_ANY_WORKER) without doubling the day's run."""
        try:
            lease_seconds = max(600, int(os.getenv("RESEARCH_LAB_BASELINE_LEASE_SECONDS", "5400")))
        except ValueError:
            lease_seconds = 5400
        try:
            rows = await select_many(
                "research_lab_scoring_dispatch_events",
                columns="dispatch_event_id,dispatch_status,worker_ref,event_doc,created_at",
                filters=(("dispatch_type", "private_baseline_rebenchmark"),),
                order_by=(("created_at", True),),
                limit=50,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_lease_check_failed error=%s", str(exc)[:200])
            return False
        for row in rows:
            doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
            if str(doc.get("benchmark_date") or "") != today:
                continue
            # Rows are newest-first: the first event for today decides.
            status = str(row.get("dispatch_status") or "")
            if status in ("completed", "failed"):
                return False
            if status == "assigned" and str(row.get("worker_ref") or "") != self.worker_ref:
                age = _status_age_seconds(row.get("created_at"))
                if age is not None and age <= lease_seconds:
                    logger.info(
                        "research_lab_baseline_leased_elsewhere worker_ref=%s owner=%s age_seconds=%.0f",
                        self.worker_ref,
                        row.get("worker_ref"),
                        age,
                    )
                    return True
            return False
        return False

    async def _confirm_baseline_lease(
        self,
        *,
        today: str,
        lease_event: Mapping[str, Any] | None,
    ) -> bool:
        """Ownership confirm after writing our `assigned` lease marker.

        The winner is the earliest unexpired open lease for today: per worker,
        only its LATEST private_baseline_rebenchmark event counts (a later
        completed/failed closes that worker's lease), leases older than
        RESEARCH_LAB_BASELINE_LEASE_SECONDS are expired (dead worker), and ties
        break on dispatch_event_id. Read failures confirm optimistically — the
        same-day-reuse guard and the pre-record conflict check backstop
        duplicate recordings; this confirm only avoids wasting a duplicate
        multi-hour run.
        """
        try:
            lease_seconds = max(600, int(os.getenv("RESEARCH_LAB_BASELINE_LEASE_SECONDS", "5400")))
        except ValueError:
            lease_seconds = 5400
        try:
            rows = await select_many(
                "research_lab_scoring_dispatch_events",
                columns="dispatch_event_id,dispatch_status,worker_ref,event_doc,created_at",
                filters=(("dispatch_type", "private_baseline_rebenchmark"),),
                order_by=(("created_at", True),),
                limit=100,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_lease_confirm_failed error=%s", str(exc)[:200])
            return True
        latest_by_worker: dict[str, dict[str, Any]] = {}
        for row in rows:
            doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
            if str(doc.get("benchmark_date") or "") != today:
                continue
            worker = str(row.get("worker_ref") or "")
            if worker and worker not in latest_by_worker:
                # Rows are newest-first: the first row per worker is its
                # current lease state for today.
                latest_by_worker[worker] = row
        open_leases: list[dict[str, Any]] = []
        for row in latest_by_worker.values():
            if str(row.get("dispatch_status") or "") != "assigned":
                continue
            age = _status_age_seconds(row.get("created_at"))
            if age is None or age > lease_seconds:
                continue
            open_leases.append(row)
        if not open_leases:
            return True
        winner = min(
            open_leases,
            key=lambda row: (str(row.get("created_at") or ""), str(row.get("dispatch_event_id") or "")),
        )
        if str(winner.get("worker_ref") or "") == self.worker_ref:
            return True
        lease_event_id = str((lease_event or {}).get("dispatch_event_id") or "")
        if lease_event_id and str(winner.get("dispatch_event_id") or "") == lease_event_id:
            return True
        logger.info(
            "research_lab_baseline_lease_lost worker_ref=%s winner_worker=%s winner_event=%s",
            self.worker_ref,
            winner.get("worker_ref"),
            compact_ref(winner.get("dispatch_event_id")),
        )
        return False

    def _is_private_baseline_owner(self) -> bool:
        if self.config.scoring_worker_index == 0:
            return True
        # Opt-in: let any worker run the daily baseline (leased via dispatch
        # events) instead of being hostage to worker 0.
        return _env_flag("RESEARCH_LAB_BASELINE_ANY_WORKER")

    async def _resolve_evaluation_epoch(self) -> int:
        now = time.monotonic()
        if self._resolved_epoch_cache is not None:
            cached_epoch, cached_at = self._resolved_epoch_cache
            if now - cached_at <= 60.0:
                return cached_epoch

        epoch, block, source = await resolve_research_lab_evaluation_epoch(self.config.evaluation_epoch)

        if epoch <= 0:
            raise RuntimeError(
                "Research Lab evaluation epoch resolved to 0; refusing to write epoch-0 score/audit bundles"
            )
        self._resolved_epoch_cache = (epoch, now)
        logger.info(
            "Research Lab scoring worker resolved evaluation epoch: epoch=%s block=%s source=%s",
            epoch,
            block,
            source,
        )
        return epoch

    async def _write_audit_bundle(self, epoch: int) -> None:
        """Best-effort audit bundle write: it runs on the scoring hot path after
        every scored candidate/baseline, so a failure here must never fail the
        pass (bug #9/#36 — it used to mislabel every scored candidate)."""
        try:
            await self._write_audit_bundle_inner(epoch)
        except Exception as exc:
            logger.warning(
                "research_lab_audit_bundle_write_failed epoch=%s error=%s",
                epoch,
                _short_error(exc),
            )
            try:
                await create_scoring_dispatch_event(
                    dispatch_type="audit_bundle_build",
                    dispatch_status="failed",
                    worker_ref=self.worker_ref,
                    proxy_ref_hash=self.proxy_ref_hash,
                    event_doc={
                        "evaluation_epoch": int(epoch),
                        "error_diagnostics": _event_error_diagnostics(exc),
                    },
                )
            except Exception:
                logger.exception("research_lab_audit_bundle_failure_event_write_failed")

    async def _write_audit_bundle_inner(self, epoch: int) -> None:
        # The five append-only event tables grow without bound and overflow the
        # select_all cap; scope them to a recent window like score bundles are
        # scoped by epoch (bug #9).
        event_window = self._audit_event_window_filters()
        ticket_rows = await self._audit_select_all("research_loop_ticket_current", current_view=True)
        queue_rows = await self._audit_select_all("research_loop_run_queue_current", current_view=True)
        receipt_rows = await self._audit_select_all("research_loop_receipt_current", current_view=True)
        candidate_rows = await self._audit_select_all("research_lab_candidate_evaluation_current", current_view=True)
        candidate_event_rows = await self._audit_select_all("research_lab_candidate_evaluation_events", filters=event_window)
        loop_event_rows = await self._audit_select_all("research_lab_auto_research_loop_events", filters=event_window)
        dispatch_event_rows = await self._audit_select_all("research_lab_scoring_dispatch_events", filters=event_window)
        rolling_window_rows = await self._audit_select_all("research_lab_rolling_icp_windows")
        benchmark_rows = await self._audit_select_all("research_lab_private_model_benchmark_current", current_view=True)
        private_model_version_rows = await self._audit_select_all("research_lab_private_model_version_current", current_view=True)
        promotion_event_rows = await self._audit_select_all("research_lab_candidate_promotion_events", filters=event_window)
        private_repo_commit_event_rows = await self._audit_select_all("research_lab_private_repo_commit_events", filters=event_window)
        public_benchmark_report_rows = await self._audit_select_all(
            "research_lab_public_benchmark_report_current",
            current_view=True,
        )
        score_bundle_rows = await self._audit_select_all(
            "research_evaluation_score_bundle_current",
            filters=(("evaluation_epoch", epoch),),
            current_view=True,
        )
        bundle_doc = build_research_lab_audit_bundle(
            epoch=epoch,
            ticket_rows=ticket_rows,
            queue_rows=queue_rows,
            receipt_rows=receipt_rows,
            candidate_rows=candidate_rows,
            candidate_event_rows=candidate_event_rows,
            loop_event_rows=loop_event_rows,
            dispatch_event_rows=dispatch_event_rows,
            rolling_window_rows=rolling_window_rows,
            benchmark_rows=benchmark_rows,
            private_model_version_rows=private_model_version_rows,
            promotion_event_rows=promotion_event_rows,
            private_repo_commit_event_rows=private_repo_commit_event_rows,
            public_benchmark_report_rows=public_benchmark_report_rows,
            score_bundle_rows=score_bundle_rows,
        )
        audit_hash = canonical_hash(bundle_doc)
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=audit_hash,
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        bundle, _event = await create_signed_audit_bundle(
            epoch=epoch,
            bundle_doc=bundle_doc,
            signature_ref=signature_ref,
        )
        await create_scoring_dispatch_event(
            dispatch_type="audit_bundle_build",
            dispatch_status="completed",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            event_doc={
                "audit_bundle_id": str(bundle["audit_bundle_id"]),
                "audit_bundle_hash": str(bundle["audit_bundle_hash"]),
            },
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB AUDIT BUNDLE WRITTEN",
                (
                    ("Worker", self.worker_ref),
                    ("Epoch", epoch),
                    ("Audit bundle", compact_ref(bundle["audit_bundle_id"])),
                    ("Audit hash", compact_ref(bundle["audit_bundle_hash"])),
                ),
            )
        )

    def _audit_event_window_filters(self) -> tuple[tuple[Any, ...], ...]:
        try:
            days = float(os.getenv("RESEARCH_LAB_AUDIT_EVENT_WINDOW_DAYS", "3"))
        except ValueError:
            days = 3.0
        if days <= 0:
            # Explicit opt-out restores unscoped full-history fetches.
            return ()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return (("created_at", "gte", cutoff.isoformat()),)

    def _audit_select_limits(self) -> tuple[int, int]:
        max_rows = max(1000, _env_int("RESEARCH_LAB_AUDIT_SELECT_MAX_ROWS", 10000))
        batch_size = max(100, min(max_rows, _env_int("RESEARCH_LAB_AUDIT_SELECT_BATCH_SIZE", 500)))
        return max_rows, batch_size

    async def _audit_select_all(
        self,
        table: str,
        *,
        filters: tuple[tuple[Any, ...], ...] = (),
        current_view: bool = False,
    ) -> list[dict[str, Any]]:
        primary_order = (("current_status_at", True),) if current_view else (("created_at", True),)
        max_rows, batch_size = self._audit_select_limits()
        try:
            return await select_all(
                table,
                filters=filters,
                order_by=primary_order,
                max_rows=max_rows,
                batch_size=batch_size,
                allow_partial=True,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_audit_select_order_fallback table=%s order=%s error=%s",
                table,
                primary_order,
                str(exc)[:200],
            )
            return await select_all(
                table,
                filters=filters,
                max_rows=max_rows,
                batch_size=batch_size,
                allow_partial=True,
            )

    async def _maybe_finalize_candidate_receipt(self, candidate: Mapping[str, Any]) -> bool:
        receipt_id = candidate.get("receipt_id")
        if not receipt_id:
            return False
        candidates = await select_many(
            "research_lab_candidate_evaluation_current",
            filters=(("run_id", str(candidate["run_id"])),),
            limit=1000,
        )
        if not candidates:
            return False
        terminal_statuses = {"scored", "failed", "rejected", "tombstoned"}
        status_counts: dict[str, int] = {}
        score_bundle_ids: list[str] = []
        for row in candidates:
            status = str(row.get("current_candidate_status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status not in terminal_statuses:
                return False
            score_bundle_id = row.get("current_score_bundle_id")
            if score_bundle_id:
                score_bundle_ids.append(str(score_bundle_id))
        receipt = await select_one(
            "research_loop_receipt_current",
            filters=(("receipt_id", str(receipt_id)),),
        )
        if not receipt or receipt.get("current_receipt_status") != "queued":
            return False
        has_scored_candidate = status_counts.get("scored", 0) > 0
        event_doc = {
            "run_id": str(candidate["run_id"]),
            "candidate_status_counts": status_counts,
            "score_bundle_ids": score_bundle_ids,
            "finalization_source": "gateway_qualification_worker_results",
        }
        try:
            await create_receipt_event(
                receipt_id=str(receipt_id),
                ticket_id=str(candidate["ticket_id"]),
                event_type="completed" if has_scored_candidate else "failed",
                receipt_status="completed" if has_scored_candidate else "failed",
                event_doc=event_doc,
            )
        except Exception as exc:
            if not _is_event_sequence_race_error(exc):
                raise
            latest_receipt = await select_one(
                "research_loop_receipt_current",
                filters=(("receipt_id", str(receipt_id)),),
            )
            if latest_receipt and latest_receipt.get("current_receipt_status") != "queued":
                logger.info(
                    "research_lab_receipt_finalization_race_lost receipt_id=%s status=%s",
                    compact_ref(receipt_id),
                    latest_receipt.get("current_receipt_status"),
                )
                return False
            raise
        try:
            await create_ticket_event(
                ticket_id=str(candidate["ticket_id"]),
                event_type="completed" if has_scored_candidate else "cancelled",
                actor_hotkey=None,
                reason=(
                    "gateway_research_lab_candidate_evaluation_completed"
                    if has_scored_candidate
                    else "gateway_research_lab_candidate_evaluation_failed"
                ),
                event_doc=event_doc,
            )
        except Exception as exc:
            if not _is_event_sequence_race_error(exc):
                raise
            logger.warning(
                "research_lab_ticket_finalization_race_lost ticket_id=%s receipt_id=%s error=%s",
                compact_ref(candidate["ticket_id"]),
                compact_ref(receipt_id),
                str(exc)[:240],
            )
        logger.info(
            format_worker_block(
                "RESEARCH LAB RECEIPT FINALIZED",
                (
                    ("Worker", self.worker_ref),
                    ("Receipt", compact_ref(receipt_id)),
                    ("Run", compact_ref(candidate["run_id"])),
                    ("Status", "completed" if has_scored_candidate else "failed"),
                    ("Candidates scored", status_counts.get("scored", 0)),
                    ("Candidates failed", status_counts.get("failed", 0)),
                    ("Score bundles", len(score_bundle_ids)),
                ),
            )
        )
        return True

    def _candidate_run_context(
        self,
        candidate: Mapping[str, Any],
        *,
        window_hash: str,
        evaluation_epoch: int,
    ) -> dict[str, Any]:
        # Bundle→trace forward linkage: when the candidate row carries its loop
        # node id (worker-side annotation in candidate_build_doc), stamp the
        # SAME deterministic execution_trace:<uuid5> ref the trajectory
        # projector writes for that node, so score bundles, engine_trace_
        # mappings, and execution_traces all join on one key. Candidates
        # without node linkage (pre-annotation rows) keep the legacy
        # worker-scoped ref; readers must tolerate both formats — historical
        # bundles are immutable.
        loop_node_id = ""
        build_doc_for_node = candidate.get("candidate_build_doc")
        if isinstance(build_doc_for_node, Mapping):
            loop_node_id = str(build_doc_for_node.get("loop_node_id") or "")
        if loop_node_id:
            execution_trace_ref = (
                f"execution_trace:{execution_trace_id_for_node(str(candidate['run_id']), loop_node_id)}"
            )
        else:
            execution_trace_ref = (
                f"gateway_qualification_worker:{self.worker_ref}:{candidate['candidate_id']}"
            )
        context = {
            "run_id": str(candidate["run_id"]),
            "ticket_id": str(candidate["ticket_id"]),
            "miner_hotkey": str(candidate["miner_hotkey"]),
            "island": str(candidate.get("island") or "generalist"),
            "evaluation_epoch": int(evaluation_epoch),
            "evaluator_version": "leadpoet-gateway-qualification-worker:research-lab:v1",
            "evidence_bundle_refs": [f"research_lab_rolling_icp_window:{window_hash}"],
            "execution_trace_ref": execution_trace_ref,
            "cost_ledger_ref": "cost_ledger:" + canonical_hash(
                {
                    "candidate_id": candidate["candidate_id"],
                    "worker_ref": self.worker_ref,
                    "rolling_window_hash": window_hash,
                }
            ).split(":", 1)[1],
        }
        if str(candidate.get("candidate_kind") or "") == "image_build":
            if candidate.get("candidate_source_diff_hash"):
                context["candidate_source_diff_hash"] = str(candidate["candidate_source_diff_hash"])
            build_doc = candidate.get("candidate_build_doc")
            if isinstance(build_doc, Mapping):
                context["candidate_build_ref"] = str(
                    build_doc.get("build_doc_hash")
                    or canonical_hash(build_doc)
                )
        return context

    def _evaluation_policy(self) -> dict[str, Any]:
        return {
            "min_delta": float(
                os.environ.get(
                    "RESEARCH_LAB_MIN_DELTA",
                    str(self.config.improvement_threshold_points),
                )
            ),
            "min_successful_icps": int(
                os.environ.get(
                    "RESEARCH_LAB_MIN_SUCCESSFUL_ICPS",
                    str(self.config.lab_champion_eval_days * self.config.lab_champion_icps_per_day),
                )
            ),
            "max_hard_failures": int(os.environ.get("RESEARCH_LAB_MAX_HARD_FAILURES", "0")),
            "min_candidate_score": float(os.environ.get("RESEARCH_LAB_MIN_CANDIDATE_SCORE", "0")),
            "observed_cost_usd": 0.0,
        }

    def _private_scoring_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for name in (
            "EXA_API_KEY",
            "EXA_MAX_RPS",
            "SCRAPINGDOG_API_KEY",
            "QUALIFICATION_SCRAPINGDOG_API_KEY",
            "OPENROUTER_API_KEY",
            "QUALIFICATION_OPENROUTER_API_KEY",
            "OPENROUTER_KEY",
            # Domain-keyed firmographics fallback for failed LinkedIn
            # resolution/scrapes inside the model. Absent vars keep the
            # fallback disabled and the model byte-identical to before.
            "DEEPLINE_API_KEY",
            "SOURCING_DEEPLINE_FALLBACK",
            "SOURCING_DEEPLINE_TIMEOUT_S",
        ):
            value = os.getenv(name)
            if value:
                env[name] = value
        # Provider evidence replay: when the worker carries a recorded per-ICP
        # cache directory, candidate runs replay the recorded baseline
        # evidence for identical provider requests.
        evidence_cache_dir = os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR")
        if evidence_cache_dir:
            env["RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR"] = evidence_cache_dir
        # Every run records its provider I/O at full fidelity: cached hits
        # replay recorded evidence, and any live call (cache miss or fresh
        # surface) is captured completely for audit and future replay.
        env["RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD"] = "1"
        evidence_proxy_url = os.getenv("RESEARCH_LAB_EVIDENCE_PROXY_URL")
        if evidence_proxy_url:
            env["RESEARCH_LAB_EVIDENCE_PROXY_URL"] = evidence_proxy_url
        env["RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP"] = str(
            self.config.provider_cost_cap_usd_per_icp
        )
        env["RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD"] = str(
            self.config.scrapingdog_cost_per_credit_usd
        )
        env["RESEARCH_LAB_SCRAPINGDOG_UNKNOWN_ENDPOINT_CREDITS"] = str(
            self.config.scrapingdog_unknown_endpoint_credits
        )
        env["RESEARCH_LAB_PROVIDER_COST_UNKNOWN_ENDPOINT_POLICY"] = str(
            self.config.provider_cost_unknown_endpoint_policy
        )
        scoring_cache_dir = os.getenv("RESEARCH_LAB_SCORING_CACHE_DIR")
        if scoring_cache_dir:
            env["RESEARCH_LAB_SCORING_CACHE_DIR"] = scoring_cache_dir
        if self.proxy_url and self.config.private_model_docker_global_proxy_enabled:
            env.update(
                {
                    "HTTP_PROXY": self.proxy_url,
                    "HTTPS_PROXY": self.proxy_url,
                    "http_proxy": self.proxy_url,
                    "https_proxy": self.proxy_url,
                }
            )
        no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
        if no_proxy:
            env["NO_PROXY"] = no_proxy
            env["no_proxy"] = no_proxy
        # P13 corpus-mode capture budgets: with an S3 persistence prefix set,
        # containers get the larger byte caps so tool-call payloads are not
        # silently truncated (explicit operator caps always win).
        from research_lab.eval.private_runtime import incontainer_trace_corpus_env

        env.update(incontainer_trace_corpus_env())
        return env

    def _with_provider_cost_evaluation_scope(
        self,
        env: Mapping[str, str],
        *,
        run_type: str,
        rolling_window_hash: str = "",
        artifact_hash: str = "",
        candidate_id: str = "",
        run_id: str = "",
        ticket_id: str = "",
        dispatch_event_id: str = "",
        benchmark_date: str = "",
        benchmark_attempt: int | None = None,
        confirmation_attempt: int | None = None,
        side: str = "",
        evaluation_epoch: int | None = None,
        started_at: float | None = None,
    ) -> dict[str, str]:
        scoped = dict(env)
        scope_doc: dict[str, Any] = {
            "schema_version": "research_lab_provider_cost_evaluation_scope.v1",
            "run_type": str(run_type or "unknown"),
            "worker_ref": self.worker_ref,
        }
        optional_values: dict[str, Any] = {
            "rolling_window_hash": rolling_window_hash,
            "artifact_hash": artifact_hash,
            "candidate_id": candidate_id,
            "run_id": run_id,
            "ticket_id": ticket_id,
            "dispatch_event_id": dispatch_event_id,
            "benchmark_date": benchmark_date,
            "side": side,
        }
        for key, value in optional_values.items():
            text = str(value or "").strip()
            if text:
                scope_doc[key] = text
        if benchmark_attempt is not None:
            scope_doc["benchmark_attempt"] = int(benchmark_attempt)
        if confirmation_attempt is not None:
            scope_doc["confirmation_attempt"] = int(confirmation_attempt)
        if evaluation_epoch is not None:
            scope_doc["evaluation_epoch"] = int(evaluation_epoch)
        if started_at is not None:
            scope_doc["started_at_ms"] = int(float(started_at) * 1000)
        scoped[PROVIDER_COST_EVALUATION_SCOPE_ENV] = sha256_json(scope_doc)
        return scoped

    def _private_baseline_scoring_env(self) -> dict[str, str]:
        """Candidate scoring env with the benchmark's dedicated Exa budget.

        The daily baseline can run many model containers at once; EXA_MAX_RPS is
        enforced per container, so the aggregate burst must be isolated from the
        prod Exa key and split across the concurrent containers. Both overrides
        are opt-in — unset config falls back to the prod values.
        """
        env = self._private_scoring_env()
        # The baseline run seeds the replay cache and must observe live
        # providers only, so it never receives a cache to consume (recording
        # is inherited from the base scoring env).
        env.pop("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR", None)
        if self.config.benchmark_exa_api_key:
            env["EXA_API_KEY"] = self.config.benchmark_exa_api_key
        if self.config.benchmark_exa_max_rps > 0:
            env["EXA_MAX_RPS"] = str(self.config.benchmark_exa_max_rps)
        return env

    def _private_baseline_retry_scoring_env(self) -> dict[str, str]:
        # Retry rounds run fewer containers; re-spread the SAME aggregate Exa
        # budget across them so retried ICPs finish faster instead of idling at
        # the first-pass per-container rate.
        env = self._private_baseline_scoring_env()
        if self.config.benchmark_exa_max_rps > 0:
            aggregate = self.config.benchmark_exa_max_rps * self.config.private_baseline_concurrency
            env["EXA_MAX_RPS"] = str(
                round(aggregate / max(1, self.config.private_baseline_retry_concurrency), 3)
            )
        return env

    def _missing_private_scoring_env(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not os.getenv("EXA_API_KEY"):
            missing.append("EXA_API_KEY")
        if not (os.getenv("SCRAPINGDOG_API_KEY") or os.getenv("QUALIFICATION_SCRAPINGDOG_API_KEY")):
            missing.append("SCRAPINGDOG_API_KEY or QUALIFICATION_SCRAPINGDOG_API_KEY")
        if not (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("QUALIFICATION_OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_KEY")
        ):
            missing.append("OPENROUTER_API_KEY or QUALIFICATION_OPENROUTER_API_KEY or OPENROUTER_KEY")
        return tuple(missing)

    def _private_model_env_passthrough(self) -> tuple[str, ...]:
        return private_model_env_passthrough(
            include_proxy=self.config.private_model_docker_global_proxy_enabled
        )


def _average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _next_benchmark_attempt(rows: list[Mapping[str, Any]]) -> int:
    attempts: list[int] = []
    for row in rows:
        try:
            attempts.append(int(row.get("benchmark_attempt") or 0))
        except (TypeError, ValueError):
            attempts.append(0)
    return (max(attempts) + 1) if attempts else 0


def _private_benchmark_row_is_valid(row: Mapping[str, Any]) -> bool:
    status = str(row.get("current_benchmark_status") or row.get("benchmark_status") or "")
    if status and status != "completed":
        return False
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    summaries = doc.get("per_icp_summaries") if isinstance(doc, Mapping) else None
    if not isinstance(summaries, list) or not summaries:
        return False
    if not any(_benchmark_summary_has_companies(item) for item in summaries):
        return False
    if str(row.get("benchmark_quality") or "") == "passed":
        return True
    try:
        return int(row.get("evaluation_epoch") or 0) > 0
    except (TypeError, ValueError):
        return False


def _rolling_window_fetch_kwargs(config: Any) -> dict[str, Any]:
    total = max(
        1,
        _safe_int(getattr(config, "lab_champion_eval_days", 10), default=10)
        * _safe_int(getattr(config, "lab_champion_icps_per_day", 2), default=2),
    )
    if total == 20:
        default_fresh = 10
    else:
        default_fresh = max(1, total // 2)
    default_retained = max(1, total - default_fresh)
    fresh = max(
        1,
        _safe_int(getattr(config, "lab_champion_fresh_icp_count", default_fresh), default=default_fresh),
    )
    retained = max(
        1,
        _safe_int(
            getattr(config, "lab_champion_retained_icp_count", default_retained),
            default=default_retained,
        ),
    )
    return {
        "window_mode": str(
            getattr(config, "lab_champion_window_mode", "hybrid_fresh_retained")
            or "hybrid_fresh_retained"
        ).strip().lower(),
        "fresh_icp_count": fresh,
        "retained_icp_count": retained,
        "min_new_icp_count": fresh,
    }


def _benchmark_summary_has_companies(item: Any) -> bool:
    if not isinstance(item, Mapping):
        return False
    try:
        return int(item.get("company_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def _private_holdout_gate_from_baseline_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    split = doc.get("visibility_split") if isinstance(doc.get("visibility_split"), Mapping) else {}
    items = split.get("items") if isinstance(split.get("items"), list) else []
    public_items = [
        item for item in items
        if isinstance(item, Mapping) and str(item.get("visibility") or "") == "public"
    ]
    private_count = _safe_int(split.get("private_count"), default=0)
    if private_count <= 0:
        private_count = sum(
            1
            for item in items
            if isinstance(item, Mapping) and str(item.get("visibility") or "") == "private"
        )
    public_refs = [
        str(item.get("icp_ref") or "")
        for item in public_items
        if str(item.get("icp_ref") or "").strip()
    ]
    if not public_refs or private_count <= 0:
        return None
    public_scores = [_safe_float(item.get("score"), default=0.0) for item in public_items]
    private_scores = [
        _safe_float(item.get("score"), default=0.0)
        for item in items
        if isinstance(item, Mapping) and str(item.get("visibility") or "") == "private"
    ]
    all_scores = [
        _safe_float(item.get("score"), default=0.0)
        for item in items
        if isinstance(item, Mapping)
    ]
    baseline_aggregate_score = _safe_float(
        doc.get("aggregate_score"),
        default=_average(all_scores),
    )
    return {
        "schema_version": "1.0",
        "gate_type": "public_score_before_private_holdout",
        "baseline_benchmark_bundle_id": str(row.get("benchmark_bundle_id") or ""),
        "baseline_benchmark_hash": canonical_hash(doc) if doc else "",
        "baseline_aggregate_score": baseline_aggregate_score,
        "baseline_public_score": _average(public_scores),
        "baseline_private_score": _average(private_scores),
        "baseline_public_icp_count": len(public_refs),
        "baseline_private_holdout_icp_count": private_count,
        "rolling_window_hash": str(row.get("rolling_window_hash") or ""),
        "private_model_manifest_hash": str(row.get("private_model_manifest_hash") or ""),
        "public_icp_refs": public_refs,
        # Per-ICP baseline scores let verifiers recompute the true advisory
        # delta (candidate-per-ICP minus stored-baseline-per-ICP) instead of
        # falling back to a not_applicable verdict (§0-N2).
        "baseline_per_icp_scores": {
            str(item.get("icp_ref") or ""): _safe_float(item.get("score"), default=0.0)
            for item in items
            if isinstance(item, Mapping) and str(item.get("icp_ref") or "").strip()
        },
    }


def _candidate_gate_event_doc(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "gate_type": str(value.get("gate_type") or ""),
        "decision": str(value.get("decision") or ""),
        "baseline_benchmark_bundle_id": str(value.get("baseline_benchmark_bundle_id") or ""),
        "baseline_benchmark_hash": str(value.get("baseline_benchmark_hash") or ""),
        "baseline_aggregate_score": _safe_float(value.get("baseline_aggregate_score"), default=0.0),
        "baseline_public_score": _safe_float(value.get("baseline_public_score"), default=0.0),
        "baseline_private_score": _safe_float(value.get("baseline_private_score"), default=0.0),
        "candidate_public_score": _safe_float(value.get("candidate_public_score"), default=0.0),
        "paired_base_public_score": _safe_float(value.get("paired_base_public_score"), default=0.0),
        "candidate_total_score": _safe_float(value.get("candidate_total_score"), default=0.0),
        "paired_base_total_score": _safe_float(value.get("paired_base_total_score"), default=0.0),
        "candidate_delta_vs_daily_baseline": _safe_float(
            value.get("candidate_delta_vs_daily_baseline"),
            default=0.0,
        ),
        "reference_evaluation_mode": str(value.get("reference_evaluation_mode") or ""),
        "public_icp_count": _safe_int(value.get("public_icp_count"), default=0),
        "private_holdout_icp_count": _safe_int(value.get("private_holdout_icp_count"), default=0),
        "private_holdout_evaluated": bool(value.get("private_holdout_evaluated")),
    }


def _compact_scoring_health_doc(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = {
        "schema_version",
        "health_status",
        "icp_count",
        "reference_runtime_success_rate",
        "candidate_runtime_success_rate",
        "reference_zero_company_rate",
        "candidate_zero_company_rate",
        "provider_error_rate",
        "timeout_rate",
        "invalid_output_rate",
        "skipped_candidate_rate",
        "public_holdout_decision",
        "baseline_bundle_id",
        "baseline_bundle_hash",
    }
    return {key: value[key] for key in allowed if key in value}


def _failure_rate_from_success(value: Any) -> float:
    return max(0.0, min(1.0, 1.0 - _safe_float(value, default=1.0)))


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_candidate_claim_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_candidate_claim_conflict" in message
        or "research_lab_candidate_eval_events_candidate_seq_key" in message
        or "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )


def _is_event_sequence_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )
