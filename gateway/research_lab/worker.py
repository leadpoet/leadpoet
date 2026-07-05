"""Hosted Research Lab worker for paid private-model auto-research runs."""

from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import logging
import os
import re
import socket
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.code_build import CodeEditCandidateBuilder, CodeEditInfraFailureError
from gateway.research_lab.code_loop_engine import CodeEditLoopEngine
from gateway.research_lab.config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from gateway.research_lab.key_vault import (
    OpenRouterKeyVaultError,
    decrypt_openrouter_key,
    preflight_openrouter_key,
    strict_openrouter_provider_policy,
    verify_openrouter_workspace_privacy,
)
from gateway.research_lab.logging_utils import compact_ref, format_worker_block, format_worker_line
from gateway.research_lab.loop_engine import (
    AutoResearchLoopEvent,
    AutoResearchLoopSettings,
    OpenRouterCallResult,
)
from gateway.research_lab.maintenance import autoresearch_queue_capacity_doc, is_autoresearch_maintenance_paused
from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest, ResearchLabReceiptCreateRequest
from gateway.research_lab.promotion import (
    PromotionPausedError,
    latest_public_benchmark_summary,
    load_active_private_model,
    reconcile_active_private_model_lineage,
    reconcile_pending_champion_rewards,
)
from gateway.research_lab.public_activity import (
    reproject_stale_public_cards,
    safe_project_public_loop_activity,
)
from gateway.research_lab.reimbursement_awards import (
    cost_evidence_actual_microusd,
    cost_evidence_cost_ledger,
    cost_evidence_from_loop_result,
    cost_evidence_provider_usage,
    create_reimbursement_decision,
    latest_reimbursable_loop_cost_evidence,
    normalize_cost_evidence,
)
from gateway.research_lab.store import (
    canonical_hash,
    create_auto_research_loop_event,
    create_candidate_artifact,
    create_participation_snapshot,
    create_queue_event,
    create_receipt,
    create_receipt_event,
    create_reimbursement_award,
    create_reimbursement_schedule,
    create_ticket_event,
    create_openrouter_privacy_proof_event_sync,
    find_queued_receipt_for_run,
    latest_auto_research_checkpoint,
    select_all,
    select_many,
    select_one,
)
from research_lab.reimbursements import (
    ReimbursementCapUsage,
    build_reimbursement_schedule,
    compute_participation_score,
    compute_reimbursement_award,
)
from research_lab.auto_research_prompt import coerce_component_registry
from research_lab.axis_provenance import (
    current_call_episode,
    episode_id,
    provenance_for_stage,
)
from research_lab.canonical import canonical_json, sha256_bytes, sha256_json
from research_lab.eval import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    private_model_env_passthrough,
)


logger = logging.getLogger(__name__)
_POSTGREST_TIMESTAMP_RE = re.compile(
    r"^(?P<prefix>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
    r"\.(?P<fraction>\d{1,9})(?P<suffix>Z|[+-]\d{2}(?::?\d{2})?)?$"
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


def _openrouter_generation_attempts() -> int:
    try:
        return max(1, min(5, int(os.getenv("RESEARCH_LAB_OPENROUTER_GENERATION_ATTEMPTS", "3"))))
    except ValueError:
        return 3


def _raw_trace_capture_enabled() -> bool:
    """§9.1 item 5 raw prompt/response capture flag.

    Default true (hosted runs capture by default per the plan); the write path
    is additionally inert when no S3 destination or boto3 is available, so the
    default is safe for local/dev processes too."""
    return str(
        os.getenv(_RAW_TRACE_CAPTURE_ENABLED_ENV, "true")
    ).strip().lower() in {"1", "true", "yes", "on"}


def _trace_kms_key_id() -> str:
    """KMS key used for S3 SSE-KMS trace encryption."""
    return str(os.getenv(_TRACE_KMS_KEY_ENV, "") or "").strip()


def _llm_include_reasoning_enabled() -> bool:
    """Optionally request provider reasoning output on calls without an
    explicit reasoning effort.

    Default false: several configured OpenRouter models route correctly with
    explicit reasoning fields only when requested by the model config, but
    return empty/filtered content when bare include_reasoning is forced.
    """
    return str(
        os.getenv("RESEARCH_LAB_LLM_INCLUDE_REASONING", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}


def _heartbeat_conflict_claim_lost_enabled() -> bool:
    """When true, a queue-heartbeat insert rejected by the DB claim guard aborts
    the in-flight run as claim-lost. Enable ONLY after scripts/59 is applied:
    under the scripts/42 guard every heartbeat insert conflicts, so enabling
    this first would abort every run at its first heartbeat."""
    return str(
        os.getenv("RESEARCH_LAB_HEARTBEAT_CONFLICT_CLAIM_LOST_ENABLED", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}


def _generation_stats_mode() -> str:
    """OpenRouter generation-stats reconciliation mode for worker chat calls.

    "full" (default, current behavior): up to 3 synchronous GETs with retry
    sleeps after every chat call. "best_effort_once": a single attempt with no
    retry sleeps. "off": skip the fetch entirely (cost stays at the
    chat-completion usage figure with status generation_stats_disabled).
    """
    mode = str(os.getenv("RESEARCH_LAB_GENERATION_STATS_MODE", "full")).strip().lower()
    if mode not in {"full", "best_effort_once", "off"}:
        return "full"
    return mode


def _worker_proxy_apply_to_llm_enabled() -> bool:
    """When true, the configured hosted-worker proxy is applied to the worker's
    own OpenRouter traffic (chat completions + generation stats), not just
    exported to docker runs. Defaults to current behavior (not applied)."""
    return str(
        os.getenv("RESEARCH_LAB_WORKER_PROXY_APPLY_TO_LLM", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}


def _is_openrouter_reasoning_effort_unsupported(status_code: int, message: str) -> bool:
    text = str(message or "").lower()
    if int(status_code) not in {400, 404, 422}:
        return False
    if (
        "reasoning_effort" not in text
        and "reasoning effort" not in text
        and "include_reasoning" not in text
    ):
        return False
    return any(
        marker in text
        for marker in (
            "unsupported",
            "not supported",
            "unrecognized",
            "unknown",
            "invalid",
            "not allowed",
            "extra inputs are not permitted",
            "unexpected",
        )
    )


def _status_age_seconds(raw_status_at: object) -> float | None:
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
    return (datetime.now(timezone.utc) - status_at.astimezone(timezone.utc)).total_seconds()


def _status_is_stale(raw_status_at: object, stale_after_seconds: int) -> bool:
    age_seconds = _status_age_seconds(raw_status_at)
    return age_seconds is not None and age_seconds > max(
        60,
        int(stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class HostedResearchLabWorkerError(RuntimeError):
    """Raised when a hosted Research Lab run cannot complete safely."""


class RetryableHostedResearchLabWorkerError(HostedResearchLabWorkerError):
    """Raised when a paid hosted run should be requeued instead of terminally failed."""


class OpenRouterLengthRetryableError(RetryableHostedResearchLabWorkerError):
    """Raised when OpenRouter stopped generation at the output token ceiling."""

    def __init__(
        self,
        message: str,
        *,
        provider_usage: Mapping[str, Any] | None = None,
        cost_microusd: int = 0,
    ) -> None:
        super().__init__(message)
        self.provider_usage = dict(provider_usage or {})
        self.cost_microusd = max(0, int(cost_microusd or 0))


class OpenRouterReasoningEffortUnsupportedError(HostedResearchLabWorkerError):
    """Raised internally when a model rejects the optional reasoning_effort field."""


class HostedResearchLabBuilderNotReady(RetryableHostedResearchLabWorkerError):
    """Raised when image-build candidate infrastructure is not ready yet."""


class HostedResearchLabClaimLost(HostedResearchLabWorkerError):
    """Raised when another worker safely claimed the queued run first."""


class CreditBlockedHostedRunError(HostedResearchLabWorkerError):
    """Raised when a hosted run cannot proceed because the miner's OpenRouter key has
    insufficient credits (HTTP 402 / insufficient balance). The run is paused as
    blocked_for_credit (resumable after top-up), never terminally failed, and is NOT
    retried in a tight loop — only revived by the top-up resume path."""


_RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_GENERATION_STATS_URL = "https://openrouter.ai/api/v1/generation"
_RAW_TRACE_CAPTURE_ENABLED_ENV = "RESEARCH_LAB_RAW_TRACE_CAPTURE_ENABLED"
_RAW_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_RAW_TRACE_S3_PREFIX"
_TRACE_KMS_KEY_ENV = "RESEARCH_LAB_TRACE_KMS_KEY_ID"
_RAW_TRACE_PUT_CONNECT_TIMEOUT_SECONDS = 5
_RAW_TRACE_PUT_READ_TIMEOUT_SECONDS = 15
_OPENROUTER_GENERATION_STATS_TIMEOUT_SECONDS = 5
_OPENROUTER_GENERATION_STATS_ATTEMPTS = 3
_OPENROUTER_GENERATION_STATS_RETRY_DELAYS_SECONDS = (0.5, 1.0)
_OPENROUTER_GENERATION_STATS_RETRYABLE_HTTP_CODES = {404, 408, 409, 425, 429, 500, 502, 503, 504}
_OPENROUTER_GENERATION_STATS_FIELDS = {
    "id",
    "api_type",
    "cache_discount",
    "cancelled",
    "created_at",
    "data_region",
    "finish_reason",
    "generation_time",
    "is_byok",
    "latency",
    "model",
    "native_finish_reason",
    "native_tokens_cached",
    "native_tokens_completion",
    "native_tokens_completion_images",
    "native_tokens_prompt",
    "native_tokens_reasoning",
    "num_input_audio_prompt",
    "num_media_completion",
    "num_media_prompt",
    "num_search_results",
    "provider_name",
    "router",
    "service_tier",
    "streamed",
    "tokens_completion",
    "tokens_prompt",
    "total_cost",
    "upstream_inference_cost",
    "usage",
}
_RETRYABLE_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "endpoint connection",
    "read timed out",
    "temporarily unavailable",
    "service unavailable",
    "throttl",
    "rate exceeded",
    "too many requests",
    "rate limit",
    "docker daemon",
    "cannot connect to the docker daemon",
    "manifest unknown",
    "no space left on device",
    "exit status 137",
    "killed",
    "http 408",
    "http 409",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "status 408",
    "status 409",
    "status 425",
    "status 429",
    "status 500",
    "status 502",
    "status 503",
    "status 504",
)
_PERMANENT_ERROR_MARKERS = (
    "duplicate key",
    "violates unique constraint",
    "check constraint",
    "foreign key",
    "invalid input syntax",
    "permission denied",
    "access denied",
    "accessdenied",
    "not authorized",
    "research_lab_queue_capacity_conflict",
    "research_lab_queue_hotkey_conflict",
    "research_lab_run_claim_conflict",
)
_OPENROUTER_PERMANENT_ERROR_MARKERS = (
    "invalid api key",
    "unauthorized",
    "forbidden",
    "authentication",
    "not authenticated",
    "invalid model",
    "model not found",
    "unknown model",
    "no endpoints found",
    "insufficient credits",
    "insufficient balance",
    "payment required",
)
_OPENROUTER_CREDIT_BLOCK_MARKERS = (
    "insufficient credits",
    "insufficient balance",
    "payment required",
    "credit limit",
    "limit remaining",
)
_OPENROUTER_TRANSIENT_ERROR_MARKERS = (
    "no candidate-generation choices",
    "empty candidate-generation content",
    "empty choices",
    "provider",
    "upstream",
    "overloaded",
    "capacity",
    "temporarily",
    "try again",
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "http 408",
    "http 409",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
)

# Paused-queue reasons that are deliberately parked and must only be revived by
# their explicit resume paths (never by the stale-paused reaper). Mirrors the
# maintenance-resume exclusion in maintenance.requeue_paused_autoresearch_runs.
_STALE_PAUSED_REAPER_EXCLUDED_REASONS = frozenset({"blocked_for_credit"})

# Number of most-recent queue events loaded into the hosted run context. Fields
# written only at loop start (e.g. loop_start_credit_id) can fall outside this
# window on much-requeued runs; see _resolve_loop_start_credit_id.
_RUN_CONTEXT_QUEUE_EVENT_LIMIT = 20


def _is_retryable_worker_exception(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RetryableHostedResearchLabWorkerError):
            return True
        if isinstance(current, CodeEditInfraFailureError):
            return True
        if isinstance(current, PromotionPausedError):
            # Fail-closed lineage/promotion pauses (bug #2): retry, never
            # terminally fail a paid run on a transient lineage read.
            return True
        if isinstance(current, HTTPError):
            code = int(getattr(current, "code", 0) or 0)
            return code in _RETRYABLE_HTTP_CODES
        if isinstance(current, (URLError, TimeoutError, asyncio.TimeoutError, socket.timeout, ConnectionError)):
            return True
        current = current.__cause__ or current.__context__

    message = str(exc).lower()
    if any(marker in message for marker in _PERMANENT_ERROR_MARKERS):
        return False
    return any(marker in message for marker in _RETRYABLE_ERROR_MARKERS)


# OpenRouter HTTP 402 / insufficient-credit signatures. A run hitting these is paused
# as blocked_for_credit (resumable after top-up), not terminally failed.
_OPENROUTER_CREDIT_MARKERS = (
    "insufficient credits",
    "insufficient balance",
    "payment required",
    "more credits",
    "requires more credits",
)


def _is_openrouter_credit_block(decoded: object, message: str) -> bool:
    lowered = str(message or "").lower()
    if isinstance(decoded, Mapping):
        raw_error = decoded.get("error")
        if isinstance(raw_error, Mapping):
            code = str(raw_error.get("code") or raw_error.get("status") or "").lower()
            if code == "402":
                return True
            lowered += " " + str(raw_error.get("message") or raw_error.get("type") or "").lower()
        elif raw_error:
            lowered += " " + str(raw_error).lower()
    return any(marker in lowered for marker in _OPENROUTER_CREDIT_MARKERS)


def _raise_openrouter_generation_response_error(
    decoded: object,
    *,
    failure: str,
    default_retryable: bool,
    provider_usage: Mapping[str, Any] | None = None,
    cost_microusd: int = 0,
) -> None:
    summary = _openrouter_response_summary(decoded)
    error = f"OpenRouter candidate generation failed: {failure}: {summary}"
    if _is_openrouter_credit_block(decoded, error):
        raise CreditBlockedHostedRunError(error)
    if _openrouter_generation_stopped_for_length(decoded):
        raise OpenRouterLengthRetryableError(
            error,
            provider_usage=provider_usage,
            cost_microusd=cost_microusd,
        )
    if _openrouter_generation_response_is_retryable(
        decoded,
        error,
        default_retryable=default_retryable,
    ):
        raise RetryableHostedResearchLabWorkerError(error)
    raise HostedResearchLabWorkerError(error)


def _openrouter_generation_stopped_for_length(decoded: object) -> bool:
    if not isinstance(decoded, Mapping):
        return False
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return False
    finish_reason = str(choices[0].get("finish_reason") or "").strip().lower()
    native_finish_reason = str(choices[0].get("native_finish_reason") or "").strip().lower()
    return finish_reason in {"length", "max_tokens"} or native_finish_reason in {"length", "max_tokens"}


def _openrouter_generation_retry_max_tokens(base_max_tokens: int, length_failures: int) -> int:
    base = max(1, int(base_max_tokens or 0))
    if length_failures <= 0:
        return base
    if length_failures == 1:
        return min(24_000, max(base + 4_000, int(base * 1.5)))
    return min(24_000, max(base + 8_000, base * 2))


def _openrouter_generation_response_is_retryable(
    decoded: object,
    message: str,
    *,
    default_retryable: bool,
) -> bool:
    lowered = str(message or "").lower()
    if isinstance(decoded, Mapping):
        raw_error = decoded.get("error")
        if isinstance(raw_error, Mapping):
            code = str(raw_error.get("code") or raw_error.get("status") or "").lower()
            if code in {"400", "401", "403", "404"}:
                return False
            if code in {"408", "409", "425", "429", "500", "502", "503", "504"}:
                return True
            lowered += " " + str(raw_error.get("message") or raw_error.get("type") or "").lower()
        elif raw_error:
            lowered += " " + str(raw_error).lower()
    if any(marker in lowered for marker in _OPENROUTER_PERMANENT_ERROR_MARKERS):
        return False
    if any(marker in lowered for marker in _OPENROUTER_TRANSIENT_ERROR_MARKERS):
        return True
    return bool(default_retryable)


def _openrouter_response_summary(decoded: object) -> str:
    if not isinstance(decoded, Mapping):
        return f"non_object_response:{type(decoded).__name__}"
    summary: dict[str, object] = {
        "keys": sorted(str(key)[:80] for key in decoded.keys())[:20],
    }
    if decoded.get("id"):
        summary["id"] = str(decoded.get("id"))[:120]
    if decoded.get("model"):
        summary["model"] = str(decoded.get("model"))[:120]
    choices = decoded.get("choices")
    if isinstance(choices, list):
        summary["choices_len"] = len(choices)
        if choices and isinstance(choices[0], Mapping):
            summary["first_choice_keys"] = sorted(str(key)[:80] for key in choices[0].keys())[:20]
            summary["finish_reason"] = str(choices[0].get("finish_reason") or "")[:120]
    else:
        summary["choices_type"] = type(choices).__name__
    usage = decoded.get("usage")
    if isinstance(usage, Mapping):
        summary["usage_keys"] = sorted(str(key)[:80] for key in usage.keys())[:20]
    raw_error = decoded.get("error")
    if isinstance(raw_error, Mapping):
        summary["error"] = {
            "code": str(raw_error.get("code") or raw_error.get("status") or "")[:80],
            "type": str(raw_error.get("type") or "")[:120],
            "message": _redact_openrouter_diagnostic(str(raw_error.get("message") or ""), limit=300),
        }
    elif raw_error:
        summary["error"] = _redact_openrouter_diagnostic(str(raw_error), limit=300)
    return _redact_openrouter_diagnostic(
        json.dumps(summary, sort_keys=True, separators=(",", ":")),
        limit=700,
    )


def _redact_openrouter_diagnostic(value: str, *, limit: int) -> str:
    text = str(value or "")
    text = re.sub(r"sk-or-[A-Za-z0-9._:-]+", "[redacted-openrouter-key]", text)
    text = re.sub(r"sb_secret_[A-Za-z0-9._:-]+", "[redacted-supabase-service-role-key]", text)
    text = re.sub(r"sb_publishable_[A-Za-z0-9._:-]+", "[redacted-supabase-anon-key]", text)
    text = re.sub(r"AKIA[A-Z0-9]{16}", "[redacted-aws-access-key-id]", text)
    text = re.sub(r"https?://[^@\s]+@([^\s/]+)", r"[redacted-proxy-url]@\1", text)
    lowered = text.lower()
    secret_markers = (
        "openrouter_api_key",
        "raw_openrouter_key",
        "aws_secret_access_key",
        "service_role_key",
        "proxy password",
        "judge_prompt",
        "hidden_icp",
        "icp_plaintext",
        "private_repo",
        "proxy_url",
    )
    if any(marker in lowered for marker in secret_markers):
        return "[redacted secret-like diagnostic text]"
    return text[: max(1, int(limit))]


def _redacted_ref(value: object) -> str:
    text = str(value or "")
    if len(text) <= 16:
        return text
    return f"{text[:10]}...{text[-6:]}"


def _redact_raw_trace_value(value: Any) -> Any:
    """Recursively strip credential-shaped strings from a raw-trace payload.

    Mirrors the regex half of ``_redact_openrouter_diagnostic`` /
    code_loop_engine's ``_diagnostic_text`` (OpenRouter keys, Supabase keys,
    AWS access key ids, proxy-URL userinfo, ``api_key=`` query params).
    Deliberately does NOT apply the whole-text marker nuke those helpers use:
    prompts/completions legitimately contain source excerpts mentioning words
    like "proxy" or "password", and the capture exists precisely to preserve
    that text. Authorization headers / api keys are never placed in the payload
    in the first place; this pass is the backstop for secrets echoed inside
    message content or provider error bodies."""
    if isinstance(value, Mapping):
        return {str(key): _redact_raw_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_raw_trace_value(item) for item in value]
    if isinstance(value, str):
        text = re.sub(r"sk-or-[A-Za-z0-9._:-]+", "[redacted-openrouter-key]", value)
        text = re.sub(r"sb_secret_[A-Za-z0-9._:-]+", "[redacted-supabase-service-role-key]", text)
        text = re.sub(r"sb_publishable_[A-Za-z0-9._:-]+", "[redacted-supabase-anon-key]", text)
        text = re.sub(r"AKIA[A-Z0-9]{16}", "[redacted-aws-access-key-id]", text)
        text = re.sub(r"https?://[^@\s]+@([^\s/]+)", r"[redacted-proxy-url]@\1", text)
        text = re.sub(r"(?i)(api_key=)[^&\s]+", r"\1[redacted]", text)
        return text
    return value


def _raw_trace_path_segment(value: object, *, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or ""))[:80]
    return text or fallback


class _OpenRouterRawTraceRecorder:
    """Best-effort KMS-encrypted S3 capture of raw OpenRouter request/response
    JSON at the chat-completion boundary (fableanalysis §9.1 item 5).

    Objects land under ``{prefix}/trajectories/{run_id}/{stage}/{seq}-{stage}.json.enc``
    with SSE-KMS (``RESEARCH_LAB_TRACE_KMS_KEY_ID``), where ``{prefix}`` comes
    from ``RESEARCH_LAB_RAW_TRACE_S3_PREFIX`` or falls back to the private
    model manifest's bucket/prefix. Hard rules:

      * a capture failure can NEVER fail or slow the LLM call — the S3 write is
        fire-and-forget on a small background pool, every synchronous step is
        exception-wrapped, and failures log once per run (not per call);
      * loop-event / provider-usage docs receive ONLY ``{s3_ref, sha256}``
        pointers — the trajectory-corpus protected-material scanner rejects any
        record carrying raw ``prompt``/``llm_response``/``page_content`` text;
      * request payloads are credential-redacted before write (Authorization
        headers and api keys are never included; ``_redact_raw_trace_value``
        backstops secrets echoed in content);
      * the pointer is optimistic: it is returned while the write is in flight,
        so a failed write leaves a dangling (never wrong) reference.

    Local/dev inertness: with no resolvable ``s3://`` destination the recorder
    logs once and stays silent; a missing boto3 disables it for the process
    after one logged failure.
    """

    def __init__(self, config: ResearchLabGatewayConfig):
        self.config = config
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._pending: set[Future[None]] = set()
        self._run_seq: Counter[str] = Counter()
        self._failure_logged_runs: set[str] = set()
        self._availability_warning_logged = False
        self._destination: tuple[str, str] | None = None
        self._destination_resolved = False
        self._disabled = False

    def capture(
        self,
        *,
        run_id: str,
        stage: str,
        request_doc: Mapping[str, Any],
        response_doc: Any,
        outcome: str,
    ) -> dict[str, str] | None:
        """Queue one request/response trace for encrypted upload.

        Returns the ``{s3_ref, sha256}`` pointer for event docs, or None when
        capture is disabled/unconfigured. Never raises."""
        try:
            if self._disabled or not run_id:
                return None
            if not _raw_trace_capture_enabled():
                self._log_availability_warning(
                    "disabled_by_env",
                    f"set {_RAW_TRACE_CAPTURE_ENABLED_ENV}=true to capture raw prompt/response traces",
                )
                return None
            destination = self._resolve_destination()
            if destination is None:
                return None
            kms_key_id = _trace_kms_key_id()
            if not kms_key_id:
                self._log_availability_warning(
                    "missing_kms_key",
                    f"set {_TRACE_KMS_KEY_ENV}=alias/leadpoet-research-lab-trace-encryption",
                )
                return None
            bucket, key_prefix = destination
            safe_run = _raw_trace_path_segment(run_id, fallback="run")
            safe_stage = _raw_trace_path_segment(stage, fallback="call")
            with self._lock:
                self._run_seq[safe_run] += 1
                seq = self._run_seq[safe_run]
            object_key = "/".join(
                segment
                for segment in (
                    key_prefix,
                    "trajectories",
                    safe_run,
                    safe_stage,
                    f"{seq:04d}-{safe_stage}.json.enc",
                )
                if segment
            )
            # P11: derived axis-A/B provenance on every persisted call. The
            # emitter/teacher values come from the single auditable mapping in
            # research_lab/axis_provenance.py, never per-stream constants.
            provenance = provenance_for_stage(stage)
            trace_doc: dict[str, Any] = {
                "schema_version": "1.1",
                "artifact_type": "research_lab_raw_llm_trace",
                "run_id": str(run_id),
                "stage": str(stage or ""),
                "seq": seq,
                "outcome": str(outcome or ""),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "call_emitter": provenance["call_emitter"],
                "purpose": provenance["purpose"],
                "component": provenance["component"],
                "teacher_model_flag": provenance["teacher_model_flag"],
                "request": dict(request_doc),
                "response": response_doc,
            }
            episode = current_call_episode()
            if episode:
                trace_doc["episode"] = episode
                trace_doc["episode_id"] = episode_id(episode)
            payload = _redact_raw_trace_value(trace_doc)
            body = canonical_json(payload).encode("utf-8")
            digest = sha256_bytes(body)
            self._submit(
                run_id=str(run_id),
                bucket=bucket,
                object_key=object_key,
                body=body,
                kms_key_id=kms_key_id,
            )
            return {"s3_ref": f"s3://{bucket}/{object_key}", "sha256": digest}
        except Exception as exc:
            self._log_failure_once(str(run_id or "unknown"), exc)
            return None

    def flush(self, timeout_seconds: float = 10.0) -> None:
        """Wait for in-flight uploads (tests / orderly teardown only — the hot
        path never calls this)."""
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

    def _resolve_destination(self) -> tuple[str, str] | None:
        with self._lock:
            if self._destination_resolved:
                return self._destination
        prefix_uri = str(os.getenv(_RAW_TRACE_S3_PREFIX_ENV, "")).strip().rstrip("/")
        if not prefix_uri:
            manifest_uri = str(self.config.private_model_manifest_uri or "").strip()
            if manifest_uri.startswith("s3://"):
                bucket, _sep, key = manifest_uri[5:].partition("/")
                base_prefix = key.rsplit("/", 1)[0] if "/" in key else ""
                if bucket:
                    prefix_uri = f"s3://{bucket}/{base_prefix}".rstrip("/")
        destination: tuple[str, str] | None = None
        if prefix_uri.startswith("s3://"):
            bucket, _sep, key_prefix = prefix_uri[5:].partition("/")
            if bucket:
                destination = (bucket, key_prefix.strip("/"))
        with self._lock:
            self._destination = destination
            self._destination_resolved = True
            if destination is None:
                self._disabled = True
        if destination is None:
            # Local/dev: no S3 destination — one warning, then the path stays inert.
            self._log_availability_warning(
                "missing_s3_prefix",
                (
                    f"set {_RAW_TRACE_S3_PREFIX_ENV}=s3://bucket/prefix or configure "
                    "private_model_manifest_uri with an s3:// parent"
                ),
            )
        return destination

    def _log_availability_warning(self, reason: str, hint: str) -> None:
        with self._lock:
            if self._availability_warning_logged:
                return
            self._availability_warning_logged = True
        logger.warning(
            (
                "research_lab_raw_trace_capture_unavailable reason=%s env=%s hint=%s; "
                "OpenRouter reasoning/full raw traces will not be stored for this process"
            ),
            reason,
            _RAW_TRACE_S3_PREFIX_ENV,
            hint,
        )

    def _submit(
        self,
        *,
        run_id: str,
        bucket: str,
        object_key: str,
        body: bytes,
        kms_key_id: str,
    ) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="research-lab-raw-trace",
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
        future.add_done_callback(lambda done: self._consume_put_result(run_id, done))

    def _put_object(self, *, bucket: str, object_key: str, body: bytes, kms_key_id: str) -> None:
        import boto3  # type: ignore

        client_kwargs: dict[str, Any] = {}
        try:
            from botocore.config import Config as BotoClientConfig  # type: ignore

            client_kwargs["config"] = BotoClientConfig(
                connect_timeout=_RAW_TRACE_PUT_CONNECT_TIMEOUT_SECONDS,
                read_timeout=_RAW_TRACE_PUT_READ_TIMEOUT_SECONDS,
                retries={"max_attempts": 2},
            )
        except Exception:  # pragma: no cover - botocore ships with boto3
            client_kwargs = {}
        put_kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": object_key,
            "Body": body,
            "ContentType": "application/json",
            # SSE-KMS at rest is the §9.1 encryption requirement. This must be
            # an ENCRYPT_DECRYPT KMS key, not the score-bundle SIGN_VERIFY key.
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": kms_key_id,
        }
        boto3.client("s3", **client_kwargs).put_object(**put_kwargs)

    def _consume_put_result(self, run_id: str, future: "Future[None]") -> None:
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
        self._log_failure_once(run_id, exc)

    def _log_failure_once(self, run_id: str, exc: BaseException) -> None:
        run_key = str(run_id or "unknown")
        with self._lock:
            if run_key in self._failure_logged_runs:
                return
            self._failure_logged_runs.add(run_key)
        logger.warning(
            "research_lab_raw_trace_capture_failed run=%s error=%s; capture is "
            "best-effort and the LLM call was not affected",
            compact_ref(run_key),
            _redact_openrouter_diagnostic(f"{exc.__class__.__name__}: {exc}", limit=240),
        )


@dataclass(frozen=True)
class HostedWorkerOutcome:
    processed: bool
    dry_run: bool
    run_id: str | None = None
    ticket_id: str | None = None
    status: str = "idle"
    receipt_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    score_bundle_ids: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "dry_run": self.dry_run,
            "run_id": self.run_id,
            "ticket_id": self.ticket_id,
            "status": self.status,
            "receipt_id": self.receipt_id,
            "candidate_ids": list(self.candidate_ids),
            "score_bundle_ids": list(self.score_bundle_ids),
            "error": self.error,
        }


@dataclass
class HostedRunContext:
    queue_row: Mapping[str, Any]
    ticket: Mapping[str, Any]
    payment: Mapping[str, Any] | None
    ticket_events: tuple[Mapping[str, Any], ...] = ()
    queue_events: tuple[Mapping[str, Any], ...] = ()
    receipt_id: str | None = None
    provider_env: dict[str, str] = field(default_factory=dict)
    # Set when a queue heartbeat observed that another worker owns this run (or
    # the DB claim guard fenced our heartbeat); gates further OpenRouter spend.
    claim_lost: bool = False
    # Loop-start credit ref resolved beyond the run-context event window
    # (see _resolve_loop_start_credit_id).
    resolved_loop_start_credit_id: str | None = None
    loop_start_credit_id_resolved: bool = False
    openrouter_key_ref: str = ""
    openrouter_management_key: str = ""
    # Latest cumulative loop cost evidence observed in-process before writing
    # the loop event. Used if the store write itself fails after miner spend.
    latest_loop_cost_evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return str(self.queue_row["run_id"])

    @property
    def ticket_id(self) -> str:
        return str(self.queue_row["ticket_id"])


class OpenRouterKeyResolver:
    """Resolve miner key refs to process-local env values without persisting raw keys."""

    def __init__(self, config: ResearchLabGatewayConfig):
        self.config = config

    async def resolve(self, key_ref: str, *, miner_hotkey: str) -> dict[str, str]:
        value = await self._resolve_key_value(key_ref, miner_hotkey=miner_hotkey)
        return {
            "OPENROUTER_API_KEY": value,
            "QUALIFICATION_OPENROUTER_API_KEY": value,
            "OPENROUTER_KEY": value,
        }

    async def resolve_management_key(self, key_ref: str, *, miner_hotkey: str) -> str:
        if not str(key_ref).startswith("encrypted_ref:openrouter:"):
            raise HostedResearchLabWorkerError("encrypted OpenRouter key ref with management proof is required")
        row = await select_one(
            "research_lab_openrouter_key_refs",
            filters=(("key_ref", key_ref), ("miner_hotkey", miner_hotkey)),
        )
        if not row:
            raise HostedResearchLabWorkerError("encrypted OpenRouter key ref was not found for miner")
        if str(row.get("privacy_status") or "") != "verified":
            raise HostedResearchLabWorkerError("encrypted OpenRouter key ref has not passed privacy verification")
        ciphertext = str(row.get("encrypted_management_key_ciphertext") or "").strip()
        if not ciphertext:
            raise HostedResearchLabWorkerError("encrypted OpenRouter management key is missing")
        try:
            return await asyncio.to_thread(
                decrypt_openrouter_key,
                ciphertext_b64=ciphertext,
                miner_hotkey=miner_hotkey,
                key_ref=key_ref,
            )
        except OpenRouterKeyVaultError as exc:
            raise HostedResearchLabWorkerError(str(exc)) from exc

    async def _resolve_key_value(self, key_ref: str, *, miner_hotkey: str) -> str:
        if str(key_ref).startswith("encrypted_ref:openrouter:"):
            row = await select_one(
                "research_lab_openrouter_key_refs",
                filters=(("key_ref", key_ref), ("miner_hotkey", miner_hotkey)),
            )
            if not row:
                raise HostedResearchLabWorkerError("encrypted OpenRouter key ref was not found for miner")
            try:
                return await asyncio.to_thread(
                    decrypt_openrouter_key,
                    ciphertext_b64=str(row["encrypted_key_ciphertext"]),
                    miner_hotkey=miner_hotkey,
                    key_ref=key_ref,
                )
            except OpenRouterKeyVaultError as exc:
                raise HostedResearchLabWorkerError(str(exc)) from exc

        env_name = self._env_name_for_ref(key_ref)
        if not env_name:
            raise HostedResearchLabWorkerError("no OpenRouter key env var configured for miner key ref")
        value = os.getenv(env_name)
        if not value:
            raise HostedResearchLabWorkerError(f"configured OpenRouter key env var is empty: {env_name}")
        return value

    def _env_name_for_ref(self, key_ref: str) -> str:
        if self.config.miner_openrouter_key_ref_env_map_json:
            try:
                mapping = json.loads(self.config.miner_openrouter_key_ref_env_map_json)
            except json.JSONDecodeError as exc:
                raise HostedResearchLabWorkerError("invalid OpenRouter key-ref env map JSON") from exc
            if not isinstance(mapping, Mapping):
                raise HostedResearchLabWorkerError("OpenRouter key-ref env map must be an object")
            mapped = mapping.get(str(key_ref))
            if mapped:
                return str(mapped)
        return self.config.miner_openrouter_key_env_var


class ResearchLabHostedWorker:
    """Poll and execute one hosted Research Lab queue run at a time."""

    def __init__(self, config: ResearchLabGatewayConfig | None = None, *, worker_ref: str | None = None):
        self.config = config or ResearchLabGatewayConfig.from_env()
        self.worker_ref = (
            worker_ref
            or self.config.hosted_worker_id
            or f"research-lab-hosted-worker:{os.uname().nodename}:{os.getpid()}"
        )
        self.key_resolver = OpenRouterKeyResolver(self.config)
        # §9.1 item 5: raw prompt/response capture at the OpenRouter boundary.
        self._raw_trace_recorder = _OpenRouterRawTraceRecorder(self.config)

    async def run_forever(self) -> None:
        # trajectoryimprovements.md P5: one structured capture health block at
        # startup; refuses to start in production when capture is degraded.
        from gateway.research_lab.capture_health import enforce_capture_health

        enforce_capture_health(self.config, worker_kind="hosted_worker")
        processed = 0
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
                            "RESEARCH LAB AUTO-RESEARCH POLL FAILED",
                            (
                                ("Worker", self.worker_ref),
                                ("Error", f"{exc.__class__.__name__}: {str(exc)[:300]}"),
                            ),
                        )
                    )
                    last_error_log = now
                await asyncio.sleep(max(self.config.hosted_worker_poll_seconds, error_backoff_seconds))
                continue
            if outcome.processed or outcome.status != "idle":
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB AUTO-RESEARCH PASS",
                        (
                            ("Worker", self.worker_ref),
                            ("Status", outcome.status),
                            ("Run", compact_ref(outcome.run_id)),
                            ("Receipt", compact_ref(outcome.receipt_id)),
                            ("Candidates", len(outcome.candidate_ids)),
                            ("Error", outcome.error),
                        ),
                    )
                )
            elif time.monotonic() - last_idle_log >= idle_log_seconds:
                logger.info(
                    "Research Lab hosted worker idle: worker_ref=%s poll_seconds=%s",
                    self.worker_ref,
                    self.config.hosted_worker_poll_seconds,
                )
                last_idle_log = time.monotonic()
            if outcome.processed:
                processed += 1
            if self.config.hosted_worker_max_runs and processed >= self.config.hosted_worker_max_runs:
                return
            await asyncio.sleep(max(1, self.config.hosted_worker_poll_seconds))

    async def run_once(self) -> HostedWorkerOutcome:
        self._require_enabled()
        if not self.config.hosted_worker_dry_run:
            await self._recover_stale_started_runs()
            await self._reconcile_stale_loop_projections()
        if await is_autoresearch_maintenance_paused():
            return HostedWorkerOutcome(
                processed=False,
                dry_run=self.config.hosted_worker_dry_run,
                status="maintenance_paused",
            )
        if not self.config.hosted_worker_dry_run:
            await self._recover_stale_paused_runs()
            await self._run_periodic_reconciles()
        builder_unavailable = self._code_edit_builder_unavailable_reason()
        if builder_unavailable:
            logger.warning(
                "research_lab_code_edit_builder_not_ready worker_ref=%s reason=%s",
                self.worker_ref,
                builder_unavailable[:240],
            )
            return HostedWorkerOutcome(
                processed=False,
                dry_run=self.config.hosted_worker_dry_run,
                status="code_edit_builder_not_ready",
                error=builder_unavailable[:500],
            )
        queued = await self._next_queued_run()
        if not queued:
            return HostedWorkerOutcome(processed=False, dry_run=self.config.hosted_worker_dry_run)
        run_id = str(queued["run_id"])
        ticket_id = str(queued["ticket_id"])
        if self.config.hosted_worker_dry_run:
            return HostedWorkerOutcome(
                processed=False,
                dry_run=True,
                run_id=run_id,
                ticket_id=ticket_id,
                status="dry_run_queued_run_found",
            )
        try:
            self._require_worker_proxy_for_execution()
        except HostedResearchLabWorkerError as exc:
            logger.error(
                format_worker_block(
                    "RESEARCH LAB AUTO-RESEARCH BLOCKED",
                    (
                        ("Worker", self.worker_ref),
                        ("Run", compact_ref(run_id)),
                        ("Ticket", compact_ref(ticket_id)),
                        ("Reason", "worker_proxy_required"),
                    ),
                )
            )
            return HostedWorkerOutcome(
                processed=False,
                dry_run=False,
                run_id=run_id,
                ticket_id=ticket_id,
                status="worker_proxy_required",
                error=str(exc)[:500],
            )
        context = await self._load_run_context(queued)
        try:
            return await self._process_run(context)
        except HostedResearchLabClaimLost as exc:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB AUTO-RESEARCH CLAIM LOST",
                    (
                        ("Worker", self.worker_ref),
                        ("Run", compact_ref(run_id)),
                        ("Ticket", compact_ref(ticket_id)),
                        ("Reason", str(exc)[:300]),
                    ),
                )
            )
            return HostedWorkerOutcome(
                processed=False,
                dry_run=False,
                run_id=run_id,
                ticket_id=ticket_id,
                status="claim_lost",
                error=str(exc)[:500],
            )
        except CreditBlockedHostedRunError as exc:
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB AUTO-RESEARCH BLOCKED FOR CREDIT",
                    (
                        ("Worker", self.worker_ref),
                        ("Run", compact_ref(run_id)),
                        ("Ticket", compact_ref(ticket_id)),
                        ("Reason", str(exc)[:300]),
                    ),
                )
            )
            return await self._mark_blocked_for_credit(context, str(exc))
        except HostedResearchLabBuilderNotReady as exc:
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB AUTO-RESEARCH BUILDER NOT READY",
                    (
                        ("Worker", self.worker_ref),
                        ("Run", compact_ref(run_id)),
                        ("Ticket", compact_ref(ticket_id)),
                        ("Reason", str(exc)[:300]),
                    ),
                )
            )
            return await self._mark_builder_not_ready(context, str(exc))
        except Exception as exc:
            if _is_retryable_worker_exception(exc):
                retry_count = _retryable_requeue_count(context)
                retry_limit = int(self.config.hosted_worker_retryable_failure_limit)
                if retry_count < retry_limit:
                    logger.warning(
                        format_worker_block(
                            "RESEARCH LAB AUTO-RESEARCH TRANSIENT FAILURE REQUEUED",
                            (
                                ("Worker", self.worker_ref),
                                ("Run", compact_ref(run_id)),
                                ("Ticket", compact_ref(ticket_id)),
                                ("Retry", f"{retry_count + 1}/{retry_limit}"),
                                ("Error", str(exc)[:300]),
                            ),
                        )
                    )
                    return await self._mark_retryable(context, str(exc), retry_count=retry_count + 1)
                logger.error(
                    format_worker_block(
                        "RESEARCH LAB AUTO-RESEARCH TRANSIENT RETRY LIMIT EXCEEDED",
                        (
                            ("Worker", self.worker_ref),
                            ("Run", compact_ref(run_id)),
                            ("Ticket", compact_ref(ticket_id)),
                            ("Retries", retry_count),
                            ("Error", str(exc)[:300]),
                        ),
                    )
                )
            logger.exception(
                format_worker_block(
                    "RESEARCH LAB AUTO-RESEARCH FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Run", compact_ref(run_id)),
                        ("Ticket", compact_ref(ticket_id)),
                        ("Error", str(exc)[:300]),
                    ),
                )
            )
            return await self._mark_failed(context, str(exc), failure_exception=exc)

    def _require_enabled(self) -> None:
        if not self.config.hosted_worker_enabled:
            raise HostedResearchLabWorkerError("Research Lab hosted worker is disabled")
        if not self.config.production_writes_enabled:
            raise HostedResearchLabWorkerError("Research Lab production writes are disabled")
        if not self.config.hosted_runs_enabled:
            raise HostedResearchLabWorkerError("Research Lab hosted runs are disabled")
        if not self.config.receipts_enabled:
            raise HostedResearchLabWorkerError("Research Lab receipt writes are disabled")
        if not self.config.private_model_manifest_uri:
            raise HostedResearchLabWorkerError("private model manifest URI is not configured")
        if not self.config.approved_auto_research_models():
            raise HostedResearchLabWorkerError("auto-research OpenRouter model is not configured")

    def _code_edit_builder_unavailable_reason(self) -> str | None:
        if not self.config.code_edit_candidates_enabled:
            return "RESEARCH_LAB_CODE_EDIT_CANDIDATES_ENABLED is false"
        missing: list[str] = []
        if not self.config.private_test_cmd:
            missing.append("RESEARCH_LAB_PRIVATE_TEST_CMD")
        if not self.config.private_build_cmd:
            missing.append("RESEARCH_LAB_PRIVATE_BUILD_CMD")
        if not self.config.private_artifact_manifest_output:
            missing.append("RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT")
        if missing:
            return "code-edit image-build candidate builder missing: " + ", ".join(missing)
        if not CodeEditCandidateBuilder(self.config).enabled():
            return "code-edit image-build candidate builder is not configured"
        return None

    def _require_code_edit_builder_ready(self) -> None:
        reason = self._code_edit_builder_unavailable_reason()
        if reason:
            raise HostedResearchLabBuilderNotReady(reason)

    def _require_worker_proxy_for_execution(self) -> None:
        if not self.config.hosted_worker_require_proxy:
            return
        if not _worker_proxy_url(self.config):
            raise HostedResearchLabWorkerError(
                "worker proxy is required for hosted auto-research execution; "
                "set RESEARCH_LAB_HOSTED_WORKER_PROXY or per-worker RESEARCH_LAB_WORKER_PROXY_N"
            )

    async def _next_queued_run(self) -> Mapping[str, Any] | None:
        rows = await select_many(
            "research_loop_run_queue_current",
            filters=(("current_queue_status", "queued"),),
            order_by=(("queue_priority", False), ("current_status_at", False)),
            limit=self.config.hosted_worker_queue_fetch_limit,
        )
        return self._select_preferred_queued_row(rows)

    def _select_preferred_queued_row(self, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        if not rows:
            return None
        total_workers = max(1, int(self.config.hosted_worker_total_workers or 1))
        worker_index = int(self.config.hosted_worker_index or 0) % total_workers
        if total_workers <= 1:
            return rows[0]
        for row in rows:
            if _row_partition(row, total_workers) == worker_index:
                return row
        # Avoid starvation if this worker's preferred shard is temporarily empty.
        # Claim conflicts are handled as no-ops, not failures.
        return rows[0]

    async def _run_periodic_reconciles(self) -> None:
        """Every-Nth-pass maintenance reconciles (bugs 3, 24, 34).

        Each is idempotent, self-limiting (acts only on drifted rows), and
        best-effort — a reconcile failure never blocks run processing.
        """
        counter = getattr(self, "_periodic_reconcile_pass", 0) + 1
        self._periodic_reconcile_pass = counter
        try:
            every_n = max(1, int(os.getenv("RESEARCH_LAB_WORKER_PERIODIC_RECONCILE_EVERY_N", "10")))
        except ValueError:
            every_n = 10
        if counter % every_n:
            return
        try:
            await reproject_stale_public_cards(config=self.config)
        except Exception as exc:
            logger.warning(
                "research_lab_periodic_reprojection_sweep_failed worker_ref=%s error=%s",
                self.worker_ref,
                str(exc)[:200],
            )
        try:
            await reconcile_active_private_model_lineage(
                actor_ref=self.worker_ref, dry_run=False
            )
        except Exception as exc:
            logger.warning(
                "research_lab_periodic_lineage_reconcile_failed worker_ref=%s error=%s",
                self.worker_ref,
                str(exc)[:200],
            )
        try:
            await reconcile_pending_champion_rewards(
                self.config, worker_ref=self.worker_ref, dry_run=False
            )
        except Exception as exc:
            logger.warning(
                "research_lab_periodic_reward_reconcile_failed worker_ref=%s error=%s",
                self.worker_ref,
                str(exc)[:200],
            )
        try:
            from gateway.research_lab.trajectory_projector import (
                backfill_corpus_trace_rows,
                project_completed_runs,
                projector_enabled,
            )

            # §9.1 trajectory capture: inert until scripts/27 is applied and
            # RESEARCH_LAB_TRAJECTORY_PROJECTOR_ENABLED=true.
            if projector_enabled():
                await project_completed_runs(dry_run=False)
                trace_backfill_max_candidates = _env_int(
                    "RESEARCH_LAB_TRAJECTORY_BACKFILL_MAX_CANDIDATES", 50
                )
                await backfill_corpus_trace_rows(
                    batch_size=_env_int("RESEARCH_LAB_TRAJECTORY_BACKFILL_BATCH_SIZE", 2),
                    dry_run=False,
                    max_candidates=trace_backfill_max_candidates,
                    max_attempts=_env_int(
                        "RESEARCH_LAB_TRAJECTORY_BACKFILL_MAX_ATTEMPTS",
                        trace_backfill_max_candidates,
                    ),
                )
        except Exception as exc:
            logger.warning(
                "research_lab_periodic_trajectory_projection_failed worker_ref=%s error=%s",
                self.worker_ref,
                str(exc)[:200],
            )

    async def _recover_stale_started_runs(self) -> int:
        stale_after_seconds = max(
            60,
            int(self.config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
        )
        rows = await select_many(
            "research_loop_run_queue_current",
            columns=(
                "run_id,ticket_id,current_queue_status,current_status_at,"
                "current_event_hash,queue_priority,worker_ref"
            ),
            filters=(("current_queue_status", "started"),),
            # Oldest-first so the stalest rows are still seen when the backlog
            # exceeds the fetch limit.
            order_by=(("current_status_at", False),),
            limit=50,
        )
        recovered = 0
        for row in rows:
            if not _status_is_stale(row.get("current_status_at"), stale_after_seconds):
                continue
            run_id = str(row.get("run_id") or "")
            ticket_id = str(row.get("ticket_id") or "")
            if not run_id or not ticket_id:
                continue
            if await self._loop_activity_blocks_stale_requeue(run_id, stale_after_seconds):
                continue
            try:
                await create_queue_event(
                    run_id=run_id,
                    ticket_id=ticket_id,
                    event_type="queued",
                    queue_priority=int(row.get("queue_priority") or 0),
                    worker_ref=self.worker_ref,
                    reason="stale_started_requeued",
                    event_doc={
                        **autoresearch_queue_capacity_doc(self.config),
                        "recovering_worker_ref": self.worker_ref,
                        "previous_worker_ref": row.get("worker_ref"),
                        "previous_event_hash": row.get("current_event_hash"),
                        "previous_status_at": row.get("current_status_at"),
                        "stale_after_seconds": stale_after_seconds,
                    },
                )
                recovered += 1
            except Exception as exc:
                logger.warning(
                    "research_lab_stale_hosted_run_requeue_failed run_id=%s error=%s",
                    compact_ref(run_id),
                    str(exc)[:240],
                )
        if recovered:
            logger.info(
                "research_lab_stale_hosted_runs_requeued worker_ref=%s count=%s stale_after_seconds=%s",
                self.worker_ref,
                recovered,
                stale_after_seconds,
            )
        return recovered

    async def _recover_stale_paused_runs(self) -> int:
        stale_after_seconds = max(
            60,
            int(self.config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
        )
        rows = await select_many(
            "research_loop_run_queue_current",
            columns=(
                "run_id,ticket_id,current_queue_status,current_reason,current_status_at,"
                "current_event_hash,queue_priority,worker_ref"
            ),
            filters=(("current_queue_status", "paused"),),
            # Oldest-first so the stalest rows are still seen when the backlog
            # exceeds the fetch limit.
            order_by=(("current_status_at", False),),
            limit=50,
        )
        recovered = 0
        for row in rows:
            if str(row.get("current_reason") or "") in _STALE_PAUSED_REAPER_EXCLUDED_REASONS:
                # Deliberately parked runs (e.g. blocked_for_credit awaiting a
                # key top-up) are only revived by their explicit resume paths.
                continue
            if not _status_is_stale(row.get("current_status_at"), stale_after_seconds):
                continue
            run_id = str(row.get("run_id") or "")
            ticket_id = str(row.get("ticket_id") or "")
            if not run_id or not ticket_id:
                continue
            if await self._loop_activity_blocks_stale_requeue(run_id, stale_after_seconds):
                continue
            try:
                await create_queue_event(
                    run_id=run_id,
                    ticket_id=ticket_id,
                    event_type="queued",
                    queue_priority=int(row.get("queue_priority") or 0),
                    worker_ref=self.worker_ref,
                    reason="stale_paused_requeued",
                    event_doc={
                        **autoresearch_queue_capacity_doc(self.config),
                        "resume_source": "hosted_worker_stale_paused_reaper",
                        "recovering_worker_ref": self.worker_ref,
                        "previous_worker_ref": row.get("worker_ref"),
                        "previous_event_hash": row.get("current_event_hash"),
                        "previous_status_at": row.get("current_status_at"),
                        "stale_after_seconds": stale_after_seconds,
                    },
                )
                recovered += 1
            except Exception as exc:
                logger.warning(
                    "research_lab_stale_paused_run_requeue_failed run_id=%s error=%s",
                    compact_ref(run_id),
                    str(exc)[:240],
                )
        if recovered:
            logger.info(
                "research_lab_stale_paused_runs_requeued worker_ref=%s count=%s stale_after_seconds=%s",
                self.worker_ref,
                recovered,
                stale_after_seconds,
            )
        return recovered

    async def _loop_activity_blocks_stale_requeue(self, run_id: str, stale_after_seconds: int) -> bool:
        row = await select_one(
            "research_lab_auto_research_loop_current",
            columns=(
                "run_id,current_loop_status,current_status_at,current_event_seq,"
                "current_event_type,current_worker_ref"
            ),
            filters=(("run_id", run_id),),
        )
        if not row:
            return False
        loop_status = str(row.get("current_loop_status") or "")
        if loop_status == "completed":
            logger.info(
                "research_lab_stale_hosted_requeue_skipped_completed_loop run_id=%s loop_seq=%s",
                compact_ref(run_id),
                row.get("current_event_seq"),
            )
            return True
        if loop_status in {"running", "paused"} and not _status_is_stale(
            row.get("current_status_at"),
            stale_after_seconds,
        ):
            logger.info(
                "research_lab_stale_hosted_requeue_skipped_active_loop run_id=%s loop_status=%s loop_seq=%s loop_worker=%s",
                compact_ref(run_id),
                loop_status,
                row.get("current_event_seq"),
                row.get("current_worker_ref"),
            )
            return True
        return False

    async def _load_run_context(self, queue_row: Mapping[str, Any]) -> HostedRunContext:
        ticket = await select_one(
            "research_loop_ticket_current",
            filters=(("ticket_id", str(queue_row["ticket_id"])),),
        )
        if not ticket:
            raise HostedResearchLabWorkerError("queued Research Lab run is missing ticket")
        ticket_events = await select_many(
            "research_loop_ticket_events",
            filters=(("ticket_id", str(queue_row["ticket_id"])),),
            order_by=(("seq", True),),
            limit=20,
        )
        queue_events = await select_many(
            "research_loop_run_queue_events",
            filters=(("run_id", str(queue_row["run_id"])),),
            order_by=(("seq", True),),
            limit=_RUN_CONTEXT_QUEUE_EVENT_LIMIT,
        )
        payment = None
        payment_id = _payment_id_from_queue_events(queue_events)
        if payment_id:
            payment = await select_one("research_loop_start_payments", filters=(("payment_id", payment_id),))
        if not payment:
            payment_rows = await select_many(
                "research_loop_start_payments",
                filters=(("ticket_id", str(queue_row["ticket_id"])),),
                order_by=(("verified_at", True),),
                limit=1,
            )
            payment = payment_rows[0] if payment_rows else None
        return HostedRunContext(
            queue_row=queue_row,
            ticket=ticket,
            payment=payment,
            ticket_events=tuple(ticket_events),
            queue_events=tuple(queue_events),
        )

    async def _process_run(self, context: HostedRunContext) -> HostedWorkerOutcome:
        terminal = await self._already_completed_outcome(context)
        if terminal:
            return terminal
        await self._append_started_events(context)
        self._require_code_edit_builder_ready()
        completed_receipt_outcome = await self._complete_from_existing_completed_receipt(context)
        if completed_receipt_outcome:
            return completed_receipt_outcome
        context.receipt_id = await self._ensure_queued_receipt(context)
        existing_candidate_outcome = await self._complete_from_existing_candidate_artifacts(context)
        if existing_candidate_outcome:
            return existing_candidate_outcome
        if await is_autoresearch_maintenance_paused():
            return await self._mark_paused(
                context,
                loop_result=None,
                checkpoint_doc=None,
                reason="maintenance_pause_before_execution",
            )
        openrouter_key_ref = _miner_openrouter_key_ref(context)
        resolved_openrouter_env = await self.key_resolver.resolve(
            openrouter_key_ref,
            miner_hotkey=str(context.ticket["miner_hotkey"]),
        )
        context.openrouter_key_ref = openrouter_key_ref
        context.openrouter_management_key = await self.key_resolver.resolve_management_key(
            openrouter_key_ref,
            miner_hotkey=str(context.ticket["miner_hotkey"]),
        )
        provider_env = dict(resolved_openrouter_env)
        context.provider_env = provider_env
        # Credit gate before any expensive OpenRouter generation (covers start + resume).
        await self._preflight_openrouter_credit(context, provider_env)
        docker_provider_env = _private_model_docker_env(
            self.config,
            {
                **provider_env,
                **_worker_proxy_env(self.config),
            },
        )
        budget_context = self._run_budget_context(context)
        _tier, model_id, model_doc = self.config.resolve_auto_research_model(
            str(budget_context.get("research_model_tier") or "")
        )
        max_candidates = self._max_candidates_for_run(budget_context, model_doc)
        resume_state = await latest_auto_research_checkpoint(context.run_id)

        active_start = await load_active_private_model(self.config, register_bootstrap=True)
        artifact = active_start.artifact
        # If the active model changed since this run was paused, the checkpoint is stale;
        # restart from scratch against the current model rather than resuming a stale parent.
        resume_state = self._validate_resume_state_freshness(resume_state, artifact, context.run_id)
        outcome_memory = await self._active_parent_outcome_memory(artifact)
        if outcome_memory:
            budget_context["active_parent_outcome_memory"] = outcome_memory

        def _load_runtime_metadata() -> Mapping[str, Any]:
            runner = DockerPrivateModelRunner(
                DockerPrivateModelSpec(
                    image_digest=artifact.image_digest,
                    env_passthrough=_private_model_env_passthrough(self.config),
                    extra_env=docker_provider_env,
                    timeout_seconds=900,
                )
            )
            return runner.metadata()

        with _temporary_env(provider_env):
            metadata = await self._to_thread_with_queue_heartbeat(
                context,
                heartbeat_label="private_runtime_metadata",
                func=_load_runtime_metadata,
            )
            registry = coerce_component_registry(metadata)
            benchmark_public_summary = await latest_public_benchmark_summary()
            logger.info(
                format_worker_block(
                    "RESEARCH LAB AUTO-RESEARCH STARTED",
                    (
                        ("Worker", self.worker_ref),
                        ("Run", compact_ref(context.run_id)),
                        ("Ticket", compact_ref(context.ticket_id)),
                        ("Model", model_id),
                        ("Min runtime", f"{self.config.auto_research_min_seconds}s"),
                        ("Max runtime", f"{self.config.auto_research_max_seconds}s"),
                        ("Max iterations", self.config.auto_research_max_iterations),
                        ("Max candidates", max_candidates),
                    ),
                )
            )

            latest_checkpoint: dict[str, Any] | None = None
            last_queue_heartbeat_at = 0.0

            async def _record_loop_event(event: AutoResearchLoopEvent) -> None:
                nonlocal latest_checkpoint, last_queue_heartbeat_at
                if event.event_type == "checkpoint_saved":
                    checkpoint_doc = event.event_doc.get("checkpoint") if isinstance(event.event_doc, Mapping) else None
                    if isinstance(checkpoint_doc, dict):
                        latest_checkpoint = dict(checkpoint_doc)
                event_provider_usage = event.provider_usage or self._provider_usage(context)
                context.latest_loop_cost_evidence = normalize_cost_evidence(
                    {
                        "source": "loop_event_in_memory",
                        "source_event_type": event.event_type,
                        "provider_usage": event_provider_usage,
                        "cost_ledger": event.cost_ledger,
                        "elapsed_seconds": event.elapsed_seconds,
                    }
                )
                loop_event_row = await create_auto_research_loop_event(
                    run_id=context.run_id,
                    ticket_id=context.ticket_id,
                    receipt_id=context.receipt_id,
                    event_type=event.event_type,
                    loop_status=event.loop_status,
                    worker_ref=self.worker_ref,
                    node_id=event.node_id,
                    elapsed_seconds=event.elapsed_seconds,
                    candidate_artifact_hash=event.candidate_artifact_hash,
                    candidate_patch_hash=event.candidate_patch_hash,
                    provider_usage=event_provider_usage,
                    cost_ledger=event.cost_ledger,
                    event_doc=event.event_doc,
                )
                now_monotonic = time.monotonic()
                if (
                    event.event_type in {"candidate_build_started", "candidate_build_passed", "checkpoint_saved"}
                    or now_monotonic - last_queue_heartbeat_at >= 30.0
                ):
                    if await self._append_queue_heartbeat(
                        context,
                        source_event_type=event.event_type,
                        source_event_seq=loop_event_row.get("seq"),
                        source_event_hash=loop_event_row.get("anchored_hash"),
                    ):
                        last_queue_heartbeat_at = now_monotonic
                if event.event_type in {"candidate_selected", "loop_completed", "loop_failed"}:
                    logger.info(
                        format_worker_block(
                            f"RESEARCH LAB AUTO-RESEARCH {event.event_type.replace('_', ' ').upper()}",
                            (
                                ("Worker", self.worker_ref),
                                ("Run", compact_ref(context.run_id)),
                                ("Event", event.event_type),
                                ("Elapsed", f"{event.elapsed_seconds:.1f}s"),
                                ("Node", compact_ref(event.node_id)),
                            ),
                        )
                    )
                elif event.event_type in {"patch_validation_passed", "patch_validation_failed"}:
                    logger.info(
                        format_worker_line(
                            "Research Lab auto-research event",
                            worker=self.worker_ref,
                            run=compact_ref(context.run_id),
                            event=event.event_type,
                            elapsed=f"{event.elapsed_seconds:.1f}s",
                            node=compact_ref(event.node_id),
                        )
                    )

            async def _call_loop_model(
                messages: Sequence[Mapping[str, str]],
                timeout_seconds: int,
                max_tokens: int,
                call_stage: str = "code_edit_draft",
            ) -> str:
                if context.claim_lost:
                    # A heartbeat already observed that another worker owns this
                    # run; never start more OpenRouter spend on the miner's key
                    # even if an intermediate layer swallowed the abort.
                    raise HostedResearchLabClaimLost(
                        "hosted run claim was lost; refusing further OpenRouter calls"
                    )
                stage_options = _resolve_code_edit_loop_stage_model_request(
                    self.config,
                    stage=call_stage,
                    model_id=model_id,
                    model_doc=model_doc,
                    requested_max_tokens=max_tokens,
                )
                stage = stage_options["stage"]
                stage_model_id = str(stage_options["model_id"])
                stage_reasoning_effort = str(stage_options["reasoning_effort"])
                stage_temperature = float(stage_options["temperature"])
                stage_max_tokens = int(stage_options["max_tokens"])
                stage_model_ids = tuple(str(item) for item in stage_options.get("model_ids", ()) if str(item).strip())
                if not stage_model_ids:
                    stage_model_ids = (stage_model_id,)
                allow_non_zdr = bool(stage_options.get("allow_non_zdr"))
                if stage in {"loop_planner", "plan_alignment_judge"}:
                    effective_max_tokens = max(1, int(stage_max_tokens or 0))
                else:
                    effective_max_tokens = self._auto_research_max_tokens_for_call(
                        requested_max_tokens=stage_max_tokens,
                        model_doc=model_doc,
                    )
                last_exc: Exception | None = None
                fallback_usage: list[dict[str, Any]] = []
                for model_attempt_index, attempt_model_id in enumerate(stage_model_ids):
                    try:
                        result = await self._call_openrouter(
                            messages=messages,
                            api_key=context.provider_env["OPENROUTER_API_KEY"],
                            model_id=attempt_model_id,
                            reasoning_effort=stage_reasoning_effort,
                            timeout_seconds=timeout_seconds,
                            max_tokens=effective_max_tokens,
                            temperature=stage_temperature,
                            allow_non_zdr=allow_non_zdr,
                            capture_run_id=context.run_id,
                            capture_stage=stage,
                            privacy_key_ref=context.openrouter_key_ref,
                            privacy_miner_hotkey=str(context.ticket["miner_hotkey"]),
                            privacy_management_key=context.openrouter_management_key,
                        )
                        if fallback_usage:
                            provider_usage = dict(result.provider_usage or {})
                            provider_usage["model_fallback_attempts"] = fallback_usage
                            provider_usage["model_fallback_attempt_count"] = len(fallback_usage)
                            result = OpenRouterCallResult(
                                content=result.content,
                                provider_usage=provider_usage,
                                cost_microusd=result.cost_microusd,
                            )
                        return result
                    except CreditBlockedHostedRunError:
                        raise
                    except HostedResearchLabWorkerError as exc:
                        last_exc = exc
                        if model_attempt_index >= len(stage_model_ids) - 1:
                            raise
                        fallback_usage.append(
                            {
                                "stage": stage,
                                "model_ref": compact_ref(attempt_model_id),
                                "error_hash": sha256_json({"error": str(exc)}),
                                "next_model_ref": compact_ref(stage_model_ids[model_attempt_index + 1]),
                            }
                        )
                        logger.warning(
                            "research_lab_openrouter_stage_model_fallback stage=%s model=%s next_model=%s error_hash=%s",
                            stage,
                            compact_ref(attempt_model_id),
                            compact_ref(stage_model_ids[model_attempt_index + 1]),
                            fallback_usage[-1]["error_hash"],
                        )
                if last_exc is not None:
                    raise last_exc
                raise HostedResearchLabWorkerError("OpenRouter stage model resolution failed")

            loop_settings = AutoResearchLoopSettings(
                min_seconds=self.config.auto_research_min_seconds,
                max_seconds=self.config.auto_research_max_seconds,
                min_iterations=self.config.auto_research_min_iterations,
                max_iterations=self.config.auto_research_max_iterations,
                draft_timeout_seconds=self.config.auto_research_draft_timeout_seconds,
                reflection_timeout_seconds=self.config.auto_research_reflection_timeout_seconds,
                estimated_iteration_cost_usd=self.config.auto_research_estimated_iteration_cost_usd,
                max_candidates=max_candidates,
            )
            code_builder = CodeEditCandidateBuilder(self.config)
            self._require_code_edit_builder_ready()

            loop_result = await CodeEditLoopEngine(
                settings=loop_settings,
                call_openrouter=_call_loop_model,
                event_sink=_record_loop_event,
                builder=code_builder,
            ).run(
                run_id=context.run_id,
                ticket=context.ticket,
                artifact=artifact,
                component_registry=registry.to_dict(),
                benchmark_public_summary=benchmark_public_summary,
                model_id=model_id,
                budget_context=budget_context,
                requested_loop_count=int(context.ticket.get("requested_loop_count") or 1),
                resume_state=resume_state,
                should_pause=is_autoresearch_maintenance_paused,
            )
            if loop_result.status == "paused":
                return await self._mark_paused(
                    context,
                    loop_result=loop_result,
                    checkpoint_doc=loop_result.checkpoint_doc or latest_checkpoint,
                    reason="maintenance_pause_checkpointed",
                )
            if not loop_result.selected_candidates:
                return await self._mark_failed(
                    context,
                    "auto-research loop completed without valid image-build finalists",
                    loop_result=loop_result,
                    reason="no_valid_image_build_finalists",
                )
            final_artifact = artifact
            finalists = [
                {
                    "candidate_kind": "image_build",
                    "selected": candidate,
                    "candidate_patch_manifest": candidate.build.code_edit_manifest,
                    "candidate_model_manifest": candidate.build.candidate_model_manifest.to_dict(),
                    "candidate_source_diff_hash": candidate.build.source_diff_hash,
                    "candidate_build_doc": {
                        **candidate.build.build_doc,
                        "loop_direction_plan_hash": (
                            candidate.draft.plan_alignment.get("loop_direction_plan_hash")
                            if isinstance(candidate.draft.plan_alignment, Mapping)
                            else None
                        ),
                        "selected_path_id": candidate.draft.plan_path_id,
                        "plan_alignment": dict(candidate.draft.plan_alignment or {}),
                    },
                    "hypothesis_doc": {
                        "failure_mode": candidate.draft.failure_mode,
                        "mechanism": candidate.draft.mechanism,
                        "expected_improvement": candidate.draft.expected_improvement,
                        "risk": candidate.draft.risk,
                        "focus_alignment": f"code_edit_lane:{candidate.draft.lane}",
                        "plan_path_id": candidate.draft.plan_path_id,
                        "plan_alignment": dict(candidate.draft.plan_alignment or {}),
                        "predicted_delta": candidate.draft.predicted_delta,
                        "falsifier": "official_scoring",
                    },
                    "patch_doc": {
                        "code_edit": {
                            "lane": candidate.draft.lane,
                            "plan_path_id": candidate.draft.plan_path_id,
                            "plan_alignment": dict(candidate.draft.plan_alignment or {}),
                            "target_files": list(candidate.draft.target_files),
                            "unified_diff_hash": sha256_json({"unified_diff": candidate.draft.unified_diff}),
                            "redacted_summary": candidate.draft.redacted_summary,
                            "test_plan": candidate.draft.test_plan,
                            "rollback_plan": candidate.draft.rollback_plan,
                        }
                    },
                    "iteration": candidate.iteration,
                    "node_id": candidate.node_id,
                    "redacted_public_summary": candidate.draft.redacted_summary,
                }
                for candidate in loop_result.selected_candidates
            ]

        candidate_ids: list[str] = []
        candidate_summaries: list[dict[str, Any]] = []
        for index, finalist in enumerate(finalists):
            candidate_kind = str(finalist.get("candidate_kind") or "image_build")
            if candidate_kind != "image_build":
                raise HostedResearchLabWorkerError("hosted auto-research produced a non-image-build candidate")
            candidate_patch_manifest = dict(finalist["candidate_patch_manifest"])
            hypothesis_doc = dict(finalist.get("hypothesis_doc") or {})
            patch_doc = dict(finalist.get("patch_doc") or {})
            redacted_summary = str(finalist.get("redacted_public_summary") or "")
            candidate_artifact_hash = str(candidate_patch_manifest["candidate_artifact_hash"])
            candidate_patch_hash = sha256_json(candidate_patch_manifest)
            request = ResearchLabCandidateArtifactCreateRequest(
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                receipt_id=context.receipt_id,
                miner_hotkey=str(context.ticket["miner_hotkey"]),
                island=str(context.ticket["island"]),
                candidate_kind=candidate_kind,
                private_model_manifest=final_artifact.to_dict(),
                candidate_patch_manifest=candidate_patch_manifest,
                candidate_model_manifest=finalist.get("candidate_model_manifest"),
                candidate_source_diff_hash=finalist.get("candidate_source_diff_hash"),
                candidate_build_doc={
                    **dict(finalist.get("candidate_build_doc") or {}),
                    # Worker-side annotation (deliberately outside the engine's
                    # build_doc_hash, which covers only engine-produced fields):
                    # carries the loop node id to the scoring worker so score
                    # bundles stamp the deterministic execution_trace:<uuid5>
                    # ref the trajectory projector writes (bundle→trace join).
                    "loop_node_id": str(finalist.get("node_id") or ""),
                },
                hypothesis_doc=hypothesis_doc,
                redacted_public_summary=redacted_summary,
            )
            candidate_row, _candidate_event = await self._store_write_with_retry(
                "candidate_artifact_create",
                lambda request=request: create_candidate_artifact(request),
            )
            duplicate_existing_candidate = bool(
                str(candidate_row.get("run_id") or "") and str(candidate_row.get("run_id") or "") != context.run_id
            )
            if duplicate_existing_candidate:
                await self._store_write_with_retry(
                    "duplicate_candidate_reused_loop_event",
                    lambda candidate_row=candidate_row, finalist=finalist: create_auto_research_loop_event(
                        run_id=context.run_id,
                        ticket_id=context.ticket_id,
                        receipt_id=context.receipt_id,
                        event_type="duplicate_candidate_reused",
                        loop_status="completed",
                        worker_ref=self.worker_ref,
                        node_id=str(finalist.get("node_id") or ""),
                        candidate_artifact_hash=candidate_artifact_hash,
                        candidate_patch_hash=candidate_patch_hash,
                        provider_usage=list(loop_result.provider_usage) or self._provider_usage(context),
                        cost_ledger=loop_result.cost_ledger(),
                        event_doc={
                            "candidate_id": str(candidate_row.get("candidate_id") or ""),
                            "existing_run_id": str(candidate_row.get("run_id") or ""),
                            "existing_ticket_id": str(candidate_row.get("ticket_id") or ""),
                            "current_run_id": context.run_id,
                            "current_ticket_id": context.ticket_id,
                            "candidate_artifact_hash": candidate_artifact_hash,
                            "candidate_patch_hash": candidate_patch_hash,
                            "candidate_source_diff_hash": finalist.get("candidate_source_diff_hash"),
                            "reimbursement_preserved": True,
                        },
                    ),
                )
            candidate_ids.append(str(candidate_row["candidate_id"]))
            candidate_summaries.append(
                {
                    "candidate_index": index,
                    "candidate_kind": candidate_kind,
                    "loop_iteration": finalist["iteration"],
                    "loop_node_id": finalist["node_id"],
                    "candidate_id": str(candidate_row["candidate_id"]),
                    "candidate_artifact_hash": candidate_artifact_hash,
                    "candidate_patch_hash": candidate_patch_hash,
                    "candidate_model_manifest_hash": (
                        (finalist.get("candidate_model_manifest") or {}).get("manifest_hash")
                        if isinstance(finalist.get("candidate_model_manifest"), Mapping)
                        else None
                    ),
                    "candidate_source_diff_hash": finalist.get("candidate_source_diff_hash"),
                    "duplicate_candidate_reused": duplicate_existing_candidate,
                    "hypothesis": hypothesis_doc,
                    "patch": patch_doc,
                    "parent_artifact_hash": final_artifact.model_artifact_hash,
                }
            )

        reimbursement_decision = await self._maybe_create_reimbursement_decision(
            context=context,
            budget_context=budget_context,
            loop_result=loop_result,
        )
        completion_receipt_doc = {
            "run_id": context.run_id,
            "worker_ref": self.worker_ref,
            "final_cost_ledger": loop_result.cost_ledger(),
            "provider_usage": list(loop_result.provider_usage) or self._provider_usage(context),
            "candidate_ids": list(candidate_ids),
            "reimbursement": reimbursement_decision or {"status": "not_written"},
        }

        completion_queue_doc = {
            "receipt_id": context.receipt_id,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "budget_context": _redacted_budget_context(budget_context),
            "auto_research_loop": {
                "iterations_completed": loop_result.iterations_completed,
                "elapsed_seconds": round(loop_result.elapsed_seconds, 3),
                "stop_reason": loop_result.stop_reason,
                "openrouter_call_count": loop_result.openrouter_call_count,
                "estimated_cost_usd": round(loop_result.estimated_cost_usd, 6),
                "actual_openrouter_cost_usd": round(loop_result.actual_openrouter_cost_usd, 6),
            },
            "reimbursement": reimbursement_decision or {"status": "not_written"},
            "next_stage": "gateway_qualification_worker_evaluation",
        }
        try:
            await self._store_write_with_retry(
                "completion_queue_event",
                lambda: create_queue_event(
                    run_id=context.run_id,
                    ticket_id=context.ticket_id,
                    event_type="completed",
                    queue_priority=int(context.queue_row.get("queue_priority") or 0),
                    worker_ref=self.worker_ref,
                    reason="candidate_generation_completed_evaluation_queued",
                    event_doc=completion_queue_doc,
                ),
            )
        except Exception:
            current_after_completion = await select_one(
                "research_loop_run_queue_current",
                columns="run_id,current_queue_status,current_event_hash,current_event_seq",
                filters=(("run_id", context.run_id),),
            )
            if not current_after_completion or current_after_completion.get("current_queue_status") != "completed":
                raise
            logger.warning(
                "research_lab_completion_queue_event_insert_uncertain_but_projected_completed run_id=%s event_hash=%s",
                compact_ref(context.run_id),
                current_after_completion.get("current_event_hash"),
            )
        await self._ensure_terminal_loop_projection(
            context,
            event_type="loop_completed",
            loop_status="completed",
            reason="candidate_generation_completed_evaluation_queued",
            event_doc=completion_queue_doc,
        )
        try:
            await self._store_write_with_retry(
                "completion_receipt_event",
                lambda: create_receipt_event(
                    receipt_id=str(context.receipt_id),
                    ticket_id=context.ticket_id,
                    event_type="completed",
                    receipt_status="completed",
                    event_doc=completion_receipt_doc,
                ),
            )
        except Exception as exc:
            logger.warning(
                "research_lab_completion_receipt_projection_failed run_id=%s receipt_id=%s error=%s",
                compact_ref(context.run_id),
                compact_ref(context.receipt_id),
                str(exc)[:240],
            )
        try:
            await create_ticket_event(
                ticket_id=context.ticket_id,
                event_type="running",
                actor_hotkey=None,
                reason="candidate_generation_completed_evaluation_queued",
                event_doc={
                    "run_id": context.run_id,
                    "receipt_id": context.receipt_id,
                    "candidate_ids": candidate_ids,
                    "auto_research_loop": {
                        "iterations_completed": loop_result.iterations_completed,
                        "elapsed_seconds": round(loop_result.elapsed_seconds, 3),
                        "stop_reason": loop_result.stop_reason,
                    },
                    "next_stage": "gateway_qualification_worker_evaluation",
                },
            )
            await safe_project_public_loop_activity(
                context.ticket_id,
                source_ref=f"hosted_worker_completed:{context.run_id}",
                reason="candidate_generation_completed_evaluation_queued",
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_completion_noncritical_projection_failed run_id=%s error=%s",
                compact_ref(context.run_id),
                str(exc)[:240],
            )
        logger.info(
            format_worker_block(
                "RESEARCH LAB AUTO-RESEARCH QUEUED CANDIDATES",
                (
                    ("Worker", self.worker_ref),
                    ("Run", compact_ref(context.run_id)),
                    ("Receipt", compact_ref(context.receipt_id)),
                    ("Candidates", len(candidate_ids)),
                    ("Iterations", loop_result.iterations_completed),
                    ("Elapsed", f"{loop_result.elapsed_seconds:.1f}s"),
                    ("Stop reason", loop_result.stop_reason),
                    ("Next stage", "gateway_qualification_worker_evaluation"),
                ),
            )
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="candidate_generation_completed_evaluation_queued",
            receipt_id=context.receipt_id,
            candidate_ids=tuple(candidate_ids),
        )

    async def _store_write_with_retry(self, label: str, operation: Any, *, attempts: int = 3) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                return await operation()
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts or not _is_retryable_worker_exception(exc):
                    raise
                logger.warning(
                    "research_lab_store_write_retry label=%s run_attempt=%s/%s error=%s",
                    label,
                    attempt,
                    attempts,
                    str(exc)[:240],
                )
                await asyncio.sleep(0.25 * attempt)
        raise RuntimeError(f"Research Lab store write failed after retries: {label}") from last_exc

    async def _reconcile_stale_loop_projections(self, *, limit: int = 200) -> int:
        """Finalize loop projections left ``running`` while their queue row is terminal.

        The loop ``*_current`` view is just the latest loop event; nothing forces it
        terminal when the run's queue row reaches completed/failed/cancelled. That
        decoupling produces zombie ``running`` loops. This sweep appends the matching
        terminal loop event so the projection reflects reality. Idempotent: once the
        terminal loop event is appended the row is no longer ``running`` and re-runs
        skip it.
        """
        loop_rows = await select_many(
            "research_lab_auto_research_loop_current",
            columns="run_id,ticket_id,current_loop_status",
            filters=(("current_loop_status", "running"),),
            limit=limit,
        )
        reconciled: list[str] = []
        for row in loop_rows:
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            queue = await select_one(
                "research_loop_run_queue_current",
                columns="run_id,ticket_id,current_queue_status",
                filters=(("run_id", run_id),),
            )
            if not queue:
                continue
            queue_status = str(queue.get("current_queue_status") or "")
            if queue_status not in {"completed", "failed", "cancelled", "tombstoned"}:
                continue
            ticket_id = str(queue.get("ticket_id") or row.get("ticket_id") or "")
            if not ticket_id:
                continue
            event_type = "loop_completed" if queue_status == "completed" else "loop_failed"
            loop_status = "completed" if queue_status == "completed" else "failed"
            try:
                await self._store_write_with_retry(
                    f"reconcile_loop_projection:{compact_ref(run_id)}",
                    lambda rid=run_id, tid=ticket_id, et=event_type, ls=loop_status, qs=queue_status: create_auto_research_loop_event(
                        run_id=rid,
                        ticket_id=tid,
                        event_type=et,
                        loop_status=ls,
                        worker_ref=self.worker_ref,
                        event_doc={
                            "schema_version": "1.0",
                            "reason": "stale_loop_projection_reconciled",
                            "source": "hosted_worker_loop_reconciler",
                            "queue_terminal_status": qs,
                            "previous_loop_status": "running",
                        },
                    ),
                )
                reconciled.append(run_id)
            except Exception as exc:
                logger.warning(
                    "research_lab_stale_loop_projection_reconcile_failed run_id=%s error=%s",
                    compact_ref(run_id),
                    str(exc)[:240],
                )
        if reconciled:
            # Alert (Item 6): queue/projection mismatch detected + repaired.
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB STALE LOOP PROJECTIONS RECONCILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Count", len(reconciled)),
                        ("Runs", ", ".join(compact_ref(r) for r in reconciled[:10])),
                    ),
                )
            )
            logger.warning(
                "research_lab_stale_loop_projection_reconciled worker_ref=%s count=%s",
                self.worker_ref,
                len(reconciled),
            )
        return len(reconciled)

    async def _ensure_terminal_loop_projection(
        self,
        context: HostedRunContext,
        *,
        event_type: str,
        loop_status: str,
        reason: str,
        event_doc: Mapping[str, Any] | None = None,
    ) -> None:
        current = await select_one(
            "research_lab_auto_research_loop_current",
            columns="run_id,current_loop_status,current_event_type,current_event_seq,current_event_hash",
            filters=(("run_id", context.run_id),),
        )
        if current and str(current.get("current_loop_status") or "") in {"completed", "failed"}:
            return
        try:
            provider_usage = self._provider_usage(context)
        except HostedResearchLabWorkerError:
            provider_usage = []
        projection_cost_ledger: Mapping[str, Any] | None = None
        projection_provider_usage: list[dict[str, Any]] | None = None
        if isinstance(event_doc, Mapping):
            maybe_ledger = event_doc.get("final_cost_ledger") or event_doc.get("cost_ledger")
            if isinstance(maybe_ledger, Mapping):
                projection_cost_ledger = maybe_ledger
            maybe_usage = event_doc.get("provider_usage")
            if isinstance(maybe_usage, Sequence) and not isinstance(maybe_usage, (str, bytes, bytearray)):
                projection_provider_usage = [dict(item) for item in maybe_usage if isinstance(item, Mapping)]
        projection_event_doc = {
            "schema_version": "1.0",
            "reason": reason,
            "source": "hosted_worker_terminal_queue_projection",
            "previous_loop_status": current.get("current_loop_status") if current else None,
            "previous_loop_event_type": current.get("current_event_type") if current else None,
            "previous_loop_event_seq": current.get("current_event_seq") if current else None,
            "previous_loop_event_hash": current.get("current_event_hash") if current else None,
            **dict(event_doc or {}),
        }
        try:
            # Retry transient store failures instead of silently dropping the terminal
            # projection (a dropped append leaves the loop stuck `running` forever).
            await self._store_write_with_retry(
                f"terminal_loop_projection:{event_type}",
                lambda: create_auto_research_loop_event(
                    run_id=context.run_id,
                    ticket_id=context.ticket_id,
                    receipt_id=context.receipt_id,
                    event_type=event_type,
                    loop_status=loop_status,
                    worker_ref=self.worker_ref,
                    provider_usage=projection_provider_usage or provider_usage,
                    cost_ledger=dict(projection_cost_ledger or {}),
                    event_doc=projection_event_doc,
                ),
            )
        except Exception as exc:
            after = await select_one(
                "research_lab_auto_research_loop_current",
                columns="run_id,current_loop_status,current_event_type,current_event_seq,current_event_hash",
                filters=(("run_id", context.run_id),),
            )
            if after and str(after.get("current_loop_status") or "") == loop_status:
                logger.info(
                    "research_lab_terminal_loop_projection_insert_uncertain_but_projected run_id=%s event_type=%s event_hash=%s",
                    compact_ref(context.run_id),
                    event_type,
                    after.get("current_event_hash"),
                )
                return
            logger.error(
                format_worker_block(
                    "RESEARCH LAB TERMINAL LOOP PROJECTION FAILED",
                    (
                        ("Run", compact_ref(context.run_id)),
                        ("Event", event_type),
                        ("Error", str(exc)[:300]),
                        ("Note", "loop may be left running; reconciler will finalize"),
                    ),
                )
            )
            logger.error(
                "research_lab_terminal_loop_projection_failed run_id=%s event_type=%s error=%s",
                compact_ref(context.run_id),
                event_type,
                str(exc)[:240],
            )

    async def _already_completed_outcome(self, context: HostedRunContext) -> HostedWorkerOutcome | None:
        current = await select_one(
            "research_loop_run_queue_current",
            filters=(("run_id", context.run_id),),
        )
        if current and str(current.get("current_queue_status") or "") == "completed":
            candidate_ids = await self._candidate_ids_for_run(context.run_id)
            receipt_id = await self._receipt_id_for_run(context.run_id)
            context.receipt_id = context.receipt_id or receipt_id
            await self._ensure_terminal_loop_projection(
                context,
                event_type="loop_completed",
                loop_status="completed",
                reason="queue_already_completed_terminal_projection",
                event_doc={"candidate_ids": candidate_ids, "receipt_id": receipt_id},
            )
            return HostedWorkerOutcome(
                processed=False,
                dry_run=False,
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                status="already_completed",
                receipt_id=receipt_id,
                candidate_ids=tuple(candidate_ids),
            )
        return None

    async def _complete_from_existing_completed_receipt(self, context: HostedRunContext) -> HostedWorkerOutcome | None:
        receipt_id = await self._completed_receipt_id_for_run(context.run_id)
        if not receipt_id:
            return None
        candidate_ids = await self._candidate_ids_for_run(context.run_id)
        event_doc = {
            "receipt_id": receipt_id,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "source": "existing_completed_receipt_after_recovery",
        }
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="completed",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="candidate_generation_completed_from_existing_receipt",
            event_doc=event_doc,
        )
        context.receipt_id = context.receipt_id or receipt_id
        await self._ensure_terminal_loop_projection(
            context,
            event_type="loop_completed",
            loop_status="completed",
            reason="candidate_generation_completed_from_existing_receipt",
            event_doc=event_doc,
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="candidate_generation_completed_from_existing_receipt",
            receipt_id=receipt_id,
            candidate_ids=tuple(candidate_ids),
        )

    async def _complete_from_existing_candidate_artifacts(self, context: HostedRunContext) -> HostedWorkerOutcome | None:
        candidate_ids = await self._candidate_ids_for_run(context.run_id)
        if not candidate_ids:
            return None
        receipt_id = context.receipt_id or await self._ensure_queued_receipt(context)
        context.receipt_id = receipt_id
        event_doc = {
            "receipt_id": receipt_id,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "source": "existing_candidate_artifacts_after_recovery",
            "next_stage": "gateway_qualification_worker_evaluation",
        }
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="completed",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="candidate_generation_completed_from_existing_artifacts",
            event_doc=event_doc,
        )
        await self._ensure_terminal_loop_projection(
            context,
            event_type="loop_completed",
            loop_status="completed",
            reason="candidate_generation_completed_from_existing_artifacts",
            event_doc=event_doc,
        )
        try:
            await create_receipt_event(
                receipt_id=receipt_id,
                ticket_id=context.ticket_id,
                event_type="completed",
                receipt_status="completed",
                event_doc=event_doc,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_existing_candidate_receipt_completion_failed run_id=%s receipt_id=%s error=%s",
                compact_ref(context.run_id),
                compact_ref(receipt_id),
                str(exc)[:240],
            )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="candidate_generation_completed_from_existing_artifacts",
            receipt_id=receipt_id,
            candidate_ids=tuple(candidate_ids),
        )

    async def _candidate_ids_for_run(self, run_id: str) -> list[str]:
        rows = await select_many(
            "research_lab_candidate_artifacts",
            columns="candidate_id",
            filters=(("run_id", run_id),),
            limit=max(10, int(self.config.hosted_worker_max_candidates or 1) * 5),
        )
        return [str(row["candidate_id"]) for row in rows if row.get("candidate_id")]

    async def _receipt_id_for_run(self, run_id: str) -> str | None:
        rows = await select_many(
            "research_loop_receipt_current",
            columns="receipt_id,current_receipt_status",
            filters=(("run_id", run_id),),
            order_by=(("current_status_at", True),),
            limit=10,
        )
        return str(rows[0]["receipt_id"]) if rows else None

    async def _completed_receipt_id_for_run(self, run_id: str) -> str | None:
        rows = await select_many(
            "research_loop_receipt_current",
            columns="receipt_id,current_receipt_status",
            filters=(("run_id", run_id),),
            order_by=(("current_status_at", True),),
            limit=10,
        )
        for row in rows:
            if str(row.get("current_receipt_status") or "") == "completed":
                return str(row["receipt_id"])
        return None

    async def _ensure_queued_receipt(self, context: HostedRunContext) -> str:
        if context.receipt_id:
            return context.receipt_id
        existing = await find_queued_receipt_for_run(context.run_id)
        if existing:
            return str(existing["receipt_id"])
        await self._resolve_loop_start_credit_id(context)
        receipt, _event = await create_receipt(self._queued_receipt_request(context))
        return str(receipt["receipt_id"])

    async def _mark_paused(
        self,
        context: HostedRunContext,
        *,
        loop_result: Any | None,
        checkpoint_doc: Mapping[str, Any] | None,
        reason: str,
    ) -> HostedWorkerOutcome:
        receipt_id = context.receipt_id or await self._ensure_queued_receipt(context)
        context.receipt_id = receipt_id
        cost_ledger = loop_result.cost_ledger() if loop_result is not None else {
            "schema_version": "1.0",
            "status": "paused",
            "total_usd": 0.0,
            "stage": reason,
        }
        checkpoint_ref = None
        if isinstance(checkpoint_doc, Mapping):
            checkpoint_ref = checkpoint_doc.get("checkpoint_hash")
        event_doc = {
            "run_id": context.run_id,
            "worker_ref": self.worker_ref,
            "receipt_id": receipt_id,
            "pause_reason": reason,
            "checkpoint_hash": checkpoint_ref,
            "auto_research_loop": {
                "status": "paused",
                "iterations_completed": int(getattr(loop_result, "iterations_completed", 0) or 0),
                "elapsed_seconds": round(float(getattr(loop_result, "elapsed_seconds", 0.0) or 0.0), 3),
                "stop_reason": getattr(loop_result, "stop_reason", "maintenance_pause_requested"),
            },
        }
        if checkpoint_doc:
            event_doc["checkpoint"] = dict(checkpoint_doc)
        await create_receipt_event(
            receipt_id=receipt_id,
            ticket_id=context.ticket_id,
            event_type="queued",
            receipt_status="queued",
            event_doc={
                **event_doc,
                "cost_ledger": cost_ledger,
                "provider_usage": list(getattr(loop_result, "provider_usage", ()) or []) or self._provider_usage(context),
            },
        )
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="paused",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason=reason,
            event_doc=event_doc,
        )
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="running",
            actor_hotkey=None,
            reason=reason,
            event_doc=event_doc,
        )
        await safe_project_public_loop_activity(
            context.ticket_id,
            source_ref=f"hosted_worker_paused:{context.run_id}",
            reason=reason,
            config=self.config,
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="maintenance_paused",
            receipt_id=receipt_id,
        )

    async def _mark_blocked_for_credit(self, context: HostedRunContext, error: str) -> HostedWorkerOutcome:
        """Pause a run that hit OpenRouter insufficient-credits as `blocked_for_credit`.

        Resumable (not terminal): receipt stays `queued`, queue gets a `paused` event
        with reason `blocked_for_credit`, ticket stays `running`. No loop-start payment
        is re-consumed; the run is revived only by the top-up resume path. Mirrors
        `_mark_paused` but carries the provider error + resume instructions.
        """
        receipt_id = context.receipt_id or await self._ensure_queued_receipt(context)
        context.receipt_id = receipt_id
        checkpoint_doc = await latest_auto_research_checkpoint(context.run_id)
        checkpoint_ref = checkpoint_doc.get("checkpoint_hash") if isinstance(checkpoint_doc, Mapping) else None
        model_id: str | None = None
        try:
            budget_context = self._run_budget_context(context)
            _tier, model_id, _doc = self.config.resolve_auto_research_model(
                str(budget_context.get("research_model_tier") or "")
            )
        except Exception:  # noqa: BLE001 - model id is best-effort metadata
            model_id = None
        miner_hotkey = ""
        if isinstance(context.ticket, Mapping):
            miner_hotkey = str(context.ticket.get("miner_hotkey") or "")
        event_doc = {
            "schema_version": "1.0",
            "run_id": context.run_id,
            "worker_ref": self.worker_ref,
            "receipt_id": receipt_id,
            "pause_reason": "blocked_for_credit",
            "blocked_for_credit": True,
            "provider_error": _redact_openrouter_diagnostic(error, limit=400),
            "checkpoint_hash": checkpoint_ref,
            "miner_hotkey": miner_hotkey,
            "model": model_id,
            "max_tokens": self.config.auto_research_max_tokens,
            "resume_mode": "resume_from_checkpoint" if checkpoint_ref else "restart_from_scratch",
            "resume_instructions": (
                "Top up the miner OpenRouter key, then resume via the miner CLI "
                "(/research-lab/resume-credit-blocked) or the operator "
                "resume_failed_runs_from_checkpoint; resume appends a queued event with "
                "reason credit_topup_resume. No loop-start payment is re-consumed."
            ),
        }
        await create_receipt_event(
            receipt_id=receipt_id,
            ticket_id=context.ticket_id,
            event_type="queued",
            receipt_status="queued",
            event_doc=event_doc,
        )
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="paused",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="blocked_for_credit",
            event_doc=event_doc,
        )
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="running",
            actor_hotkey=None,
            reason="blocked_for_credit",
            event_doc=event_doc,
        )
        await safe_project_public_loop_activity(
            context.ticket_id,
            source_ref=f"hosted_worker_blocked_for_credit:{context.run_id}",
            reason="blocked_for_credit",
            config=self.config,
        )
        logger.warning(
            "research_lab_run_blocked_for_credit run_id=%s ticket=%s checkpoint=%s",
            compact_ref(context.run_id),
            compact_ref(context.ticket_id),
            "yes" if checkpoint_ref else "no",
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="blocked_for_credit",
            receipt_id=receipt_id,
        )

    async def _preflight_openrouter_credit(
        self, context: HostedRunContext, provider_env: Mapping[str, str]
    ) -> None:
        """Best-effort credit gate before any expensive OpenRouter generation.

        Raises CreditBlockedHostedRunError when the resolved key reports
        `limit_remaining <= 0`. Preflight transport/auth errors are NOT treated as
        credit blocks (a real auth error surfaces during the call and fails normally);
        they're logged and skipped so a preflight blip never blocks a funded run.
        """
        raw_key = str(provider_env.get("OPENROUTER_API_KEY") or "")
        if not raw_key:
            return
        try:
            info = await asyncio.to_thread(preflight_openrouter_key, raw_key)
        except Exception as exc:  # noqa: BLE001 - preflight is best-effort
            logger.warning(
                "research_lab_openrouter_credit_preflight_skipped run_id=%s error=%s",
                compact_ref(context.run_id),
                str(exc)[:160],
            )
            return
        remaining = info.get("limit_remaining")
        if remaining is None:
            return
        try:
            depleted = float(remaining) <= 0.0
        except (TypeError, ValueError):
            return
        if depleted:
            raise CreditBlockedHostedRunError(
                f"OpenRouter key insufficient credits before generation (limit_remaining={remaining})"
            )

    def _validate_resume_state_freshness(
        self, resume_state: Any, artifact: Any, run_id: str
    ) -> Any:
        """Discard a checkpoint whose model no longer matches the active parent.

        A credit-blocked (or otherwise paused) run that resumes after the active model
        changed would resume against a stale parent. Per CTO policy, null the checkpoint
        so the run restarts from scratch against the current model (no payment re-consumed).
        """
        if not isinstance(resume_state, Mapping):
            return resume_state
        ckpt_artifact = str(resume_state.get("artifact_hash") or "")
        ckpt_manifest = str(resume_state.get("manifest_hash") or "")
        active_artifact = str(getattr(artifact, "model_artifact_hash", "") or "")
        active_manifest = str(getattr(artifact, "manifest_hash", "") or "")
        mismatch = (
            (ckpt_artifact and active_artifact and ckpt_artifact != active_artifact)
            or (ckpt_manifest and active_manifest and ckpt_manifest != active_manifest)
        )
        if mismatch:
            logger.warning(
                "research_lab_resume_checkpoint_stale_restart run_id=%s ckpt_artifact=%s active_artifact=%s",
                compact_ref(run_id),
                compact_ref(ckpt_artifact),
                compact_ref(active_artifact),
            )
            return None
        return resume_state

    async def _resolve_loop_start_credit_id(self, context: HostedRunContext) -> str | None:
        """Resolve the run's preserved loop-start credit ref, looking past the
        run-context event window when necessary.

        The credit ref is written on the earliest queued event; a much-requeued
        run's most-recent-20-events window no longer contains it, and treating
        that as "no preserved credit" wrongly denies reimbursement. Fall back to
        a targeted fetch of the run's queued events (heartbeats are `started`
        events, so this stays small) before concluding the ref is absent.
        """
        if context.loop_start_credit_id_resolved:
            return context.resolved_loop_start_credit_id
        credit_id = _loop_start_credit_id_from_queue_events(context.queue_events)
        if not credit_id and len(context.queue_events) >= _RUN_CONTEXT_QUEUE_EVENT_LIMIT:
            try:
                queued_events = await select_many(
                    "research_loop_run_queue_events",
                    columns="run_id,seq,event_type,event_doc",
                    filters=(("run_id", context.run_id), ("event_type", "queued")),
                    order_by=(("seq", False),),
                    limit=200,
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_loop_start_credit_history_fetch_failed run_id=%s error=%s",
                    compact_ref(context.run_id),
                    str(exc)[:240],
                )
                # Leave unresolved so a later call can retry the fetch.
                return None
            credit_id = _loop_start_credit_id_from_queue_events(queued_events)
        context.resolved_loop_start_credit_id = credit_id
        context.loop_start_credit_id_resolved = True
        return credit_id

    async def _maybe_create_reimbursement_decision(
        self,
        *,
        context: HostedRunContext,
        budget_context: Mapping[str, Any],
        loop_result: Any | None = None,
        cost_evidence: Mapping[str, Any] | None = None,
        source: str = "hosted_auto_research_loop_completion",
        failed_run_reimbursement: bool = False,
        failure_reason: str | None = None,
        queue_terminal_status: str | None = None,
        require_positive_cost: bool = False,
        skip_ineligible_prereqs: bool = False,
    ) -> dict[str, Any] | None:
        if not (self.config.reimbursements_enabled or self.config.shadow_reimbursements_enabled):
            return None
        evidence = dict(cost_evidence or {})
        if not evidence and loop_result is not None:
            evidence = cost_evidence_from_loop_result(loop_result)
        miner_key_ref = ""
        try:
            miner_key_ref = _miner_openrouter_key_ref(context)
        except HostedResearchLabWorkerError:
            miner_key_ref = ""
        decision = await create_reimbursement_decision(
            self.config,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            ticket=context.ticket,
            payment=context.payment,
            receipt_id=context.receipt_id,
            budget_context=budget_context,
            cost_evidence=evidence,
            source=source,
            failed_run_reimbursement=failed_run_reimbursement,
            failure_reason=failure_reason,
            queue_terminal_status=queue_terminal_status,
            actor_ref=self.worker_ref,
            miner_openrouter_key_ref=miner_key_ref,
            preserved_loop_start_credit=bool(await self._resolve_loop_start_credit_id(context)),
            require_positive_cost=require_positive_cost,
            skip_ineligible_prereqs=skip_ineligible_prereqs,
        )
        if decision and "award_id" in decision:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB REIMBURSEMENT DECISION",
                    (
                        ("Run", compact_ref(context.run_id)),
                        ("Status", decision.get("status")),
                        ("Target USD", f"{float(decision.get('target_reimbursement_usd') or 0.0):.6f}"),
                        ("OpenRouter USD", f"{float(decision.get('actual_openrouter_cost_usd') or 0.0):.6f}"),
                        ("Source", source),
                        ("Shadow only", decision.get("shadow_only")),
                    ),
                )
            )
        return decision

    async def _create_participation_snapshot(
        self,
        context: HostedRunContext,
        policy: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        island = str(context.ticket.get("island") or self.config.reimbursement_default_island)
        lookback_end = datetime.now(timezone.utc)
        lookback_start = lookback_end - timedelta(days=7)
        ticket_rows = await select_all(
            "research_loop_ticket_current",
            filters=(("island", island),),
            order_by=(("created_at", True),),
        )
        ticket_rows = [
            row
            for row in ticket_rows
            if _row_dt(row.get("created_at") or row.get("current_status_at")) >= lookback_start
        ]
        ticket_ids = {str(row.get("ticket_id")) for row in ticket_rows}
        queue_rows: list[dict[str, Any]] = []
        for ticket_id in ticket_ids:
            if not ticket_id:
                continue
            queue_rows.extend(
                await select_all(
                    "research_loop_run_queue_current",
                    filters=(("ticket_id", ticket_id),),
                    order_by=(("current_status_at", True),),
                    max_rows=100,
                )
            )
        funded_queue_rows = [
            row
            for row in queue_rows
            if str(row.get("ticket_id")) in ticket_ids
            and str(row.get("current_queue_status")) in {"queued", "started", "paused", "completed"}
            and _row_dt(row.get("current_status_at")) >= lookback_start
        ]
        distinct_hotkeys = {str(row.get("miner_hotkey")) for row in ticket_rows if row.get("miner_hotkey")}
        brief_refs = {str(row.get("brief_sanitized_ref")) for row in ticket_rows if row.get("brief_sanitized_ref")}
        snapshot_doc = {
            "snapshot_id": f"participation:{island}:{lookback_end.date().isoformat()}",
            "island": island,
            "lookback_start": lookback_start.isoformat(),
            "lookback_end": lookback_end.isoformat(),
            "distinct_funded_hotkeys": len(distinct_hotkeys),
            "paid_loop_count": len(funded_queue_rows),
            "unique_brief_count": len(brief_refs),
        }
        participation_score = compute_participation_score(snapshot_doc, policy)
        snapshot_row = await create_participation_snapshot(
            island=island,
            lookback_start=snapshot_doc["lookback_start"],
            lookback_end=snapshot_doc["lookback_end"],
            distinct_funded_hotkeys=snapshot_doc["distinct_funded_hotkeys"],
            paid_loop_count=snapshot_doc["paid_loop_count"],
            unique_brief_count=snapshot_doc["unique_brief_count"],
            participation_score=float(participation_score),
            policy_id=str(policy["policy_id"]),
            snapshot_doc={
                **snapshot_doc,
                "participation_score": float(participation_score),
                "source": "research_loop_ticket_current_and_run_queue_current",
                "postgrest_limit": 1000,
            },
        )
        return snapshot_doc, snapshot_row

    async def _reimbursement_cap_usage(self, context: HostedRunContext, *, run_day: str) -> dict[str, float]:
        rows = await select_all(
            "research_reimbursement_award_current",
            filters=(
                ("current_award_status", "awarded"),
                ("run_day", run_day),
            ),
        )
        miner_hotkey = str(context.ticket["miner_hotkey"])
        island = str(context.ticket["island"] or self.config.reimbursement_default_island)
        eligible_rows = [
            row
            for row in rows
            if str(row.get("run_day")) == run_day
            and str(row.get("current_award_status") or row.get("award_status")) == "awarded"
        ]
        return {
            "hotkey_day_awarded_usd": _sum_award_usd(row for row in eligible_rows if str(row.get("miner_hotkey")) == miner_hotkey),
            "island_day_awarded_usd": _sum_award_usd(row for row in eligible_rows if str(row.get("island")) == island),
            "global_awarded_usd": _sum_award_usd(eligible_rows),
        }

    async def _append_started_events(self, context: HostedRunContext) -> None:
        current = await select_one(
            "research_loop_run_queue_current",
            filters=(("run_id", context.run_id),),
        )
        if not current or current.get("current_queue_status") != "queued":
            raise HostedResearchLabClaimLost("queued Research Lab run is no longer queued")
        try:
            started_event = await create_queue_event(
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                event_type="started",
                queue_priority=int(context.queue_row.get("queue_priority") or 0),
                worker_ref=self.worker_ref,
                reason="hosted_worker_started",
                event_doc={"worker_ref": self.worker_ref},
            )
        except Exception as exc:
            if _is_claim_race_error(exc):
                raise HostedResearchLabClaimLost("queued Research Lab run was claimed by another worker") from exc
            raise
        current_after = await select_one(
            "research_loop_run_queue_current",
            columns="run_id,current_queue_status,worker_ref,current_event_hash,current_event_seq",
            filters=(("run_id", context.run_id),),
        )
        if (
            not current_after
            or current_after.get("current_queue_status") != "started"
            or current_after.get("worker_ref") != self.worker_ref
            or current_after.get("current_event_hash") != started_event.get("anchored_hash")
        ):
            raise HostedResearchLabClaimLost("queued Research Lab run claim was superseded before execution")
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="running",
            actor_hotkey=None,
            reason="hosted_worker_started",
            event_doc={"run_id": context.run_id, "worker_ref": self.worker_ref},
        )
        await safe_project_public_loop_activity(
            context.ticket_id,
            source_ref=f"hosted_worker_started:{context.run_id}",
            reason="hosted_worker_started",
            config=self.config,
        )

    async def _append_queue_heartbeat(
        self,
        context: HostedRunContext,
        *,
        source_event_type: str,
        source_event_seq: object,
        source_event_hash: object,
    ) -> bool:
        current = await select_one(
            "research_loop_run_queue_current",
            columns="run_id,current_queue_status,worker_ref,current_event_hash,current_event_seq,queue_priority",
            filters=(("run_id", context.run_id),),
        )
        if (
            current
            and current.get("current_queue_status") == "started"
            and current.get("worker_ref")
            and current.get("worker_ref") != self.worker_ref
        ):
            # Another worker re-claimed this run (stale requeue steal). Abort
            # promptly so this superseded worker stops charging the miner's
            # OpenRouter key; the new claimant owns the run now, so no
            # terminal event is written on this path.
            context.claim_lost = True
            raise HostedResearchLabClaimLost(
                "hosted run claim now owned by another worker "
                f"(current worker_ref={current.get('worker_ref')})"
            )
        if (
            not current
            or current.get("current_queue_status") != "started"
            or current.get("worker_ref") != self.worker_ref
        ):
            return False
        try:
            await create_queue_event(
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                event_type="started",
                queue_priority=int(current.get("queue_priority") or context.queue_row.get("queue_priority") or 0),
                worker_ref=self.worker_ref,
                reason="hosted_worker_heartbeat",
                event_doc={
                    "worker_ref": self.worker_ref,
                    "source_event_type": source_event_type,
                    "source_event_seq": source_event_seq,
                    "source_event_hash": source_event_hash,
                    "previous_queue_event_hash": current.get("current_event_hash"),
                    "previous_queue_event_seq": current.get("current_event_seq"),
                },
            )
            return True
        except Exception as exc:
            if _is_claim_race_error(exc) and _heartbeat_conflict_claim_lost_enabled():
                # The DB claim guard fenced this heartbeat: the run was
                # re-claimed, requeued, or paused between the read above and
                # the insert. Treat as claim lost (see the flag docstring for
                # why this is gated on the scripts/59 guard being applied).
                context.claim_lost = True
                raise HostedResearchLabClaimLost(
                    "hosted run queue heartbeat was rejected by the claim guard"
                ) from exc
            logger.warning(
                "research_lab_hosted_queue_heartbeat_failed run_id=%s worker_ref=%s error=%s",
                compact_ref(context.run_id),
                self.worker_ref,
                str(exc)[:240],
            )
            return False

    async def _to_thread_with_queue_heartbeat(
        self,
        context: HostedRunContext,
        *,
        heartbeat_label: str,
        func: Any,
    ) -> Any:
        await self._append_queue_heartbeat(
            context,
            source_event_type=f"{heartbeat_label}_started",
            source_event_seq=None,
            source_event_hash=None,
        )
        task = asyncio.create_task(asyncio.to_thread(func))
        heartbeat_index = 0
        while not task.done():
            done, _pending = await asyncio.wait({task}, timeout=30.0)
            if done:
                break
            heartbeat_index += 1
            try:
                await self._append_queue_heartbeat(
                    context,
                    source_event_type=f"{heartbeat_label}_heartbeat_{heartbeat_index}",
                    source_event_seq=None,
                    source_event_hash=None,
                )
            except HostedResearchLabClaimLost:
                # Abort the run promptly; the worker thread itself cannot be
                # cancelled, so detach it and discard its eventual result.
                task.add_done_callback(_discard_backgrounded_task_result)
                raise
        return await task

    async def _mark_failed(
        self,
        context: HostedRunContext,
        error: str,
        *,
        loop_result: Any | None = None,
        failure_exception: BaseException | None = None,
        reason: str = "hosted_research_lab_run_failed",
    ) -> HostedWorkerOutcome:
        if loop_result is not None:
            cost_evidence = cost_evidence_from_loop_result(loop_result)
        elif context.latest_loop_cost_evidence:
            cost_evidence = dict(context.latest_loop_cost_evidence)
        else:
            try:
                cost_evidence = await latest_reimbursable_loop_cost_evidence(context.run_id)
            except Exception as exc:  # noqa: BLE001 - terminal failure must still project
                logger.warning(
                    "research_lab_failed_run_cost_evidence_read_failed run_id=%s error=%s",
                    compact_ref(context.run_id),
                    str(exc)[:240],
                )
                cost_evidence = {}
        cost_evidence = _merge_failure_exception_cost_evidence(cost_evidence, failure_exception)
        cost_ledger = cost_evidence_cost_ledger(cost_evidence)
        provider_usage = cost_evidence_provider_usage(cost_evidence)
        receipt_id = context.receipt_id
        if not receipt_id:
            await self._resolve_loop_start_credit_id(context)
            receipt, _event = await create_receipt(
                self._failed_receipt_request(context, error, cost_evidence=cost_evidence)
            )
            receipt_id = str(receipt["receipt_id"])
        context.receipt_id = context.receipt_id or receipt_id

        budget_context = self._run_budget_context(context)
        try:
            reimbursement_decision = await self._maybe_create_reimbursement_decision(
                context=context,
                budget_context=budget_context,
                cost_evidence=cost_evidence,
                source="hosted_auto_research_loop_failed",
                failed_run_reimbursement=True,
                failure_reason=error,
                queue_terminal_status="failed",
                require_positive_cost=True,
                skip_ineligible_prereqs=True,
            )
        except Exception as exc:  # noqa: BLE001 - terminal failure must not get stuck
            reimbursement_decision = {
                "status": "reimbursement_write_failed",
                "error": str(exc)[:300],
                "failed_run_reimbursement": True,
                "source": "hosted_auto_research_loop_failed",
            }
            logger.warning(
                "research_lab_failed_run_reimbursement_write_failed run_id=%s error=%s",
                compact_ref(context.run_id),
                str(exc)[:240],
            )

        event_doc = {
            "schema_version": "1.0",
            "run_id": context.run_id,
            "worker_ref": self.worker_ref,
            "error": error[:500],
            "failure_reason": reason,
            "receipt_id": receipt_id,
            "reimbursement": reimbursement_decision or {"status": "not_written"},
        }
        if cost_ledger:
            event_doc["final_cost_ledger"] = cost_ledger
        if provider_usage:
            event_doc["provider_usage"] = provider_usage
        try:
            await create_receipt_event(
                receipt_id=receipt_id,
                ticket_id=context.ticket_id,
                event_type="failed",
                receipt_status="failed",
                event_doc=event_doc,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_failed_receipt_event_projection_failed run_id=%s receipt_id=%s error=%s",
                compact_ref(context.run_id),
                compact_ref(receipt_id),
                str(exc)[:240],
            )
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="failed",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason=reason,
            event_doc=event_doc,
        )
        await self._ensure_terminal_loop_projection(
            context,
            event_type="loop_failed",
            loop_status="failed",
            reason=reason,
            event_doc=event_doc,
        )
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="cancelled",
            actor_hotkey=None,
            reason=reason,
            event_doc=event_doc,
        )
        await safe_project_public_loop_activity(
            context.ticket_id,
            source_ref=f"hosted_worker_failed:{context.run_id}",
            reason=reason,
            config=self.config,
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="failed",
            receipt_id=receipt_id,
            error=error[:500],
        )

    async def _mark_builder_not_ready(self, context: HostedRunContext, error: str) -> HostedWorkerOutcome:
        event_doc = {
            **autoresearch_queue_capacity_doc(self.config),
            "schema_version": "1.0",
            "run_id": context.run_id,
            "worker_ref": self.worker_ref,
            "recovering_worker_ref": self.worker_ref,
            "retrying_worker_ref": self.worker_ref,
            "builder_not_ready_error": error[:500],
            "previous_event_hash": context.queue_row.get("current_event_hash"),
            "previous_status_at": context.queue_row.get("current_status_at"),
        }
        if context.receipt_id:
            try:
                await create_receipt_event(
                    receipt_id=context.receipt_id,
                    ticket_id=context.ticket_id,
                    event_type="queued",
                    receipt_status="queued",
                    event_doc=event_doc,
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_builder_not_ready_receipt_event_failed run_id=%s receipt_id=%s error=%s",
                    compact_ref(context.run_id),
                    compact_ref(context.receipt_id),
                    str(exc)[:240],
                )
        try:
            await create_queue_event(
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                event_type="queued",
                queue_priority=int(context.queue_row.get("queue_priority") or 0),
                worker_ref=self.worker_ref,
                reason="code_edit_builder_not_ready_requeued",
                event_doc=event_doc,
            )
        except Exception as exc:
            if _is_queue_capacity_conflict_error(exc):
                return await self._park_requeue_conflict(
                    context,
                    error,
                    requeue_reason="code_edit_builder_not_ready_requeued",
                    conflict_error=exc,
                    base_event_doc=event_doc,
                )
            raise
        try:
            await create_ticket_event(
                ticket_id=context.ticket_id,
                event_type="running",
                actor_hotkey=None,
                reason="code_edit_builder_not_ready_requeued",
                event_doc=event_doc,
            )
            await safe_project_public_loop_activity(
                context.ticket_id,
                source_ref=f"hosted_worker_builder_not_ready_requeued:{context.run_id}",
                reason="code_edit_builder_not_ready_requeued",
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_builder_not_ready_noncritical_projection_failed run_id=%s error=%s",
                compact_ref(context.run_id),
                str(exc)[:240],
            )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="code_edit_builder_not_ready_requeued",
            receipt_id=context.receipt_id,
            error=error[:500],
        )

    async def _mark_retryable(
        self,
        context: HostedRunContext,
        error: str,
        *,
        retry_count: int,
    ) -> HostedWorkerOutcome:
        event_doc = {
            **autoresearch_queue_capacity_doc(self.config),
            "schema_version": "1.0",
            "run_id": context.run_id,
            "worker_ref": self.worker_ref,
            "recovering_worker_ref": self.worker_ref,
            "retrying_worker_ref": self.worker_ref,
            "retryable_error": error[:500],
            "retryable_error_count": int(retry_count),
            "retryable_failure_limit": int(self.config.hosted_worker_retryable_failure_limit),
            "previous_event_hash": context.queue_row.get("current_event_hash"),
            "previous_status_at": context.queue_row.get("current_status_at"),
        }
        if context.receipt_id:
            try:
                await create_receipt_event(
                    receipt_id=context.receipt_id,
                    ticket_id=context.ticket_id,
                    event_type="queued",
                    receipt_status="queued",
                    event_doc=event_doc,
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_retryable_receipt_event_failed run_id=%s receipt_id=%s error=%s",
                    compact_ref(context.run_id),
                    compact_ref(context.receipt_id),
                    str(exc)[:240],
                )
        try:
            await create_queue_event(
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                event_type="queued",
                queue_priority=int(context.queue_row.get("queue_priority") or 0),
                worker_ref=self.worker_ref,
                reason="transient_worker_error_requeued",
                event_doc=event_doc,
            )
        except Exception as exc:
            if _is_queue_capacity_conflict_error(exc):
                return await self._park_requeue_conflict(
                    context,
                    error,
                    requeue_reason="transient_worker_error_requeued",
                    conflict_error=exc,
                    base_event_doc=event_doc,
                )
            raise
        try:
            await create_ticket_event(
                ticket_id=context.ticket_id,
                event_type="running",
                actor_hotkey=None,
                reason="transient_worker_error_requeued",
                event_doc=event_doc,
            )
            await safe_project_public_loop_activity(
                context.ticket_id,
                source_ref=f"hosted_worker_retryable_requeued:{context.run_id}",
                reason="transient_worker_error_requeued",
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_retryable_noncritical_projection_failed run_id=%s error=%s",
                compact_ref(context.run_id),
                str(exc)[:240],
            )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="transient_worker_error_requeued",
            receipt_id=context.receipt_id,
            error=error[:500],
        )

    async def _park_requeue_conflict(
        self,
        context: HostedRunContext,
        error: str,
        *,
        requeue_reason: str,
        conflict_error: BaseException,
        base_event_doc: Mapping[str, Any],
    ) -> HostedWorkerOutcome:
        """Park a run whose requeue was rejected by the queue capacity/hotkey guard.

        The guard conflict can outlive every retry (e.g. the miner keeps another
        active run), and letting it escape the failure handler would leave this
        run wedged `started` forever. A `paused` event is not capacity-guarded
        and stays recoverable: the stale-paused reaper (or a maintenance resume)
        requeues it once capacity frees up.
        """
        event_doc = {
            **dict(base_event_doc),
            "pause_reason": "requeue_capacity_conflict_parked",
            "requeue_conflict_reason": requeue_reason,
            "requeue_conflict_error": str(conflict_error)[:300],
        }
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="paused",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="requeue_capacity_conflict_parked",
            event_doc=event_doc,
        )
        try:
            await create_ticket_event(
                ticket_id=context.ticket_id,
                event_type="running",
                actor_hotkey=None,
                reason="requeue_capacity_conflict_parked",
                event_doc=event_doc,
            )
            await safe_project_public_loop_activity(
                context.ticket_id,
                source_ref=f"hosted_worker_requeue_conflict_parked:{context.run_id}",
                reason="requeue_capacity_conflict_parked",
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_requeue_conflict_parked_projection_failed run_id=%s error=%s",
                compact_ref(context.run_id),
                str(exc)[:240],
            )
        logger.warning(
            "research_lab_requeue_conflict_parked run_id=%s requeue_reason=%s error=%s",
            compact_ref(context.run_id),
            requeue_reason,
            str(conflict_error)[:240],
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="requeue_capacity_conflict_parked",
            receipt_id=context.receipt_id,
            error=error[:500],
        )

    def _queued_receipt_request(self, context: HostedRunContext) -> ResearchLabReceiptCreateRequest:
        budget_context = self._run_budget_context(context)
        return ResearchLabReceiptCreateRequest(
            internal_run_ref=f"research_lab_hosted_worker:{context.run_id}",
            ticket_id=context.ticket_id,
            run_id=context.run_id,
            loop_start_payment_id=context.payment.get("payment_id") if context.payment else None,
            miner_hotkey=str(context.ticket["miner_hotkey"]),
            island=str(context.ticket["island"]),
            receipt_status="queued",
            loop_count=int(context.ticket.get("requested_loop_count") or 1),
            loop_start_credit_id=_context_loop_start_credit_id(context),
            miner_openrouter_key_ref=_miner_openrouter_key_ref(context),
            provider_usage=self._provider_usage(context),
            cost_ledger={
                "schema_version": "1.0",
                "status": "queued",
                "total_usd": 0.0,
                "budget_context": _redacted_budget_context(budget_context),
            },
            receipt_doc={
                "schema_version": "1.0",
                "run_id": context.run_id,
                "worker_ref": self.worker_ref,
                "private_model_manifest_uri": self.config.private_model_manifest_uri,
                "budget_context": _redacted_budget_context(budget_context),
            },
        )

    def _failed_receipt_request(
        self,
        context: HostedRunContext,
        error: str,
        *,
        cost_evidence: Mapping[str, Any] | None = None,
    ) -> ResearchLabReceiptCreateRequest:
        budget_context = self._run_budget_context(context)
        cost_ledger = cost_evidence_cost_ledger(cost_evidence)
        if not cost_ledger:
            cost_ledger = {
                "schema_version": "1.0",
                "status": "failed",
                "total_usd": 0.0,
                "budget_context": _redacted_budget_context(budget_context),
            }
        try:
            key_ref = _miner_openrouter_key_ref(context)
        except HostedResearchLabWorkerError:
            key_ref = None
        try:
            provider_usage = cost_evidence_provider_usage(cost_evidence) or self._provider_usage(context)
        except HostedResearchLabWorkerError:
            provider_usage = cost_evidence_provider_usage(cost_evidence)
        return ResearchLabReceiptCreateRequest(
            internal_run_ref=f"research_lab_hosted_worker:{context.run_id}",
            ticket_id=context.ticket_id,
            run_id=context.run_id,
            loop_start_payment_id=context.payment.get("payment_id") if context.payment else None,
            miner_hotkey=str(context.ticket["miner_hotkey"]),
            island=str(context.ticket["island"]),
            receipt_status="failed",
            loop_count=int(context.ticket.get("requested_loop_count") or 1),
            loop_start_credit_id=_context_loop_start_credit_id(context),
            miner_openrouter_key_ref=key_ref,
            provider_usage=provider_usage,
            cost_ledger=cost_ledger,
            receipt_doc={
                "schema_version": "1.0",
                "run_id": context.run_id,
                "worker_ref": self.worker_ref,
                "failure_reason": error[:500],
                "budget_context": _redacted_budget_context(budget_context),
                "reimbursement": {
                    "status": "pending_failed_run_reimbursement_decision",
                    "failed_run_reimbursement": True,
                },
            },
        )

    def _provider_usage(self, context: HostedRunContext) -> list[dict[str, Any]]:
        budget_context = self._run_budget_context(context)
        return [
            {
                "provider": "openrouter",
                "key_source": "miner_key_ref",
                "key_ref": _miner_openrouter_key_ref(context),
                "research_model_tier": budget_context.get("research_model_tier"),
            },
            {"provider": "exa", "key_source": "leadpoet_server_side"},
            {"provider": "scrapingdog", "key_source": "leadpoet_server_side"},
        ]

    async def _call_openrouter(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        api_key: str,
        model_id: str,
        reasoning_effort: str = "",
        timeout_seconds: int = 90,
        max_tokens: int = 1800,
        temperature: float | None = None,
        allow_non_zdr: bool = False,
        capture_run_id: str = "",
        capture_stage: str = "",
        privacy_key_ref: str = "",
        privacy_miner_hotkey: str = "",
        privacy_management_key: str = "",
    ) -> OpenRouterCallResult:
        if not api_key:
            raise HostedResearchLabWorkerError("OpenRouter key is required for hosted auto-research")
        if not model_id:
            raise HostedResearchLabWorkerError("OpenRouter auto-research model is required")
        if not (privacy_key_ref and privacy_miner_hotkey and privacy_management_key):
            raise HostedResearchLabWorkerError(
                "OpenRouter workspace privacy proof context is required for hosted auto-research"
            )
        base_max_tokens = max(1, int(max_tokens or 0))
        requested_reasoning_effort = str(reasoning_effort or "").strip()
        request_temperature = min(
            2.0,
            max(
                0.0,
                float(self.config.auto_research_temperature if temperature is None else temperature),
            ),
        )

        def _request_body(effective_max_tokens: int, *, include_reasoning_effort: bool) -> dict[str, Any]:
            body = {
                "model": model_id,
                "messages": list(messages),
                "temperature": request_temperature,
                "max_tokens": int(effective_max_tokens),
                "response_format": {"type": "json_object"},
            }
            body["provider"] = strict_openrouter_provider_policy()
            if requested_reasoning_effort and include_reasoning_effort:
                body["reasoning_effort"] = requested_reasoning_effort
                body["reasoning"] = {"effort": requested_reasoning_effort}
                body["include_reasoning"] = True
            elif _llm_include_reasoning_enabled() and include_reasoning_effort:
                # Optional reasoning capture for operators that explicitly
                # enable it. Keep it off by default so non-reasoning planner
                # models are not forced into empty/filtered responses.
                body["include_reasoning"] = True
            return body

        proxy_opener = _worker_llm_proxy_opener(self.config)
        open_fn = proxy_opener.open if proxy_opener else urlrequest.urlopen

        def _record_privacy_proof() -> None:
            try:
                proof_doc = verify_openrouter_workspace_privacy(
                    runtime_key=api_key,
                    management_key=privacy_management_key,
                    stage=capture_stage or "openrouter_call",
                    request_policy=strict_openrouter_provider_policy(),
                )
                create_openrouter_privacy_proof_event_sync(
                    key_ref=privacy_key_ref,
                    miner_hotkey=privacy_miner_hotkey,
                    run_id=capture_run_id or None,
                    stage=capture_stage or "openrouter_call",
                    proof_status="passed",
                    proof_doc=proof_doc,
                )
            except Exception as exc:
                failure_doc = {
                    "source": "openrouter_workspace_privacy_guard",
                    "stage": capture_stage or "openrouter_call",
                    "error": exc.__class__.__name__,
                    "request_policy": strict_openrouter_provider_policy(),
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }
                failure_doc["proof_hash"] = canonical_hash(failure_doc)
                try:
                    create_openrouter_privacy_proof_event_sync(
                        key_ref=privacy_key_ref,
                        miner_hotkey=privacy_miner_hotkey,
                        run_id=capture_run_id or None,
                        stage=capture_stage or "openrouter_call",
                        proof_status="failed",
                        proof_doc=failure_doc,
                    )
                except Exception:
                    logger.warning(
                        "research_lab_openrouter_privacy_proof_failed_to_record",
                        extra={
                            "run_id": capture_run_id,
                            "stage": capture_stage,
                            "key_ref": privacy_key_ref,
                        },
                    )
                raise HostedResearchLabWorkerError(
                    "OpenRouter workspace privacy verification failed before hidden prompt"
                ) from exc

        def _call_once(*, effective_max_tokens: int, include_reasoning_effort: bool) -> OpenRouterCallResult:
            body = _request_body(
                effective_max_tokens,
                include_reasoning_effort=include_reasoning_effort,
            )

            def _record_raw_trace(response_doc: Any, *, outcome: str) -> dict[str, str] | None:
                # §9.1 item 5: best-effort encrypted raw prompt/response
                # capture — every attempt (success, HTTP error, length stop) is
                # training signal. Never raises and never blocks: the recorder
                # wraps itself and uploads on a background pool. Authorization
                # headers / api keys are deliberately never part of the doc.
                return self._raw_trace_recorder.capture(
                    run_id=capture_run_id,
                    stage=capture_stage,
                    request_doc={
                        "url": _OPENROUTER_CHAT_COMPLETIONS_URL,
                        "method": "POST",
                        "body": body,
                    },
                    response_doc=response_doc,
                    outcome=outcome,
                )

            _record_privacy_proof()
            req = urlrequest.Request(
                _OPENROUTER_CHAT_COMPLETIONS_URL,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with open_fn(req, timeout=int(timeout_seconds)) as response:
                    response_status = getattr(response, "status", None) or getattr(response, "code", None)
                    response_text = response.read().decode("utf-8", errors="replace")
                    try:
                        decoded = json.loads(response_text)
                    except json.JSONDecodeError as exc:
                        excerpt = response_text[:500]
                        trace_doc = {"error_excerpt": excerpt}
                        if response_status is not None:
                            trace_doc["http_status"] = response_status
                        _record_raw_trace(trace_doc, outcome="invalid_json_response")
                        error = (
                            "OpenRouter candidate generation failed: non-JSON response"
                            f" status={response_status or 'unknown'} excerpt={excerpt[:200]}"
                        )
                        raise RetryableHostedResearchLabWorkerError(error) from exc
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")[:500]
                _record_raw_trace(
                    {"http_status": int(exc.code), "error_excerpt": message},
                    outcome="http_error",
                )
                error = f"OpenRouter candidate generation failed: HTTP {exc.code}: {message}"
                if (
                    include_reasoning_effort
                    and (requested_reasoning_effort or _llm_include_reasoning_enabled())
                    and _is_openrouter_reasoning_effort_unsupported(int(exc.code), message)
                ):
                    raise OpenRouterReasoningEffortUnsupportedError(error) from exc
                if int(exc.code) == 402 or _is_openrouter_credit_block(None, error):
                    raise CreditBlockedHostedRunError(error) from exc
                if int(exc.code) in _RETRYABLE_HTTP_CODES:
                    raise RetryableHostedResearchLabWorkerError(error) from exc
                raise HostedResearchLabWorkerError(error) from exc
            except URLError as exc:
                _record_raw_trace({"error_excerpt": str(exc)[:500]}, outcome="transport_error")
                raise RetryableHostedResearchLabWorkerError(
                    f"OpenRouter candidate generation failed: {exc}"
                ) from exc
            raw_trace_ref = _record_raw_trace(decoded, outcome="response")
            choices = decoded.get("choices") if isinstance(decoded, Mapping) else None
            if not isinstance(choices, list) or not choices:
                _raise_openrouter_generation_response_error(
                    decoded,
                    failure="no candidate-generation choices",
                    default_retryable=True,
                )
            first_choice = choices[0] if isinstance(choices[0], Mapping) else {}
            message = first_choice.get("message") if isinstance(first_choice.get("message"), Mapping) else {}
            content = message.get("content")
            usage = decoded.get("usage") if isinstance(decoded.get("usage"), Mapping) else {}
            provider_usage, cost_microusd = _build_openrouter_provider_usage(
                decoded=decoded,
                usage=usage,
                model_id=model_id,
                api_key=api_key,
                generation_stats_opener=proxy_opener.open if proxy_opener else None,
                reasoning_requested=include_reasoning_effort,
                requested_reasoning_effort=requested_reasoning_effort,
            )
            if raw_trace_ref:
                # Pointer only ({s3_ref, sha256}) — never raw prompt/response
                # text: the trajectory-corpus protected-material scanner fails
                # any event doc carrying content keys.
                provider_usage = {**provider_usage, "raw_trace_ref": dict(raw_trace_ref)}
                reasoning_capture = dict(
                    provider_usage.get("reasoning_capture")
                    if isinstance(provider_usage.get("reasoning_capture"), Mapping)
                    else {}
                )
                reasoning_capture["raw_trace_ref_present"] = True
                reasoning_capture["storage_state"] = "raw_trace_ref"
                provider_usage["reasoning_capture"] = reasoning_capture
            if not content:
                _raise_openrouter_generation_response_error(
                    decoded,
                    failure="empty candidate-generation content",
                    default_retryable=True,
                    provider_usage=provider_usage,
                    cost_microusd=cost_microusd,
                )
            if _openrouter_generation_stopped_for_length(decoded):
                raise OpenRouterLengthRetryableError(
                    f"OpenRouter candidate generation stopped at output token cap: {_openrouter_response_summary(decoded)}",
                    provider_usage=provider_usage,
                    cost_microusd=cost_microusd,
                )
            return OpenRouterCallResult(
                content=str(content),
                provider_usage=provider_usage,
                cost_microusd=cost_microusd,
            )

        def _call() -> OpenRouterCallResult:
            attempts = _openrouter_generation_attempts()
            last_exc: RetryableHostedResearchLabWorkerError | None = None
            length_failures = 0
            retry_provider_usage: list[dict[str, Any]] = []
            retry_cost_microusd = 0
            # Also true for opt-in bare include_reasoning requests so the
            # unsupported-drop retry can clear it for strict models.
            include_reasoning_effort = bool(requested_reasoning_effort) or _llm_include_reasoning_enabled()
            reasoning_effort_drop_error_hash = ""
            for attempt in range(1, attempts + 1):
                effective_max_tokens = _openrouter_generation_retry_max_tokens(
                    base_max_tokens,
                    length_failures,
                )
                try:
                    try:
                        result = _call_once(
                            effective_max_tokens=effective_max_tokens,
                            include_reasoning_effort=include_reasoning_effort,
                        )
                    except OpenRouterReasoningEffortUnsupportedError as exc:
                        if not include_reasoning_effort:
                            raise
                        include_reasoning_effort = False
                        reasoning_effort_drop_error_hash = sha256_json({"error": str(exc)})
                        logger.warning(
                            (
                                "research_lab_openrouter_reasoning_effort_unsupported "
                                "model=%s effort=%s attempt=%s error_hash=%s; retrying_without_reasoning_effort"
                            ),
                            compact_ref(model_id),
                            requested_reasoning_effort,
                            attempt,
                            reasoning_effort_drop_error_hash,
                        )
                        result = _call_once(
                            effective_max_tokens=effective_max_tokens,
                            include_reasoning_effort=False,
                        )
                    if reasoning_effort_drop_error_hash:
                        provider_usage = dict(result.provider_usage or {})
                        provider_usage["reasoning_effort_dropped"] = True
                        provider_usage["reasoning_request_dropped"] = True
                        provider_usage["requested_reasoning_effort"] = requested_reasoning_effort
                        provider_usage["reasoning_effort_drop_error_hash"] = reasoning_effort_drop_error_hash
                        reasoning_capture = dict(
                            provider_usage.get("reasoning_capture")
                            if isinstance(provider_usage.get("reasoning_capture"), Mapping)
                            else {}
                        )
                        reasoning_capture["requested"] = True
                        reasoning_capture["request_dropped"] = True
                        reasoning_capture["drop_error_hash"] = reasoning_effort_drop_error_hash
                        reasoning_capture["requested_reasoning_effort"] = requested_reasoning_effort
                        provider_usage["reasoning_capture"] = reasoning_capture
                        result = OpenRouterCallResult(
                            content=result.content,
                            provider_usage=provider_usage,
                            cost_microusd=result.cost_microusd,
                        )
                    if retry_cost_microusd <= 0 and not retry_provider_usage:
                        return result
                    provider_usage = dict(result.provider_usage or {})
                    provider_usage["retry_attempt_count"] = len(retry_provider_usage)
                    provider_usage["retry_cost_microusd"] = retry_cost_microusd
                    provider_usage["retry_cost_usd"] = round(retry_cost_microusd / 1_000_000, 6)
                    provider_usage["retry_attempts"] = retry_provider_usage
                    aggregate_cost_microusd = max(0, int(result.cost_microusd)) + retry_cost_microusd
                    provider_usage["aggregate_cost_microusd"] = aggregate_cost_microusd
                    provider_usage["aggregate_cost_usd"] = round(aggregate_cost_microusd / 1_000_000, 6)
                    return OpenRouterCallResult(
                        content=result.content,
                        provider_usage=provider_usage,
                        cost_microusd=aggregate_cost_microusd,
                    )
                except CreditBlockedHostedRunError:
                    raise
                except RetryableHostedResearchLabWorkerError as exc:
                    last_exc = exc
                    if isinstance(exc, OpenRouterLengthRetryableError):
                        length_failures += 1
                        if exc.provider_usage:
                            retry_provider_usage.append(
                                {
                                    **exc.provider_usage,
                                    "retry_attempt": attempt,
                                    "retry_reason": "finish_reason_length",
                                }
                            )
                        retry_cost_microusd += max(0, int(exc.cost_microusd))
                    if attempt >= attempts:
                        if isinstance(exc, OpenRouterLengthRetryableError):
                            # Bug 25: the spend accumulated across failed length
                            # retries must ride the exception or it vanishes
                            # from every ledger when the call terminally fails.
                            terminal = HostedResearchLabWorkerError(
                                "OpenRouter candidate generation exceeded output token cap after length retries"
                            )
                            terminal.cost_microusd = retry_cost_microusd  # type: ignore[attr-defined]
                            terminal.provider_usage = {  # type: ignore[attr-defined]
                                "retry_attempt_count": len(retry_provider_usage),
                                "retry_cost_microusd": retry_cost_microusd,
                                "retry_attempts": retry_provider_usage,
                            }
                            raise terminal from exc
                        raise
                    next_max_tokens = _openrouter_generation_retry_max_tokens(
                        base_max_tokens,
                        length_failures,
                    )
                    logger.warning(
                        (
                            "research_lab_openrouter_generation_retrying model=%s attempt=%s attempts=%s "
                            "max_tokens=%s next_max_tokens=%s length_retry=%s error_hash=%s"
                        ),
                        compact_ref(model_id),
                        attempt,
                        attempts,
                        effective_max_tokens,
                        next_max_tokens,
                        isinstance(exc, OpenRouterLengthRetryableError),
                        sha256_json({"error": str(exc)}),
                    )
                    time.sleep(min(2.0, 0.25 * attempt))
            if last_exc is not None:
                raise last_exc
            raise RetryableHostedResearchLabWorkerError("OpenRouter candidate generation failed without response")

        return await asyncio.to_thread(_call)

    def _run_budget_context(self, context: HostedRunContext) -> dict[str, Any]:
        ticket_doc = context.ticket.get("ticket_doc") if isinstance(context.ticket.get("ticket_doc"), Mapping) else {}
        queue_doc = _latest_event_doc(context.queue_events)
        payment_doc = (
            context.payment.get("verification_doc")
            if context.payment and isinstance(context.payment.get("verification_doc"), Mapping)
            else {}
        )
        tier = (
            queue_doc.get("research_model_tier")
            or payment_doc.get("research_model_tier")
            or ticket_doc.get("research_model_tier")
            or self.config.default_auto_research_model_tier
        )
        requested_budget = (
            queue_doc.get("requested_compute_budget_usd")
            or payment_doc.get("compute_budget_usd")
            or payment_doc.get("requested_compute_budget_usd")
            or ticket_doc.get("requested_compute_budget_usd")
            or self.config.default_compute_budget_usd
        )
        max_budget = (
            queue_doc.get("max_compute_budget_usd")
            or payment_doc.get("max_compute_budget_usd")
            or ticket_doc.get("max_compute_budget_usd")
            or self.config.max_compute_budget_usd
        )
        payment_kind = queue_doc.get("payment_kind") or payment_doc.get("payment_kind") or "loop_start"
        additional_budget = queue_doc.get("additional_compute_budget_usd") or payment_doc.get("additional_compute_budget_usd")
        context_doc: dict[str, Any] = {
            "schema_version": "1.0",
            "research_model_tier": str(tier),
            "requested_compute_budget_usd": self.config.clamp_compute_budget_usd(requested_budget),
            "max_compute_budget_usd": self.config.clamp_compute_budget_usd(max_budget),
            "payment_kind": str(payment_kind),
            "budget_policy_version": "research-lab-budget:v1",
        }
        if additional_budget is not None:
            context_doc["additional_compute_budget_usd"] = self.config.clamp_compute_budget_usd(additional_budget)
        if queue_doc.get("continue_from_run_id"):
            context_doc["continue_from_run_id"] = str(queue_doc["continue_from_run_id"])
        if isinstance(queue_doc.get("continuation_context"), Mapping):
            context_doc["continuation_context"] = dict(queue_doc["continuation_context"])
        if queue_doc.get("topup_reason"):
            context_doc["topup_reason"] = str(queue_doc["topup_reason"])
        return context_doc

    def _max_candidates_for_run(self, budget_context: Mapping[str, Any], model_doc: Mapping[str, Any]) -> int:
        configured = int(model_doc.get("max_candidates") or self.config.hosted_worker_max_candidates)
        budget = float(budget_context.get("requested_compute_budget_usd") or self.config.default_compute_budget_usd)
        budget_limited = max(1, min(configured, int(max(1.0, budget // max(1.0, self.config.min_compute_budget_usd)))))
        return max(1, min(self.config.hosted_worker_max_candidates, budget_limited))

    def _auto_research_max_tokens_for_call(
        self,
        *,
        requested_max_tokens: int,
        model_doc: Mapping[str, Any],
    ) -> int:
        configured = model_doc.get("max_tokens")
        try:
            configured_tokens = int(configured) if configured not in (None, "") else 0
        except (TypeError, ValueError):
            configured_tokens = 0
        if configured_tokens <= 0:
            configured_tokens = int(self.config.auto_research_max_tokens)
        return max(1, max(int(requested_max_tokens or 0), configured_tokens))

    async def _active_parent_outcome_memory(
        self,
        artifact: Any,
        *,
        limit: int = 100,
    ) -> dict[str, Any]:
        parent_hash = str(getattr(artifact, "model_artifact_hash", "") or "")
        if not parent_hash:
            return {}
        try:
            candidates = await select_many(
                "research_lab_candidate_evaluation_current",
                columns=(
                    "candidate_id,run_id,parent_artifact_hash,current_candidate_status,current_reason,"
                    "current_score_bundle_id,candidate_patch_manifest,redacted_public_summary,created_at,current_status_at"
                ),
                filters=(("parent_artifact_hash", parent_hash),),
                order_by=(("created_at", True),),
                limit=max(1, min(limit, 200)),
            )
            score_bundles = await select_many(
                "research_evaluation_score_bundle_current",
                columns="score_bundle_id,candidate_artifact_hash,parent_artifact_hash,score_bundle_doc,created_at",
                filters=(("parent_artifact_hash", parent_hash),),
                order_by=(("created_at", True),),
                limit=max(1, min(limit, 200)),
            )
            promotion_events = await select_many(
                "research_lab_candidate_promotion_events",
                columns="candidate_id,event_type,promotion_status,event_doc,created_at",
                filters=(),
                order_by=(("created_at", True),),
                limit=max(1, min(limit * 2, 400)),
            )
        except Exception as exc:
            logger.warning("research_lab_active_parent_outcome_memory_unavailable: %s", str(exc)[:200])
            return {}

        candidate_ids = {str(row.get("candidate_id") or "") for row in candidates}
        candidate_ids.discard("")
        lane_counts: Counter[str] = Counter()
        target_file_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        failure_class_counts: Counter[str] = Counter()
        recent_attempts: list[dict[str, Any]] = []
        for row in candidates:
            status = str(row.get("current_candidate_status") or "unknown")
            reason = str(row.get("current_reason") or "")
            status_counts[status] += 1
            if reason:
                reason_counts[reason] += 1
            lane, files = _candidate_lane_and_files(row.get("candidate_patch_manifest"))
            lane_counts[lane] += 1
            for path in files:
                target_file_counts[path] += 1
            failure_class = _candidate_failure_class_for_memory(row)
            if failure_class:
                failure_class_counts[failure_class] += 1
            attempt = _candidate_attempt_memory(row)
            if attempt:
                recent_attempts.append(attempt)

        promotion_counts: Counter[str] = Counter()
        public_holdout_rejected = 0
        for row in promotion_events:
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id and candidate_id not in candidate_ids:
                continue
            event_type = str(row.get("event_type") or "")
            if event_type:
                promotion_counts[event_type] += 1
            if event_type == "public_holdout_rejected":
                public_holdout_rejected += 1

        scored_count = 0
        positive_delta_count = 0
        nonpositive_delta_count = 0
        best_mean_delta: float | None = None
        worst_mean_delta: float | None = None
        best_delta_lcb: float | None = None
        score_health_counts: Counter[str] = Counter()
        for row in score_bundles:
            doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
            aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), Mapping) else {}
            if not aggregates:
                continue
            scored_count += 1
            mean_delta = _safe_float_for_memory(aggregates.get("mean_delta"))
            delta_lcb = _safe_float_for_memory(aggregates.get("delta_lcb"))
            if mean_delta > 0:
                positive_delta_count += 1
            else:
                nonpositive_delta_count += 1
            best_mean_delta = mean_delta if best_mean_delta is None else max(best_mean_delta, mean_delta)
            worst_mean_delta = mean_delta if worst_mean_delta is None else min(worst_mean_delta, mean_delta)
            best_delta_lcb = delta_lcb if best_delta_lcb is None else max(best_delta_lcb, delta_lcb)
            health = doc.get("scoring_health") if isinstance(doc.get("scoring_health"), Mapping) else {}
            score_health_counts[str(health.get("health_status") or "unknown")] += 1

        guidance: list[str] = []
        if failure_class_counts:
            guidance.append("Avoid repeating recently failed provider/runtime failure modes.")
        if public_holdout_rejected:
            guidance.append("Public holdout rejected recent candidates; prefer changes likely to improve public-visible ICP quality before private holdout.")
        if nonpositive_delta_count >= positive_delta_count and scored_count:
            guidance.append("Recent scored candidates had weak or negative deltas; prefer narrow, testable code edits over broad prompt rewrites.")

        return {
            "schema_version": "1.0",
            "source": "active_parent_recent_outcome_memory",
            "active_parent_artifact_hash": parent_hash,
            "sampled_candidate_count": len(candidates),
            "sampled_score_bundle_count": scored_count,
            "candidate_status_counts": _top_counter(status_counts, limit=8),
            "candidate_reason_counts": _top_counter(reason_counts, limit=8),
            "promotion_decision_counts": _top_counter(promotion_counts, limit=8),
            "public_holdout_rejected_count": public_holdout_rejected,
            "lane_counts": _top_counter(lane_counts, limit=8),
            "target_file_counts": _top_counter(target_file_counts, limit=8),
            "failure_class_counts": _top_counter(failure_class_counts, limit=8),
            "score_bundle_health_counts": _top_counter(score_health_counts, limit=8),
            "scored_delta_summary": {
                "count": scored_count,
                "positive_count": positive_delta_count,
                "negative_or_zero_count": nonpositive_delta_count,
                "best_mean_delta": round(best_mean_delta, 6) if best_mean_delta is not None else None,
                "worst_mean_delta": round(worst_mean_delta, 6) if worst_mean_delta is not None else None,
                "best_delta_lcb": round(best_delta_lcb, 6) if best_delta_lcb is not None else None,
            },
            "recent_attempts": recent_attempts[:25],
            "guidance": guidance[:4],
        }


def _miner_openrouter_key_ref(context: HostedRunContext) -> str:
    for event in (*context.queue_events, *context.ticket_events):
        event_doc = event.get("event_doc")
        if isinstance(event_doc, Mapping) and event_doc.get("miner_openrouter_key_ref"):
            return str(event_doc["miner_openrouter_key_ref"])
    direct = str(context.ticket.get("miner_openrouter_key_ref") or "")
    if direct:
        return direct
    raise HostedResearchLabWorkerError("Research Lab run is missing miner OpenRouter key ref")


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _row_dt(value: Any) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromtimestamp(0, timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sum_award_usd(rows: Iterable[Mapping[str, Any]]) -> float:
    total = Decimal("0")
    for row in rows:
        try:
            total += Decimal(str(row.get("target_reimbursement_microusd", 0))) / Decimal("1000000")
        except Exception:
            continue
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _merge_failure_exception_cost_evidence(
    cost_evidence: Mapping[str, Any] | None,
    failure_exception: BaseException | None,
) -> dict[str, Any]:
    evidence = normalize_cost_evidence(cost_evidence)
    if failure_exception is None:
        return evidence
    exception_cost_microusd = max(0, int(getattr(failure_exception, "cost_microusd", 0) or 0))
    exception_usage = _failure_exception_provider_usage(failure_exception)
    if exception_cost_microusd <= 0 and not exception_usage:
        return evidence

    prior_cost_microusd = cost_evidence_actual_microusd(evidence)
    merged_cost_microusd = prior_cost_microusd + exception_cost_microusd
    prior_ledger = cost_evidence_cost_ledger(evidence)
    merged_ledger = {
        **prior_ledger,
        "schema_version": "1.0",
        "status": "failed",
        "stage": "failure_exception_after_latest_loop_evidence",
        "total_usd": round(merged_cost_microusd / 1_000_000, 6),
        "actual_openrouter_cost_usd": round(merged_cost_microusd / 1_000_000, 6),
        "actual_openrouter_cost_microusd": merged_cost_microusd,
        "previous_actual_openrouter_cost_microusd": prior_cost_microusd,
        "failure_exception_cost_microusd": exception_cost_microusd,
    }
    if "openrouter_call_count" not in merged_ledger:
        merged_ledger["openrouter_call_count"] = int(evidence.get("openrouter_call_count") or 0)
    if "estimated_cost_usd" not in merged_ledger:
        merged_ledger["estimated_cost_usd"] = float(evidence.get("estimated_cost_usd") or 0.0)

    return normalize_cost_evidence(
        {
            "source": (
                "loop_event_plus_failure_exception"
                if prior_cost_microusd > 0 or prior_ledger
                else "failure_exception"
            ),
            "trusted_cost_ledger": bool(prior_ledger) or exception_cost_microusd > 0,
            "provider_usage": cost_evidence_provider_usage(evidence) + exception_usage,
            "cost_ledger": merged_ledger,
            "actual_openrouter_cost_microusd": merged_cost_microusd,
            "estimated_cost_usd": evidence.get("estimated_cost_usd"),
            "openrouter_call_count": evidence.get("openrouter_call_count"),
            "iterations_completed": evidence.get("iterations_completed"),
            "stop_reason": str(evidence.get("stop_reason") or "") or str(failure_exception)[:160],
            "elapsed_seconds": evidence.get("elapsed_seconds"),
        }
    )


def _failure_exception_provider_usage(exc: BaseException) -> list[dict[str, Any]]:
    usage = getattr(exc, "provider_usage", None)
    if isinstance(usage, Mapping):
        return [{**dict(usage), "call_outcome": "failure_exception"}]
    if isinstance(usage, Sequence) and not isinstance(usage, (str, bytes, bytearray)):
        return [
            {**dict(item), "call_outcome": str(dict(item).get("call_outcome") or "failure_exception")}
            for item in usage
            if isinstance(item, Mapping)
        ]
    return []


def _redacted_reimbursement_run_cost(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "run_id",
        "miner_hotkey",
        "island",
        "run_day",
        "funded_compute_budget_usd",
        "actual_openrouter_cost_usd",
        "loop_start_tao_fee_usd",
        "paid_research_loop",
        "valid_receipt",
        "verified_loop_start_payment",
        "preserved_loop_start_credit",
        "miner_openrouter_key_present",
        "trusted_cost_ledger",
        "passed_abuse_checks",
        "refunded",
        "voided",
        "duplicate",
        "novelty_rejected",
        "self_cancelled_before_minimum_work",
        "banned_hotkey",
    }
    return {key: value[key] for key in allowed if key in value}


def _ticket_doc_value(context: HostedRunContext, key: str) -> Any:
    ticket_doc = context.ticket.get("ticket_doc")
    if isinstance(ticket_doc, Mapping):
        return ticket_doc.get(key)
    return None


def _payment_id_from_queue_events(events: Sequence[Mapping[str, Any]]) -> str:
    for event in events:
        event_doc = event.get("event_doc")
        if isinstance(event_doc, Mapping) and event_doc.get("payment_id"):
            return str(event_doc["payment_id"])
    return ""


def _retryable_requeue_count(context: HostedRunContext) -> int:
    count = 0
    for event in context.queue_events:
        event_doc = event.get("event_doc")
        if not isinstance(event_doc, Mapping):
            event_doc = {}
        if str(event.get("reason") or "") == "transient_worker_error_requeued" or event_doc.get("retryable_error"):
            count += 1
            try:
                count = max(count, int(event_doc.get("retryable_error_count") or 0))
            except (TypeError, ValueError):
                pass
    return count


def _loop_start_credit_id_from_queue_events(events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in events:
        event_doc = event.get("event_doc")
        if isinstance(event_doc, Mapping) and event_doc.get("loop_start_credit_id"):
            return str(event_doc["loop_start_credit_id"])
    return None


def _context_loop_start_credit_id(context: HostedRunContext) -> str | None:
    """Synchronous credit-ref read for receipt builders: prefer the value
    resolved by _resolve_loop_start_credit_id (which looks past the run-context
    event window), falling back to the in-window scan."""
    if context.loop_start_credit_id_resolved:
        return context.resolved_loop_start_credit_id
    return _loop_start_credit_id_from_queue_events(context.queue_events)


def _latest_event_doc(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for event in events:
        event_doc = event.get("event_doc")
        if isinstance(event_doc, Mapping):
            return dict(event_doc)
    return {}


def _redacted_budget_context(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "research_model_tier",
        "requested_compute_budget_usd",
        "max_compute_budget_usd",
        "payment_kind",
        "budget_policy_version",
        "additional_compute_budget_usd",
        "continue_from_run_id",
        "continuation_context",
        "topup_reason",
    }
    return {key: value[key] for key in allowed if key in value}


def _candidate_lane_and_files(value: Any) -> tuple[str, tuple[str, ...]]:
    manifest = value if isinstance(value, Mapping) else {}
    patch_doc = manifest.get("patch_doc") if isinstance(manifest.get("patch_doc"), Mapping) else {}
    code_edit = patch_doc.get("code_edit") if isinstance(patch_doc.get("code_edit"), Mapping) else {}
    lane = str(code_edit.get("lane") or patch_doc.get("lane") or "unknown").strip() or "unknown"
    raw_files = code_edit.get("target_files") or patch_doc.get("target_files") or ()
    files: list[str] = []
    if isinstance(raw_files, Sequence) and not isinstance(raw_files, (str, bytes, bytearray)):
        for item in raw_files:
            path = str(item or "").strip()
            if path and not _looks_secret_like(path):
                files.append(path[:160])
    return lane[:80], tuple(files[:12])


def _candidate_attempt_memory(row: Mapping[str, Any]) -> dict[str, Any]:
    manifest = row.get("candidate_patch_manifest") if isinstance(row.get("candidate_patch_manifest"), Mapping) else {}
    patch_doc = manifest.get("patch_doc") if isinstance(manifest.get("patch_doc"), Mapping) else {}
    code_edit = patch_doc.get("code_edit") if isinstance(patch_doc.get("code_edit"), Mapping) else {}
    lane, files = _candidate_lane_and_files(manifest)
    summary = str(row.get("redacted_public_summary") or manifest.get("redacted_summary") or "")[:500]
    if not summary:
        summary = str(code_edit.get("expected_improvement") or code_edit.get("test_plan") or "")[:500]
    return {
        "candidate_id": str(row.get("candidate_id") or "")[:120],
        "run_id": str(row.get("run_id") or "")[:120],
        "lane": lane,
        "plan_path_id": str(code_edit.get("plan_path_id") or patch_doc.get("plan_path_id") or "")[:120],
        "target_files": list(files),
        "unified_diff_hash": str(code_edit.get("unified_diff_hash") or patch_doc.get("unified_diff_hash") or "")[:120],
        "candidate_source_diff_hash": str(
            row.get("candidate_source_diff_hash")
            or manifest.get("candidate_source_diff_hash")
            or manifest.get("patch_payload_hash")
            or ""
        )[:120],
        "semantic_edit_summary": summary,
        "status": str(row.get("current_candidate_status") or "")[:120],
        "reason": str(row.get("current_reason") or "")[:240],
    }


def _resolve_code_edit_loop_stage_model_request(
    config: ResearchLabGatewayConfig,
    *,
    stage: str,
    model_id: str,
    model_doc: Mapping[str, Any],
    requested_max_tokens: int,
) -> dict[str, Any]:
    normalized_stage = str(stage or "code_edit_draft")
    base_reasoning_effort = str(model_doc.get("reasoning_effort") or "")
    if normalized_stage == "loop_planner":
        model_ids = tuple(
            item
            for item in (
                config.loop_planner_model or model_id,
                *config.loop_planner_fallback_models,
            )
            if str(item or "").strip()
        )
        return {
            "stage": normalized_stage,
            "model_id": model_ids[0] if model_ids else model_id,
            "model_ids": model_ids or (model_id,),
            "reasoning_effort": config.loop_planner_reasoning_effort or base_reasoning_effort,
            "temperature": config.loop_planner_temperature,
            "max_tokens": max(int(requested_max_tokens or 0), config.loop_planner_max_tokens),
            "allow_non_zdr": config.loop_planner_allow_non_zdr,
        }
    if normalized_stage == "plan_alignment_judge":
        return {
            "stage": normalized_stage,
            "model_id": config.loop_alignment_judge_model or config.loop_executor_model or model_id,
            "reasoning_effort": config.loop_alignment_judge_reasoning_effort,
            "temperature": config.loop_alignment_judge_temperature,
            "max_tokens": max(int(requested_max_tokens or 0), config.loop_alignment_judge_max_tokens),
        }
    return {
        "stage": normalized_stage,
        "model_id": config.loop_executor_model or model_id,
        "reasoning_effort": config.loop_executor_reasoning_effort,
        "temperature": config.auto_research_temperature,
        "max_tokens": int(requested_max_tokens or 0),
    }


def _candidate_failure_class_for_memory(row: Mapping[str, Any]) -> str:
    reason = str(row.get("current_reason") or "")
    if reason:
        return reason[:120]
    status = str(row.get("current_candidate_status") or "")
    return status[:120] if status in {"failed", "rejected"} else ""


def _top_counter(counter: Counter[str], *, limit: int) -> dict[str, int]:
    return {
        str(key)[:160]: int(value)
        for key, value in counter.most_common(max(0, limit))
        if key
    }


def _safe_float_for_memory(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _looks_secret_like(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "sk-or-",
            "service_role",
            "api_key",
            "secret",
            "proxy",
            "://",
        )
    )


def _validator_evaluation_summary() -> dict[str, Any]:
    return {
        "benchmark_id": "research_lab:rolling_icp_window:latest_10_days",
        "icp_set_hash": "gateway_resolves_rolling_icp_window_hash",
        "split_ref": "gateway_private_rolling_icp_window",
        "item_count": 60,
        "scoring_version": "qualification-company-scorer:v1",
        "owner": "gateway_qualification_workers",
    }


def _worker_proxy_url(config: ResearchLabGatewayConfig) -> str:
    return str(config.hosted_worker_proxy_url or "").strip()


def _worker_llm_proxy_opener(config: ResearchLabGatewayConfig) -> Any | None:
    """Opener routing the worker's own OpenRouter traffic through the configured
    hosted-worker proxy. Gated on RESEARCH_LAB_WORKER_PROXY_APPLY_TO_LLM
    (default: current behavior — proxy exported to docker runs only)."""
    if not _worker_proxy_apply_to_llm_enabled():
        return None
    proxy = _worker_proxy_url(config)
    if not proxy:
        return None
    return urlrequest.build_opener(urlrequest.ProxyHandler({"http": proxy, "https": proxy}))


def _worker_proxy_env(config: ResearchLabGatewayConfig) -> dict[str, str]:
    proxy = _worker_proxy_url(config)
    if not proxy:
        return {}
    env = {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
    }
    no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy
    return env


_PROXY_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


def _private_model_env_passthrough(config: ResearchLabGatewayConfig) -> tuple[str, ...]:
    return private_model_env_passthrough(
        include_proxy=config.private_model_docker_global_proxy_enabled
    )


def _private_model_docker_env(
    config: ResearchLabGatewayConfig,
    provider_env: Mapping[str, str],
) -> dict[str, str]:
    env = {str(key): str(value) for key, value in provider_env.items() if value}
    if config.private_model_docker_global_proxy_enabled:
        return env
    return {key: value for key, value in env.items() if key not in _PROXY_ENV_NAMES}


def _row_partition(row: Mapping[str, Any], total_workers: int) -> int:
    total = max(1, int(total_workers))
    digest = canonical_hash({"run_id": str(row.get("run_id", ""))}).split(":", 1)[1]
    return int(digest[:12], 16) % total


def _is_claim_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_run_claim_conflict" in message
        or "research_loop_run_queue_events_run_seq_key" in message
        or "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )


def _is_queue_capacity_conflict_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_queue_capacity_conflict" in message
        or "research_lab_queue_hotkey_conflict" in message
    )


def _discard_backgrounded_task_result(task: "asyncio.Task[Any]") -> None:
    """Consume a detached background task's outcome so asyncio never logs an
    unretrieved exception for work abandoned after a claim-lost abort."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "research_lab_backgrounded_phase_failed_after_claim_lost error=%s",
            str(exc)[:240],
        )


def _usage_cost_usd(usage: Mapping[str, Any]) -> Decimal:
    for key in ("cost", "total_cost", "cost_usd"):
        value = usage.get(key)
        if value is None:
            continue
        try:
            return Decimal(str(value))
        except Exception:
            continue
    return Decimal("0")


def _generation_stats_cost_usd(stats: Mapping[str, Any]) -> Decimal | None:
    for key in ("total_cost", "usage"):
        value = stats.get(key)
        if value is None:
            continue
        try:
            decimal_value = Decimal(str(value))
        except Exception:
            continue
        if decimal_value < 0:
            return Decimal("0")
        return decimal_value
    return None


def _usd_to_microusd(value: Decimal | float | int | str) -> int:
    try:
        decimal_value = Decimal(str(value))
    except Exception:
        decimal_value = Decimal("0")
    if decimal_value < 0:
        decimal_value = Decimal("0")
    return int((decimal_value * Decimal("1000000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("research_lab_openrouter_usage_int_parse_failed value_type=%s", type(value).__name__)
        return None


def _build_openrouter_provider_usage(
    *,
    decoded: Mapping[str, Any],
    usage: Mapping[str, Any],
    model_id: str,
    api_key: str,
    generation_stats_opener: Any | None = None,
    reasoning_requested: bool = False,
    requested_reasoning_effort: str = "",
) -> tuple[dict[str, Any], int]:
    usage_cost_microusd = _usd_to_microusd(_usage_cost_usd(usage))
    response_id = str(decoded.get("id") or "")
    choices = decoded.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    reasoning_logs = _safe_openrouter_reasoning_logs(decoded)
    usage_reasoning_token_count = _openrouter_reasoning_token_count(usage)
    reasoning_capture: dict[str, Any] = {
        "requested": bool(reasoning_requested),
        "returned": bool(reasoning_logs),
        "fields_present": list(reasoning_logs.get("fields_present") or []) if reasoning_logs else [],
        "reasoning_hashes": list(reasoning_logs.get("reasoning_hashes") or []) if reasoning_logs else [],
        "reasoning_token_count": usage_reasoning_token_count,
        "requested_reasoning_effort": str(requested_reasoning_effort or "")[:80],
        "raw_trace_ref_present": False,
        "storage_state": (
            "metadata_only"
            if reasoning_logs
            else ("requested_but_absent" if reasoning_requested else "not_requested")
        ),
        "storage_policy": (
            str(reasoning_logs.get("storage_policy") or "") if reasoning_logs else "no_reasoning_returned"
        ),
    }
    provider_usage: dict[str, Any] = {
        "provider": "openrouter",
        "key_source": "miner_key_ref",
        "response_id": response_id,
        "model": str(decoded.get("model") or model_id),
        "finish_reason": str(first_choice.get("finish_reason") or "")[:120],
        "native_finish_reason": str(first_choice.get("native_finish_reason") or "")[:120],
        "prompt_tokens": _int_or_none(usage.get("prompt_tokens")),
        "completion_tokens": _int_or_none(usage.get("completion_tokens")),
        "total_tokens": _int_or_none(usage.get("total_tokens")),
        "usage_cost_usd": round(usage_cost_microusd / 1_000_000, 6),
        "usage_cost_microusd": usage_cost_microusd,
        "cost_usd": round(usage_cost_microusd / 1_000_000, 6),
        "cost_microusd": usage_cost_microusd,
        "cost_source": "chat_completion_usage",
        "cost_reconciliation_status": "pending_generation_stats",
        "cost_details": _safe_cost_details(usage.get("cost_details")),
        "reasoning_capture": reasoning_capture,
    }
    if usage_reasoning_token_count is not None:
        provider_usage["reasoning_token_count"] = usage_reasoning_token_count
    if reasoning_logs:
        provider_usage["reasoning_logs"] = reasoning_logs
    if not response_id:
        provider_usage["cost_reconciliation_status"] = "missing_response_id"
        return provider_usage, usage_cost_microusd

    generation_stats, stats_status = _fetch_openrouter_generation_stats(
        api_key=api_key,
        response_id=response_id,
        opener=generation_stats_opener,
    )
    provider_usage["generation_stats_status"] = stats_status
    if generation_stats is None:
        provider_usage["cost_reconciliation_status"] = "generation_stats_unavailable"
        return provider_usage, usage_cost_microusd

    provider_usage["generation_stats"] = _safe_generation_stats(generation_stats)
    stats_reasoning_token_count = _openrouter_reasoning_token_count(generation_stats)
    if stats_reasoning_token_count is not None:
        provider_usage["reasoning_token_count"] = stats_reasoning_token_count
        reasoning_capture = dict(provider_usage.get("reasoning_capture") or {})
        reasoning_capture["reasoning_token_count"] = stats_reasoning_token_count
        provider_usage["reasoning_capture"] = reasoning_capture
    generation_cost_usd = _generation_stats_cost_usd(generation_stats)
    if generation_cost_usd is None:
        provider_usage["cost_reconciliation_status"] = "generation_stats_missing_cost"
        return provider_usage, usage_cost_microusd

    generation_cost_microusd = _usd_to_microusd(generation_cost_usd)
    provider_usage.update(
        {
            "generation_cost_usd": round(generation_cost_microusd / 1_000_000, 6),
            "generation_cost_microusd": generation_cost_microusd,
            "cost_usd": round(generation_cost_microusd / 1_000_000, 6),
            "cost_microusd": generation_cost_microusd,
            "cost_source": "openrouter_generation_stats",
            "cost_reconciliation_status": "confirmed",
        }
    )
    if generation_cost_microusd != usage_cost_microusd:
        provider_usage["usage_generation_cost_delta_microusd"] = generation_cost_microusd - usage_cost_microusd
    return provider_usage, generation_cost_microusd


def _fetch_openrouter_generation_stats(
    *,
    api_key: str,
    response_id: str,
    opener: Any | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not api_key or not response_id:
        return None, "missing_inputs"
    stats_mode = _generation_stats_mode()
    if stats_mode == "off":
        return None, "disabled"
    # "best_effort_once" skips the retry sleeps that otherwise run synchronously
    # after every chat call in the hot path.
    attempts_allowed = 1 if stats_mode == "best_effort_once" else _OPENROUTER_GENERATION_STATS_ATTEMPTS
    open_fn = opener or urlrequest.urlopen
    query = urlparse.urlencode({"id": response_id})
    req = urlrequest.Request(
        f"{_OPENROUTER_GENERATION_STATS_URL}?{query}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    last_status = "not_attempted"
    for attempt in range(1, attempts_allowed + 1):
        try:
            with open_fn(req, timeout=_OPENROUTER_GENERATION_STATS_TIMEOUT_SECONDS) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_status = f"http_{int(exc.code)}"
            if int(exc.code) not in _OPENROUTER_GENERATION_STATS_RETRYABLE_HTTP_CODES:
                logger.warning(
                    "research_lab_openrouter_generation_stats_failed response_id=%s status=%s",
                    compact_ref(response_id),
                    last_status,
                )
                return None, last_status
        except URLError as exc:
            last_status = f"url_error:{type(exc.reason).__name__}"
        except TimeoutError:
            last_status = "timeout"
        except Exception as exc:
            last_status = f"error:{type(exc).__name__}"
            logger.warning(
                "research_lab_openrouter_generation_stats_error response_id=%s status=%s",
                compact_ref(response_id),
                last_status,
            )
            return None, last_status
        else:
            data = decoded.get("data") if isinstance(decoded, Mapping) else None
            if isinstance(data, Mapping):
                return dict(data), "ok"
            logger.warning(
                "research_lab_openrouter_generation_stats_invalid_response response_id=%s",
                compact_ref(response_id),
            )
            return None, "invalid_response"
        if attempt < attempts_allowed:
            delay_index = min(attempt - 1, len(_OPENROUTER_GENERATION_STATS_RETRY_DELAYS_SECONDS) - 1)
            time.sleep(_OPENROUTER_GENERATION_STATS_RETRY_DELAYS_SECONDS[delay_index])

    logger.warning(
        "research_lab_openrouter_generation_stats_unavailable response_id=%s status=%s attempts=%s",
        compact_ref(response_id),
        last_status,
        attempts_allowed,
    )
    return None, last_status


def _safe_cost_details(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if key.lower() in {"api_key", "openrouter_api_key", "raw_secret", "raw_openrouter_key"}:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            allowed[key] = item
    return allowed


def _bounded_openrouter_diagnostic(value: Any, *, limit: int = 20000) -> tuple[str, int, bool]:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    byte_count = len(text.encode("utf-8"))
    redacted = _redact_openrouter_diagnostic(text, limit=limit)
    return redacted, byte_count, byte_count > max(1, int(limit))


def _openrouter_reasoning_token_count(*docs: Mapping[str, Any]) -> int | None:
    for doc in docs:
        if not isinstance(doc, Mapping):
            continue
        for key in ("native_tokens_reasoning", "reasoning_tokens", "tokens_reasoning"):
            value = _int_or_none(doc.get(key))
            if value is not None:
                return max(0, value)
        completion_details = doc.get("completion_tokens_details")
        if isinstance(completion_details, Mapping):
            value = _int_or_none(completion_details.get("reasoning_tokens"))
            if value is not None:
                return max(0, value)
    return None


def _safe_openrouter_reasoning_logs(decoded: Mapping[str, Any]) -> dict[str, Any]:
    choices = decoded.get("choices")
    if not isinstance(choices, list):
        return {}
    choice_logs: list[dict[str, Any]] = []
    all_fields: set[str] = set()
    all_hashes: list[str] = []
    first_reasoning: str | None = None
    first_reasoning_hash = ""
    first_reasoning_details: str | None = None
    first_reasoning_details_hash = ""

    for choice_index, raw_choice in enumerate(choices):
        if not isinstance(raw_choice, Mapping):
            continue
        message = raw_choice.get("message") if isinstance(raw_choice.get("message"), Mapping) else {}
        if not message:
            continue
        fields_present: list[str] = []
        choice_doc: dict[str, Any] = {"choice_index": int(choice_index), "fields_present": fields_present}
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            fields_present.append("reasoning")
            preview, byte_count, truncated = _bounded_openrouter_diagnostic(reasoning)
            reasoning_hash = sha256_json({"choice_index": choice_index, "reasoning": reasoning})
            choice_doc.update(
                {
                    "reasoning": preview,
                    "reasoning_hash": reasoning_hash,
                    "reasoning_byte_count": byte_count,
                    "reasoning_truncated": truncated,
                }
            )
            all_hashes.append(reasoning_hash)
            if first_reasoning is None:
                first_reasoning = preview
                first_reasoning_hash = reasoning_hash
        reasoning_details = message.get("reasoning_details")
        if reasoning_details:
            fields_present.append("reasoning_details")
            preview, byte_count, truncated = _bounded_openrouter_diagnostic(reasoning_details)
            reasoning_details_hash = sha256_json(
                {"choice_index": choice_index, "reasoning_details": reasoning_details}
            )
            choice_doc.update(
                {
                    "reasoning_details": preview,
                    "reasoning_details_hash": reasoning_details_hash,
                    "reasoning_details_byte_count": byte_count,
                    "reasoning_details_truncated": truncated,
                }
            )
            all_hashes.append(reasoning_details_hash)
            if first_reasoning_details is None:
                first_reasoning_details = preview
                first_reasoning_details_hash = reasoning_details_hash
        if fields_present:
            all_fields.update(fields_present)
            choice_logs.append(choice_doc)

    if not choice_logs:
        return {}
    logs: dict[str, Any] = {
        "storage_policy": "redacted_bounded_openrouter_message_reasoning",
        "schema_version": "1.1",
        "choice_count": len(choices),
        "choices_with_reasoning_count": len(choice_logs),
        "fields_present": sorted(all_fields),
        "reasoning_hashes": list(all_hashes),
        "choices": choice_logs,
    }
    # Backward-compatible first-choice/top-level fields for existing readers.
    if first_reasoning is not None:
        logs["reasoning"] = first_reasoning
        logs["reasoning_hash"] = first_reasoning_hash
    if first_reasoning_details is not None:
        logs["reasoning_details"] = first_reasoning_details
        logs["reasoning_details_hash"] = first_reasoning_details_hash
    return logs


def _safe_generation_stats(value: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in sorted(_OPENROUTER_GENERATION_STATS_FIELDS):
        if key not in value:
            continue
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            safe[key] = item
    return safe


@contextmanager
def _temporary_env(values: Mapping[str, str]):
    original: dict[str, str | None] = {}
    try:
        for key, value in values.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
