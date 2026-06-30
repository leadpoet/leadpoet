"""Hosted Research Lab worker for paid private-model auto-research runs."""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import logging
import os
import re
import socket
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.code_build import CodeEditCandidateBuilder
from gateway.research_lab.code_loop_engine import CodeEditLoopEngine
from gateway.research_lab.config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from gateway.research_lab.key_vault import OpenRouterKeyVaultError, decrypt_openrouter_key, preflight_openrouter_key
from gateway.research_lab.logging_utils import compact_ref, format_worker_block, format_worker_line
from gateway.research_lab.loop_engine import (
    AutoResearchLoopEvent,
    AutoResearchLoopSettings,
    OpenRouterCallResult,
)
from gateway.research_lab.maintenance import autoresearch_queue_capacity_doc, is_autoresearch_maintenance_paused
from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest, ResearchLabReceiptCreateRequest
from gateway.research_lab.promotion import latest_public_benchmark_summary, load_active_private_model
from gateway.research_lab.public_activity import safe_project_public_loop_activity
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
from research_lab.canonical import sha256_json
from research_lab.eval import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    private_model_env_passthrough,
)


logger = logging.getLogger(__name__)


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


def _status_age_seconds(raw_status_at: object) -> float | None:
    if not raw_status_at:
        return None
    try:
        status_at = datetime.fromisoformat(str(raw_status_at).replace("Z", "+00:00"))
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


class HostedResearchLabWorkerError(RuntimeError):
    """Raised when a hosted Research Lab run cannot complete safely."""


class RetryableHostedResearchLabWorkerError(HostedResearchLabWorkerError):
    """Raised when a paid hosted run should be requeued instead of terminally failed."""


class HostedResearchLabBuilderNotReady(RetryableHostedResearchLabWorkerError):
    """Raised when image-build candidate infrastructure is not ready yet."""


class OpenRouterCreditBlockedError(HostedResearchLabWorkerError):
    """Raised when a miner OpenRouter key cannot currently fund more model calls."""


class HostedResearchLabClaimLost(HostedResearchLabWorkerError):
    """Raised when another worker safely claimed the queued run first."""


_RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_OPENROUTER_GENERATION_STATS_URL = "https://openrouter.ai/api/v1/generation"
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


def _is_retryable_worker_exception(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RetryableHostedResearchLabWorkerError):
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


def _raise_openrouter_generation_response_error(
    decoded: object,
    *,
    failure: str,
    default_retryable: bool,
) -> None:
    summary = _openrouter_response_summary(decoded)
    error = f"OpenRouter candidate generation failed: {failure}: {summary}"
    if _is_openrouter_credit_block_message(error):
        raise OpenRouterCreditBlockedError(error)
    if _openrouter_generation_response_is_retryable(
        decoded,
        error,
        default_retryable=default_retryable,
    ):
        raise RetryableHostedResearchLabWorkerError(error)
    raise HostedResearchLabWorkerError(error)


def _is_openrouter_credit_block_message(message: object) -> bool:
    lowered = str(message or "").lower()
    return any(marker in lowered for marker in _OPENROUTER_CREDIT_BLOCK_MARKERS)


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
    )
    if any(marker in lowered for marker in secret_markers):
        return "[redacted secret-like diagnostic text]"
    return text[: max(1, int(limit))]


def _redacted_ref(value: object) -> str:
    text = str(value or "")
    if len(text) <= 16:
        return text
    return f"{text[:10]}...{text[-6:]}"


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

    async def run_forever(self) -> None:
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
        if await is_autoresearch_maintenance_paused():
            return HostedWorkerOutcome(
                processed=False,
                dry_run=self.config.hosted_worker_dry_run,
                status="maintenance_paused",
            )
        if not self.config.hosted_worker_dry_run:
            await self._recover_stale_paused_runs()
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
        except OpenRouterCreditBlockedError as exc:
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB AUTO-RESEARCH BLOCKED FOR OPENROUTER CREDIT",
                    (
                        ("Worker", self.worker_ref),
                        ("Run", compact_ref(run_id)),
                        ("Ticket", compact_ref(ticket_id)),
                        ("Reason", str(exc)[:300]),
                    ),
                )
            )
            return await self._mark_credit_blocked(context, str(exc))
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
            return await self._mark_failed(context, str(exc))

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
            order_by=(("current_status_at", True),),
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
                "run_id,ticket_id,current_queue_status,current_status_at,"
                "current_event_hash,queue_priority,worker_ref"
            ),
            filters=(("current_queue_status", "paused"),),
            order_by=(("current_status_at", True),),
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
            limit=20,
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
        resolved_openrouter_env = await self.key_resolver.resolve(
            _miner_openrouter_key_ref(context),
            miner_hotkey=str(context.ticket["miner_hotkey"]),
        )
        provider_env = dict(resolved_openrouter_env)
        context.provider_env = provider_env
        await self._preflight_openrouter_credit(context, provider_env["OPENROUTER_API_KEY"])
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
                    provider_usage=event.provider_usage or self._provider_usage(context),
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
            ) -> str:
                return await self._call_openrouter(
                    messages=messages,
                    api_key=context.provider_env["OPENROUTER_API_KEY"],
                    model_id=model_id,
                    timeout_seconds=timeout_seconds,
                    max_tokens=max_tokens,
                )

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
                raise HostedResearchLabWorkerError("auto-research loop completed without valid image-build finalists")
            final_artifact = artifact
            finalists = [
                {
                    "candidate_kind": "image_build",
                    "selected": candidate,
                    "candidate_patch_manifest": candidate.build.code_edit_manifest,
                    "candidate_model_manifest": candidate.build.candidate_model_manifest.to_dict(),
                    "candidate_source_diff_hash": candidate.build.source_diff_hash,
                    "candidate_build_doc": candidate.build.build_doc,
                    "hypothesis_doc": {
                        "failure_mode": candidate.draft.failure_mode,
                        "mechanism": candidate.draft.mechanism,
                        "expected_improvement": candidate.draft.expected_improvement,
                        "risk": candidate.draft.risk,
                        "focus_alignment": f"code_edit_lane:{candidate.draft.lane}",
                        "predicted_delta": candidate.draft.predicted_delta,
                        "falsifier": "official_scoring",
                    },
                    "patch_doc": {
                        "code_edit": {
                            "lane": candidate.draft.lane,
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
                candidate_build_doc=dict(finalist.get("candidate_build_doc") or {}),
                hypothesis_doc=hypothesis_doc,
                redacted_public_summary=redacted_summary,
            )
            candidate_row, _candidate_event = await self._store_write_with_retry(
                "candidate_artifact_create",
                lambda request=request: create_candidate_artifact(request),
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
        try:
            await create_auto_research_loop_event(
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                receipt_id=context.receipt_id,
                event_type=event_type,
                loop_status=loop_status,
                worker_ref=self.worker_ref,
                provider_usage=provider_usage,
                event_doc={
                    "schema_version": "1.0",
                    "reason": reason,
                    "source": "hosted_worker_terminal_queue_projection",
                    "previous_loop_status": current.get("current_loop_status") if current else None,
                    "previous_loop_event_type": current.get("current_event_type") if current else None,
                    "previous_loop_event_seq": current.get("current_event_seq") if current else None,
                    "previous_loop_event_hash": current.get("current_event_hash") if current else None,
                    **dict(event_doc or {}),
                },
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
            logger.warning(
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

    async def _maybe_create_reimbursement_decision(
        self,
        *,
        context: HostedRunContext,
        budget_context: Mapping[str, Any],
        loop_result: Any,
    ) -> dict[str, Any] | None:
        if not (self.config.reimbursements_enabled or self.config.shadow_reimbursements_enabled):
            return None
        policy = self.config.reimbursement_policy_doc(
            enabled=self.config.reimbursements_enabled or self.config.shadow_reimbursements_enabled
        )
        snapshot_doc, snapshot_row = await self._create_participation_snapshot(context, policy)
        cap_usage = await self._reimbursement_cap_usage(context, run_day=_utc_day())
        run_cost = {
            "run_id": context.run_id,
            "miner_hotkey": str(context.ticket["miner_hotkey"]),
            "island": str(context.ticket["island"] or self.config.reimbursement_default_island),
            "run_day": _utc_day(),
            "funded_compute_budget_usd": float(budget_context.get("requested_compute_budget_usd") or 0.0),
            "actual_openrouter_cost_usd": float(loop_result.actual_openrouter_cost_usd),
            "loop_start_tao_fee_usd": float(self.config.loop_start_fee_usd),
            "paid_research_loop": True,
            "valid_receipt": bool(context.receipt_id),
            "verified_loop_start_payment": bool(context.payment),
            "preserved_loop_start_credit": False,
            "miner_openrouter_key_present": True,
            "trusted_cost_ledger": True,
            "passed_abuse_checks": True,
            "refunded": False,
            "voided": False,
            "duplicate": False,
            "novelty_rejected": False,
            "self_cancelled_before_minimum_work": False,
            "banned_hotkey": False,
        }
        award_obj = compute_reimbursement_award(run_cost, snapshot_doc, policy, ReimbursementCapUsage.from_mapping(cap_usage))
        award = award_obj.to_dict()
        evaluation_epoch, _block, _source = await resolve_research_lab_evaluation_epoch(self.config.evaluation_epoch)
        schedule = build_reimbursement_schedule(
            award,
            start_epoch=max(0, int(evaluation_epoch) + 1),
        ).to_dict()
        shadow_only = not self.config.reimbursements_enabled
        award_doc = {
            "schema_version": "1.0",
            "award": award,
            "run_cost": _redacted_reimbursement_run_cost(run_cost),
            "policy": policy,
            "participation_snapshot": snapshot_doc,
            "cap_usage": cap_usage,
            "shadow_only": shadow_only,
            "submission_allowed": self.config.reimbursements_enabled,
            "source": "hosted_auto_research_loop_completion",
            "evaluation_epoch": int(evaluation_epoch),
        }
        schedule_doc = {
            "schema_version": "1.0",
            "schedule": schedule,
            "shadow_only": shadow_only,
            "submission_allowed": self.config.reimbursements_enabled,
            "source": "hosted_auto_research_loop_completion",
            "evaluation_epoch": int(evaluation_epoch),
        }
        award_row, _award_event = await create_reimbursement_award(
            award=award,
            receipt_id=context.receipt_id,
            participation_snapshot_id=str(snapshot_row["participation_snapshot_id"]),
            policy_id=str(policy["policy_id"]),
            award_doc=award_doc,
        )
        if str(award_row["award_id"]) != str(schedule["award_id"]):
            schedule = build_reimbursement_schedule(
                {**award, "award_id": str(award_row["award_id"])},
                start_epoch=max(0, int(evaluation_epoch) + 1),
            ).to_dict()
            schedule_doc = {**schedule_doc, "schedule": schedule}
        schedule_row = await create_reimbursement_schedule(schedule=schedule, schedule_doc=schedule_doc)
        logger.info(
            format_worker_block(
                "RESEARCH LAB REIMBURSEMENT DECISION",
                (
                    ("Run", compact_ref(context.run_id)),
                    ("Status", award["status"]),
                    ("Target USD", f"{float(award['target_reimbursement_usd']):.6f}"),
                    ("OpenRouter USD", f"{float(loop_result.actual_openrouter_cost_usd):.6f}"),
                    ("Shadow only", shadow_only),
                ),
            )
        )
        return {
            "status": award["status"],
            "award_id": str(award_row["award_id"]),
            "schedule_id": str(schedule_row["schedule_id"]),
            "target_reimbursement_usd": award["target_reimbursement_usd"],
            "rebate_rate": award["rebate_rate"],
            "actual_openrouter_cost_usd": round(float(loop_result.actual_openrouter_cost_usd), 6),
            "shadow_only": shadow_only,
        }

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
            await self._append_queue_heartbeat(
                context,
                source_event_type=f"{heartbeat_label}_heartbeat_{heartbeat_index}",
                source_event_seq=None,
                source_event_hash=None,
            )
        return await task

    async def _preflight_openrouter_credit(self, context: HostedRunContext, api_key: str) -> None:
        try:
            doc = await asyncio.to_thread(preflight_openrouter_key, api_key)
        except OpenRouterKeyVaultError as exc:
            message = str(exc)
            lowered = message.lower()
            if "invalid or unauthorized" in lowered or "disabled" in lowered:
                raise HostedResearchLabWorkerError(f"OpenRouter key preflight failed permanently: {message}") from exc
            logger.warning(
                "research_lab_openrouter_credit_preflight_unavailable run_id=%s key_ref=%s error=%s",
                compact_ref(context.run_id),
                _redacted_ref(_miner_openrouter_key_ref(context)),
                message[:240],
            )
            return
        remaining = doc.get("limit_remaining")
        try:
            remaining_value = float(remaining) if remaining is not None else None
        except (TypeError, ValueError):
            remaining_value = None
        if remaining_value is not None and remaining_value <= 0:
            raise OpenRouterCreditBlockedError(
                "OpenRouter key credit preflight blocked execution: limit_remaining <= 0"
            )

    async def _mark_credit_blocked(self, context: HostedRunContext, error: str) -> HostedWorkerOutcome:
        checkpoint = await latest_auto_research_checkpoint(context.run_id)
        event_doc = {
            **autoresearch_queue_capacity_doc(self.config),
            "schema_version": "1.0",
            "run_id": context.run_id,
            "worker_ref": self.worker_ref,
            "credit_blocked": True,
            "failure_class": "openrouter_credit_blocked",
            "error": _redact_openrouter_diagnostic(error, limit=500),
            "previous_event_hash": context.queue_row.get("current_event_hash"),
            "previous_status_at": context.queue_row.get("current_status_at"),
            "checkpoint_hash": checkpoint.get("checkpoint_hash") if isinstance(checkpoint, Mapping) else None,
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
                    "research_lab_credit_blocked_receipt_event_failed run_id=%s receipt_id=%s error=%s",
                    compact_ref(context.run_id),
                    compact_ref(context.receipt_id),
                    str(exc)[:240],
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
        await self._ensure_terminal_loop_projection(
            context,
            event_type="loop_paused",
            loop_status="paused",
            reason="blocked_for_credit",
            event_doc=event_doc,
        )
        try:
            await create_ticket_event(
                ticket_id=context.ticket_id,
                event_type="running",
                actor_hotkey=None,
                reason="blocked_for_credit",
                event_doc=event_doc,
            )
            await safe_project_public_loop_activity(
                context.ticket_id,
                source_ref=f"hosted_worker_credit_blocked:{context.run_id}",
                reason="blocked_for_credit",
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_credit_blocked_noncritical_projection_failed run_id=%s error=%s",
                compact_ref(context.run_id),
                str(exc)[:240],
            )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="blocked_for_credit",
            receipt_id=context.receipt_id,
            error=error[:500],
        )

    async def _mark_failed(self, context: HostedRunContext, error: str) -> HostedWorkerOutcome:
        event_doc = {"run_id": context.run_id, "worker_ref": self.worker_ref, "error": error[:500]}
        receipt_id = context.receipt_id
        if receipt_id:
            await create_receipt_event(
                receipt_id=receipt_id,
                ticket_id=context.ticket_id,
                event_type="failed",
                receipt_status="failed",
                event_doc=event_doc,
            )
        else:
            receipt, _event = await create_receipt(self._failed_receipt_request(context, error))
            receipt_id = str(receipt["receipt_id"])
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="failed",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="hosted_research_lab_run_failed",
            event_doc={**event_doc, "receipt_id": receipt_id},
        )
        context.receipt_id = context.receipt_id or receipt_id
        await self._ensure_terminal_loop_projection(
            context,
            event_type="loop_failed",
            loop_status="failed",
            reason="hosted_research_lab_run_failed",
            event_doc={**event_doc, "receipt_id": receipt_id},
        )
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="cancelled",
            actor_hotkey=None,
            reason="hosted_research_lab_run_failed",
            event_doc={**event_doc, "receipt_id": receipt_id},
        )
        await safe_project_public_loop_activity(
            context.ticket_id,
            source_ref=f"hosted_worker_failed:{context.run_id}",
            reason="hosted_research_lab_run_failed",
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
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="queued",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="code_edit_builder_not_ready_requeued",
            event_doc=event_doc,
        )
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
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="queued",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="transient_worker_error_requeued",
            event_doc=event_doc,
        )
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
            loop_start_credit_id=_loop_start_credit_id_from_queue_events(context.queue_events),
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

    def _failed_receipt_request(self, context: HostedRunContext, error: str) -> ResearchLabReceiptCreateRequest:
        budget_context = self._run_budget_context(context)
        return ResearchLabReceiptCreateRequest(
            internal_run_ref=f"research_lab_hosted_worker:{context.run_id}",
            ticket_id=context.ticket_id,
            run_id=context.run_id,
            loop_start_payment_id=context.payment.get("payment_id") if context.payment else None,
            miner_hotkey=str(context.ticket["miner_hotkey"]),
            island=str(context.ticket["island"]),
            receipt_status="failed",
            loop_count=int(context.ticket.get("requested_loop_count") or 1),
            loop_start_credit_id=_loop_start_credit_id_from_queue_events(context.queue_events),
            miner_openrouter_key_ref=_miner_openrouter_key_ref(context),
            provider_usage=self._provider_usage(context),
            cost_ledger={
                "schema_version": "1.0",
                "status": "failed",
                "total_usd": 0.0,
                "budget_context": _redacted_budget_context(budget_context),
            },
            receipt_doc={
                "schema_version": "1.0",
                "run_id": context.run_id,
                "worker_ref": self.worker_ref,
                "failure_reason": error[:500],
                "budget_context": _redacted_budget_context(budget_context),
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
        timeout_seconds: int = 90,
        max_tokens: int = 1800,
    ) -> OpenRouterCallResult:
        if not api_key:
            raise HostedResearchLabWorkerError("OpenRouter key is required for hosted auto-research")
        if not model_id:
            raise HostedResearchLabWorkerError("OpenRouter auto-research model is required")
        body = {
            "model": model_id,
            "messages": list(messages),
            "temperature": 0.2,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "provider": {
                "data_collection": "deny",
                "zdr": True,
            },
        }

        def _call() -> OpenRouterCallResult:
            req = urlrequest.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urlrequest.urlopen(req, timeout=int(timeout_seconds)) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")[:500]
                error = f"OpenRouter candidate generation failed: HTTP {exc.code}: {message}"
                if int(exc.code) == 402 or _is_openrouter_credit_block_message(error):
                    raise OpenRouterCreditBlockedError(error) from exc
                if int(exc.code) in _RETRYABLE_HTTP_CODES:
                    raise RetryableHostedResearchLabWorkerError(error) from exc
                raise HostedResearchLabWorkerError(error) from exc
            except URLError as exc:
                raise RetryableHostedResearchLabWorkerError(
                    f"OpenRouter candidate generation failed: {exc}"
                ) from exc
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
            if not content:
                _raise_openrouter_generation_response_error(
                    decoded,
                    failure="empty candidate-generation content",
                    default_retryable=True,
                )
            usage = decoded.get("usage") if isinstance(decoded.get("usage"), Mapping) else {}
            provider_usage, cost_microusd = _build_openrouter_provider_usage(
                decoded=decoded,
                usage=usage,
                model_id=model_id,
                api_key=api_key,
            )
            return OpenRouterCallResult(
                content=str(content),
                provider_usage=provider_usage,
                cost_microusd=cost_microusd,
            )

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
) -> tuple[dict[str, Any], int]:
    usage_cost_microusd = _usd_to_microusd(_usage_cost_usd(usage))
    response_id = str(decoded.get("id") or "")
    provider_usage: dict[str, Any] = {
        "provider": "openrouter",
        "key_source": "miner_key_ref",
        "response_id": response_id,
        "model": str(decoded.get("model") or model_id),
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
    }
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
    for attempt in range(1, _OPENROUTER_GENERATION_STATS_ATTEMPTS + 1):
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
        if attempt < _OPENROUTER_GENERATION_STATS_ATTEMPTS:
            delay_index = min(attempt - 1, len(_OPENROUTER_GENERATION_STATS_RETRY_DELAYS_SECONDS) - 1)
            time.sleep(_OPENROUTER_GENERATION_STATS_RETRY_DELAYS_SECONDS[delay_index])

    logger.warning(
        "research_lab_openrouter_generation_stats_unavailable response_id=%s status=%s attempts=%s",
        compact_ref(response_id),
        last_status,
        _OPENROUTER_GENERATION_STATS_ATTEMPTS,
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
