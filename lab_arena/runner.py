"""Arena runner: sandbox one bundle on one benchmark input.

A runner follows every running Arena round, claims one pending assignment per
free local slot, downloads and caches the admitted source bytes, materializes
the trusted Python root filesystem, and executes the host-owned agent entrypoint in a fresh gVisor
sandbox for that single ICP, bridges the sandbox's provider requests (plain
HTTP over the worker socket, or the judge shim's operation frames) to the
service's provider endpoint, appends an operational event log, and submits one
small run result through its authenticated API request. It never reports a score, never holds a database
credential, and never chooses a miner or ICP.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import http.server
import json
import os
import re
import secrets
import select
import shutil
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

import httpx

from lab_arena import integrity, contact_policy, intent_details_policy, quality_policy
from lab_arena import scoring_provider_compat, trajectory, validator_events
from lab_arena import contracts, images, lab_arena_checkpoint, leased_images, operations, reviewer_runtime, runtime, runtime_version, scoring, shim, source_bundle, training_input, training_observations
from lab_arena.contracts import ArenaContractError
from lab_arena.broker import (
    provider_authentication_failure_reason, provider_credit_failure_reason,
)
from lab_arena.output import (
    OutputInvalid,
    SUPPORTED_OUTPUT_SCHEMA_VERSIONS,
    output_document_from_bytes,
    output_invalid_reason,
)
from lab_arena.runtime_host import (
    DEFAULT_RUNNER_SOCKET_ROOT, RuntimeHostError, runtime_host_diagnostic,
    runtime_host_private_diagnostic,
)

DEFAULT_MAX_PARALLEL_RUNS = 8  # compatibility for direct test/embedded configurations
MAX_PARALLEL_ENV = "LAB_ARENA_MAX_PARALLEL_RUNS"
DEFAULT_SOCKET_ROOT = str(DEFAULT_RUNNER_SOCKET_ROOT)
AGENT_ENTRYPOINT_PATH = Path(__file__).with_name("agent_entrypoint.py").resolve()
CHECKPOINT_MODULE_PATH = Path(__file__).with_name("lab_arena_checkpoint.py").resolve()
AUTOMATIC_CAPTURE_PATH = Path(__file__).with_name("automatic_capture.py").resolve()
CODEX_MODULE_PATH = Path(__file__).with_name("lab_arena_codex.py").resolve()
WEB_BRIDGE_PATH = Path(__file__).with_name("web_egress_bridge.py").resolve()
MAX_REFUSED_FRAMES = 25  # bound repeated refusals for the same operation/tool
REFUSAL_PROBE_COOLDOWN_SECONDS = 30.0
MAX_BUFFERED_PROVIDER_ERROR_EVENTS = 32
MAX_BUFFERED_DECISION_EVENTS = 64
MAX_TRAINING_CONTROL_FRAMES = 4096
MAX_TRAINING_CONTROL_BYTES = 192 * 1024 * 1024
QUOTA_SNAPSHOT_SCHEMA_VARIANTS = 2
QUOTA_SNAPSHOT_STARTUP_REQUESTS = 2
QUOTA_SNAPSHOT_CACHE_MILLISECONDS = 1000
QUOTA_SNAPSHOT_CACHE_SECONDS = QUOTA_SNAPSHOT_CACHE_MILLISECONDS / 1000.0
# A request on the worker socket is either a length-prefixed operation frame
# (first byte 0x00: the judge shim) or an HTTP request (an ASCII method).
HTTP_FIRST_BYTES = b"GPHDO"
HTTP_ERROR_STATUS = {
    "capability_unavailable": 400,
    "budget_exhausted": 402, "budget_refused": 402,
    "miner_credentials_unavailable": 402, "miner_provider_not_configured": 400,
    "call_refused": 402, "provider_request_refused": 403,
    "provider_unavailable": 502, "broker_unavailable": 503,
    "lease_stale": 409, "call_uncertain": 409,
    "worker_unavailable": 503, "request_too_large": 413,
}
IMAGE_DIGEST_RE = __import__("re").compile(r"^(?:[a-z0-9][a-z0-9._/-]{0,200}@)?sha256:[0-9a-f]{64}$")
REQUIREMENT_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    r"(?:\[[A-Za-z0-9_,.-]{1,255}\])?"
    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]{0,127}"
    r"(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]{0,127})*)?$"
)
MAX_SOCKET_PATH_BYTES = 100
API_TIMEOUT_SECONDS = 30.0
TRAINING_CAPTURE_TIMEOUT_SECONDS = 10.0
TRAINING_OBSERVATIONS_TIMEOUT_SECONDS = 60.0
COMPLETION_RECOVERY_RETRY_SECONDS = (
    2.0, 5.0, 10.0, 20.0, 30.0, 45.0,
    60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 80.0,
)
# Both availability and accounting retries share this elapsed-time bound.
# Reserve one API timeout before each attempt; even a mixed failure sequence
# cannot retain a worker past the minimum lease TTL solely for completion.
COMPLETION_RECOVERY_SECONDS = (
    sum(COMPLETION_RECOVERY_RETRY_SECONDS)
    + (len(COMPLETION_RECOVERY_RETRY_SECONDS) + 1) * API_TIMEOUT_SECONDS
)
SOURCE_DOWNLOAD_RETRY_SECONDS = (0.5, 1.0)
COMPLETION_SIGNATURE_REFRESH_AGE_SECONDS = (
    contracts.REQUEST_TIMESTAMP_WINDOW_SECONDS - int(API_TIMEOUT_SECONDS)
)
PROVIDER_API_TIMEOUT_GRACE_SECONDS = operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS
MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS = max(
    float(operation.timeout_seconds) for operation in operations.OPERATIONS.values()
)
MAX_PROVIDER_API_TIMEOUT_SECONDS = (
    operations.BUDGET_ADMISSION_MAX_SECONDS
    + MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS
    + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
    + PROVIDER_API_TIMEOUT_GRACE_SECONDS
)
RESPONSES_RATE_LIMIT_RETRIES = 2
RESPONSES_RATE_LIMIT_BACKOFF_SECONDS = (20.0, 40.0)
RESPONSES_RATE_LIMIT_MIN_PROVIDER_SECONDS = 30.0
RESPONSES_RATE_LIMIT_JITTER_MILLISECONDS = 2000
DEFAULT_IMAGE_CACHE_MAX_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_IMAGE_CACHE_MAX_ENTRIES = 32
DEFAULT_SOURCE_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_SOURCE_CACHE_MAX_ENTRIES = 64
MAX_REQUIREMENTS_BYTES = 64 * 1024
MAX_REQUIREMENTS = 128
MAX_AGENT_ENTRYPOINT_BYTES = 1024 * 1024
MAX_DEPENDENCY_BYTES = 512 * 1024 * 1024
# pip retains wheel archives and temporary files alongside the installed tree.
MAX_DEPENDENCY_WORKSPACE_BYTES = 1024 * 1024 * 1024
MAX_DEPENDENCY_FILES = 20_000
DEPENDENCY_INSTALL_TIMEOUT_SECONDS = 300
DEPENDENCY_MOUNT_TIMEOUT_SECONDS = 30
_DEPENDENCY_INSTALL_STDERR_TAIL_BYTES = 8 * 1024
_DEPENDENCY_INSTALL_NETWORK_MARKERS = (
    b"ConnectTimeoutError(",
    b"NetworkConnectionError(",
    b"NetworkConnectionError:",
    b"NewConnectionError(",
    b"ProxyError(",
    b"ReadTimeoutError(",
)
_DEPENDENCY_INSTALL_HOST_MARKERS = (
    b"No module named pip",
)
MAX_WORKER_CONNECTIONS = 8
WORKER_SOCKET_READ_TIMEOUT_SECONDS = 10.0
TEMPORARY_HOLD_RETRY_SECONDS = 1.0
SETTLED_MICROUSD_HEADER = "x-leadpoet-settled-microusd"
CALL_IDENTITY_HEADER = "x-leadpoet-call-identity"
MAX_JUDGE_DIAGNOSTIC_CHARS = scoring.MAX_FAILURE_DETAIL_CHARS
_DIAGNOSTIC_URL_QUERY_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://[^\s?#]+)\?[^\s#]*"
)
_DIAGNOSTIC_URL_AUTHORITY_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s?#]+"
)
_DIAGNOSTIC_CREDENTIAL_RE = re.compile(
    r"(?i)((?<![a-z0-9])[\"']?(?:[a-z0-9]+[_-])*(?:api[_-]?key|apikey|"
    r"access[_-]?token|token|authorization|secret|password|private[_-]?key)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)
_DIAGNOSTIC_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_DIAGNOSTIC_KNOWN_TOKEN_RE = re.compile(
    r"(?i)\b(?:sk-|sb_secret)[A-Za-z0-9._-]*"
)
_JUDGE_FAILURE_STAGES = frozenset(
    {"sandbox", "sandbox_output", "scoring_output", "scorer", "provider_call"}
)
_JUDGE_FAILURE_CLASSES = frozenset(
    {
        "judge_timeout",
        "sandbox_output_error",
        "missing_output",
        "scoring_output_invalid",
        "judge_error",
        "provider_unavailable",
    }
)
_PICKUP_PHASES = frozenset({"round_discovery", "claim"})
_PICKUP_FAILURE_REASONS = frozenset({"request_failed", "claim_denied"})
_IDLE_CLAIM_STATUSES = frozenset({
    "no_pending", "no_open_round", "stage_closed", "paused", "no_free_slot", "cache_busy",
    "no_work",
})
_DEFINITE_CLAIM_STATUSES = _IDLE_CLAIM_STATUSES | frozenset({
    "leased", "rejected", "request_id_reused",
})
_ACTIVE_RESCAN_CLAIM_STATUSES = frozenset({"no_pending", "no_free_slot", "paused"})
_CLAIM_AVAILABILITY_CODES = frozenset(
    {
        "arena_store_unavailable",
        "runner_benchmark_eligibility_unavailable",
        "runner_validator_authority_unavailable",
    }
)

_EXECUTION_DIAGNOSTIC_PREFIX = b"LAB_ARENA_EXECUTION_DIAGNOSTIC "
_EXECUTION_DIAGNOSTIC_MAX_BYTES = 256
_CHECKPOINT_TRANSITION_DIAGNOSTIC_MAX_BYTES = 512
_EXECUTION_DIAGNOSTIC_FAILURE_CLASSES = frozenset(
    {"timeout", "runtime_error", "validation_error", "os_error", "other"}
)
_EXECUTION_DIAGNOSTIC_REASONS = frozenset(
    {
        "deadline_or_idle_timeout",
        "saved_dispatch_accounting",
        "operational_block",
        "two_failed_codex_exits",
        "unchanged_exit_limit",
        "invocation_limit",
        "checkpoint_unavailable",
        "output_validation",
        "unexpected",
    }
)
_KNOWN_CLAIM_DENIAL_CODES = frozenset(
    {
        "arena_store_unavailable",
        "declared_parallelism_invalid",
        "hotkey_banned",
        "round_ended",
        "round_mode_mismatch",
        "round_network_mismatch",
        "round_scope_mismatch",
        "round_unknown",
        "runner_benchmark_eligibility_unavailable",
        "runner_hotkey_unregistered",
        "runner_stake_below_minimum",
        "runner_validator_authority_unavailable",
        "runner_validator_required",
        "validator_output_schema_upgrade_required",
        "validator_checkpoint_upgrade_required",
        "signature_invalid",
    }
)
_QUOTA_SNAPSHOT_FAILURE_REASONS = frozenset(
    ("read_cap", "upstream_exception", "invalid_snapshot")
)


def _bounded_http_status(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 100 <= value <= 599 else None


def _known_claim_denial_code(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value in _KNOWN_CLAIM_DENIAL_CODES else None


class RunnerError(RuntimeError):
    """A runner-side failure; the attempt fails closed."""

    def __init__(
        self,
        message: str = "",
        *,
        http_status: Optional[int] = None,
        denial_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.http_status = _bounded_http_status(http_status)
        self.denial_code = _known_claim_denial_code(denial_code)


def _retryable_claim_availability(exc: RunnerError) -> bool:
    """Rescan transport and availability failures, preserving contract denials."""
    if exc.denial_code and exc.denial_code not in _CLAIM_AVAILABILITY_CODES:
        return False
    return (
        exc.http_status is None
        or exc.http_status in (408, 429)
        or exc.http_status >= 500
    )


class AgentDependencyError(RunnerError):
    """The submitted dependency declaration cannot run in the Arena."""

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        # Keep only bounded, redacted operator detail, never raw pip output.
        self.diagnostic_detail = _safe_judge_diagnostic_text(detail or message)


class DependencyInstallInfrastructureError(RunnerError):
    """The host could not complete a submitted dependency installation."""

    def __init__(self, failure_kind: str, *, detail: str = "") -> None:
        if failure_kind not in ("timeout", "network_error", "installer_error"):
            raise ValueError("dependency install failure kind is invalid")
        super().__init__("dependency installation infrastructure failure")
        self.failure_kind = failure_kind
        self.diagnostic_detail = _safe_judge_diagnostic_text(detail or failure_kind)


def _dependency_install_failure_kind(
    failure: OSError | subprocess.TimeoutExpired,
) -> str:
    return "timeout" if isinstance(failure, subprocess.TimeoutExpired) else "installer_error"


def _pip_stderr_tail(path: Path) -> bytes:
    try:
        with path.open("rb") as captured:
            captured.seek(0, os.SEEK_END)
            captured.seek(
                max(0, captured.tell() - _DEPENDENCY_INSTALL_STDERR_TAIL_BYTES)
            )
            return captured.read(_DEPENDENCY_INSTALL_STDERR_TAIL_BYTES)
    except OSError:
        return b""


def _pip_stderr_has_network_marker(path: Path) -> bool:
    tail = _pip_stderr_tail(path)
    return any(marker in tail for marker in _DEPENDENCY_INSTALL_NETWORK_MARKERS)


def _pip_stderr_has_host_marker(path: Path) -> bool:
    tail = _pip_stderr_tail(path)
    return any(marker in tail for marker in _DEPENDENCY_INSTALL_HOST_MARKERS)


def _pip_failure_detail(path: Path) -> str:
    tail = _pip_stderr_tail(path)
    if len(tail) == _DEPENDENCY_INSTALL_STDERR_TAIL_BYTES:
        # Drop a possibly partial first line: its credential label may be cut.
        tail = tail.partition(b"\n")[2]
    redacted = _safe_judge_diagnostic_text(
        tail.decode("utf-8", errors="replace"),
        max_chars=_DEPENDENCY_INSTALL_STDERR_TAIL_BYTES,
    )
    # The final error normally follows pip's resolver/progress messages.
    return redacted[-MAX_JUDGE_DIAGNOSTIC_CHARS:]


def _safe_judge_diagnostic_text(
    value: Any,
    *,
    max_chars: int = MAX_JUDGE_DIAGNOSTIC_CHARS,
) -> str:
    """Return bounded operator diagnostics with common credentials removed."""

    text = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", str(value or ""))
    text = _DIAGNOSTIC_URL_QUERY_RE.sub(r"\1?[redacted]", text)
    text = _DIAGNOSTIC_URL_AUTHORITY_RE.sub(r"\1[redacted]", text)
    text = _DIAGNOSTIC_BEARER_RE.sub("Bearer [redacted]", text)
    text = _DIAGNOSTIC_CREDENTIAL_RE.sub(r"\1[redacted]", text)
    text = _DIAGNOSTIC_KNOWN_TOKEN_RE.sub("[redacted]", text)
    return " ".join(text.split())[:max_chars]


def _log_judge_failure(
    run_id: str,
    *,
    stage: str,
    error_class: str,
    detail: Any = "",
) -> None:
    safe_run_id = _safe_judge_diagnostic_text(run_id, max_chars=128) or "-"
    safe_detail = _safe_judge_diagnostic_text(detail) or "-"
    safe_stage = stage if stage in _JUDGE_FAILURE_STAGES else "unknown"
    safe_error_class = (
        error_class if error_class in _JUDGE_FAILURE_CLASSES else "unknown"
    )
    print(
        "Lab Arena judge failure: "
        f"run_id={safe_run_id} stage={safe_stage} "
        f"error_class={safe_error_class} "
        f"detail={safe_detail}",
        file=sys.stderr,
        flush=True,
    )


_JUDGE_EXHAUSTED_DETAIL_RE = re.compile(
    r"judge_exhausted attempts=([1-9][0-9]?) reason=([a-z_]+) "
    r"(?:fit=absent|origin=exception|"
    r"fit_dims=([a-z_,]+) fit_class=([a-z_]+) fit_reason=([a-z_]+) "
    r"inv_complete=(yes|no|missing) inv_reason=([a-z_]+) "
    r"fetch=(none|[a-z_0-9:,]+))"
)


def _judge_exhausted_summary(detail: Any) -> Optional[str]:
    """Admit only the scorer's fixed labels to the private trajectory."""

    if not isinstance(detail, str) or len(detail) >= scoring.MAX_FAILURE_DETAIL_CHARS:
        return None  # A full-length detail may have been truncated by the scorer.
    match = _JUDGE_EXHAUSTED_DETAIL_RE.fullmatch(detail)
    if match is None or match[2] not in scoring.FAILURE_REASON_CODES:
        return None
    if match[3] is None:
        suffix = "fit=absent" if detail.endswith("fit=absent") else "origin=exception"
    else:
        dimensions = match[3].split(",")
        if dimensions != ["none"] and dimensions != [
            name for name in scoring._PRIVATE_FIT_DIMENSIONS if name in dimensions
        ]:
            return None
        if match[4] not in scoring._PRIVATE_FIT_CLASSES | {"missing"}:
            return None
        if match[5] not in scoring.FAILURE_REASON_CODES | {"missing"}:
            return None
        if match[7] not in scoring.FAILURE_REASON_CODES | {"none", "missing"}:
            return None
        fetch = match[8]
        if fetch != "none":
            parts = fetch.split(",")
            labels = ("ok", *scoring._PRIVATE_FETCH_CLASSES, "other")
            parsed = [re.fullmatch(r"([a-z_]+):([1-9][0-9]?)", part) for part in parts]
            if any(part is None for part in parsed):
                return None
            names = [part[1] for part in parsed if part is not None]
            if names != [name for name in labels if name in names]:
                return None
        suffix = (
            f"fit_dims={match[3]} fit_class={match[4]} fit_reason={match[5]} "
            f"inv_complete={match[6]} inv_reason={match[7]} fetch={fetch}"
        )
    return f"judge_exhausted attempts={int(match[1])} reason={match[2]} {suffix}"


def _log_dependency_failure(
    run_id: str,
    failure: AgentDependencyError | DependencyInstallInfrastructureError,
) -> None:
    safe_run_id = _safe_judge_diagnostic_text(run_id, max_chars=128) or "-"
    kind = (
        failure.failure_kind
        if isinstance(failure, DependencyInstallInfrastructureError)
        else "source_error"
    )
    try:
        print(
            "Lab Arena dependency installation failure: "
            f"run_id={safe_run_id} stage=dependency_install "
            f"failure_kind={kind} detail={failure.diagnostic_detail}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        # Private diagnostics cannot change completion, cleanup, or retries.
        pass


def _execution_diagnostic_from_stderr(stderr: Any) -> Optional[Dict[str, Any]]:
    """Return the last valid untrusted supervisor diagnostic."""

    if not isinstance(stderr, (bytes, bytearray)):
        return None
    latest = None
    for raw_line in bytes(stderr).splitlines(keepends=True):
        if (
            len(raw_line) > _EXECUTION_DIAGNOSTIC_MAX_BYTES
            or not raw_line.endswith(b"\n")
            or b"\r" in raw_line
            or not raw_line.startswith(_EXECUTION_DIAGNOSTIC_PREFIX)
        ):
            continue
        payload = raw_line[len(_EXECUTION_DIAGNOSTIC_PREFIX) : -1]
        try:
            document = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        try:
            canonical = json.dumps(
                document, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        except (TypeError, UnicodeEncodeError):
            continue
        if canonical != payload or type(document.get("schema_version")) is not int:
            continue
        if document["schema_version"] != 1:
            continue
        if set(document) != {
            "schema_version",
            "event",
            "failure_class",
            "reason",
        } or document.get("event") != "supervisor_failure":
            continue
        failure_class = document.get("failure_class")
        reason = document.get("reason")
        if (
            not isinstance(failure_class, str)
            or failure_class not in _EXECUTION_DIAGNOSTIC_FAILURE_CLASSES
            or not isinstance(reason, str)
            or reason not in _EXECUTION_DIAGNOSTIC_REASONS
        ):
            continue
        latest = dict(document)
    return latest


def _checkpoint_transition_from_stderr(stderr: Any) -> Optional[Dict[str, Any]]:
    """Return the last closed, informational checkpoint transition summary."""

    if not isinstance(stderr, (bytes, bytearray)):
        return None
    latest = None
    for raw_line in bytes(stderr).splitlines(keepends=True):
        if (
            len(raw_line) > _CHECKPOINT_TRANSITION_DIAGNOSTIC_MAX_BYTES
            or not raw_line.endswith(b"\n")
            or b"\r" in raw_line
            or not raw_line.startswith(_EXECUTION_DIAGNOSTIC_PREFIX)
        ):
            continue
        payload = raw_line[len(_EXECUTION_DIAGNOSTIC_PREFIX) : -1]
        try:
            document = json.loads(payload.decode("ascii"))
            canonical = json.dumps(
                document, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        except (TypeError, UnicodeDecodeError, ValueError):
            continue
        if canonical != payload:
            continue
        try:
            latest = contracts.validate_checkpoint_transition(document)
        except ArenaContractError:
            continue
    return latest


def _checkpoint_output_sha256(document: Mapping[str, Any]) -> str:
    """Hash the exact model checkpoint envelope without private field projection."""

    payload = json.dumps(
        {"companies": document["companies"]},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _log_execution_diagnostic(
    run_id: str,
    diagnostic: Optional[Mapping[str, Any]],
    result: runtime.SandboxResult,
) -> None:
    """Write one fixed, informational sandbox diagnostic to the private journal."""

    safe_run_id = _safe_judge_diagnostic_text(run_id, max_chars=128) or "-"
    exit_code = result.exit_code
    safe_exit_code = (
        str(exit_code)
        if type(exit_code) is int and -255 <= exit_code <= 255
        else "-"
    )
    fields = [
        "Lab Arena execution diagnostic:",
        f"run_id={safe_run_id}",
        "trust=untrusted" if diagnostic is not None else "trust=host_observed",
        "event=supervisor_failure" if diagnostic is not None else "event=sandbox_outcome",
    ]
    if diagnostic is not None:
        fields.extend(
            (
                f"failure_class={diagnostic['failure_class']}",
                f"reason={diagnostic['reason']}",
            )
        )
    fields.extend(
        (
            f"host_exit_code={safe_exit_code}",
            f"host_timed_out={str(result.timed_out is True).lower()}",
        )
    )
    try:
        print(" ".join(fields), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        # Informational telemetry must never change completion behavior.
        pass


def _log_pickup_failure(
    *,
    phase: str,
    reason: str,
    http_status: Any = None,
    denial_code: Any = None,
) -> None:
    """Log only fixed, bounded pickup diagnostics; never log request data."""

    safe_phase = phase if phase in _PICKUP_PHASES else "unknown"
    safe_reason = reason if reason in _PICKUP_FAILURE_REASONS else "unknown"
    safe_http_status = _bounded_http_status(http_status)
    safe_denial_code = _known_claim_denial_code(denial_code)
    try:
        print(
            "Lab Arena pickup failure: "
            f"phase={safe_phase} reason={safe_reason} "
            f"http_status={safe_http_status if safe_http_status is not None else '-'} "
            f"denial_code={safe_denial_code or '-'}",
            file=sys.stderr,
            flush=True,
        )
    except OSError:
        pass


def _log_quota_snapshot_failure(
    *,
    reason: str,
    read_count: int,
    read_limit: int,
    include_sourcing_cost: bool,
) -> None:
    """Write one payload-free quota diagnostic to the private journal."""

    safe_reason = (
        reason if reason in _QUOTA_SNAPSHOT_FAILURE_REASONS else "invalid_snapshot"
    )
    safe_count = read_count if type(read_count) is int and read_count >= 0 else 0
    safe_limit = read_limit if type(read_limit) is int and read_limit >= 1 else 1
    try:
        print(
            "Lab Arena quota snapshot failure: "
            f"reason={safe_reason} read_count={safe_count} "
            f"read_limit={safe_limit} "
            f"include_sourcing_cost={str(include_sourcing_cost is True).lower()}",
            file=sys.stderr,
            flush=True,
        )
    except (OSError, ValueError):
        # Diagnostics must never change the fail-closed quota result.
        pass


class _HttpResponseDocument(dict):
    """A wire-compatible response mapping with local-only HTTP diagnostics."""

    def __init__(self, payload: Mapping[str, Any], http_status: int) -> None:
        super().__init__(payload)
        self.http_status = _bounded_http_status(http_status)


class SignatureFn(Protocol):
    def __call__(self, message: str) -> str: ...


def _stage_agent_entrypoint(source_path: Path, run_dir: Path, *, filename: str = "agent-entrypoint.py") -> Path:
    """Copy the trusted entrypoint without changing its deployed permissions."""

    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    if filename not in ("agent-entrypoint.py", "web-egress-bridge.py", "lab_arena_checkpoint.py",
                        "lab_arena_codex.py", "automatic_capture.py"):
        raise RunnerError("trusted runtime filename is invalid")
    destination = Path(run_dir) / filename
    source_fd = destination_fd = None
    destination_created = False
    try:
        source_fd = os.open(Path(source_path), source_flags)
        source_info = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_info.st_mode)
            or source_info.st_size <= 0
            or source_info.st_size > MAX_AGENT_ENTRYPOINT_BYTES
        ):
            raise RunnerError("trusted agent entrypoint is not a bounded regular file")
        with os.fdopen(source_fd, "rb") as source_file:
            source_fd = None
            content = source_file.read(MAX_AGENT_ENTRYPOINT_BYTES + 1)
        if len(content) != source_info.st_size:
            raise RunnerError("trusted agent entrypoint changed during staging")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        destination_created = True
        with os.fdopen(destination_fd, "wb") as destination_file:
            destination_fd = None
            if destination_file.write(content) != len(content):
                raise RunnerError("trusted agent entrypoint could not be staged")
            destination_file.flush()
            os.fchmod(destination_file.fileno(), 0o444)
    except RunnerError:
        if destination_created:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise
    except OSError as exc:
        if destination_created:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise RunnerError("trusted agent entrypoint could not be staged safely") from exc
    finally:
        for descriptor in (destination_fd, source_fd):
            if descriptor is not None:
                os.close(descriptor)
    return destination


# ---------------------------------------------------------------------------
# Service client
# ---------------------------------------------------------------------------


class ArenaApiClient(Protocol):
    def claim(self, envelope: Mapping[str, Any]) -> Dict[str, Any]: ...

    def provider(self, run_id: str, lease_token: str, frame: Mapping[str, Any]) -> Dict[str, Any]: ...

    def quota_usage(
        self,
        run_id: str,
        lease_token: str,
        *,
        include_sourcing_cost: bool = False,
    ) -> Dict[str, Any]: ...

    def complete(self, envelope: Mapping[str, Any]) -> Dict[str, Any]: ...

    def trajectory(self, run_id: str, lease_token: str, events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]: ...

    def training_input(self, run_id: str, lease_token: str, document: Mapping[str, Any]) -> Dict[str, Any]: ...

    def training_observations(self, run_id: str, lease_token: str, document: Mapping[str, Any]) -> Dict[str, Any]: ...

    def verify_training_observations(self, run_id: str, lease_token: str, payload_sha256: str) -> bool: ...

    def training_observation_segment(self, run_id: str, lease_token: str, document: Mapping[str, Any]) -> Dict[str, Any]: ...

    def source(self, run_id: str, lease_token: str) -> bytes: ...

    def image_access(self, run_id: str, lease_token: str) -> Dict[str, Any]: ...

    def current(self) -> Dict[str, Any]: ...

    def round(self, round_id: str) -> Dict[str, Any]: ...


class HttpArenaApiClient:
    """HTTPS client for ``/arena/v1`` runner endpoints (section 14.3)."""

    def __init__(self, base_url: str, *, client: Optional[httpx.Client] = None) -> None:
        try:
            parsed = urlsplit(base_url)
            hostname = parsed.hostname
            parsed.port  # force validation of a malformed port
        except ValueError:
            hostname = None
            parsed = urlsplit("")
        secure = parsed.scheme == "https" and bool(hostname)
        loopback = parsed.scheme == "http" and hostname in ("localhost", "127.0.0.1", "::1")
        if (not secure and not loopback) or parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise RunnerError("Arena API base URL must be https (or loopback for tests)")
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(http1=True, http2=False, follow_redirects=False, timeout=httpx.Timeout(API_TIMEOUT_SECONDS), trust_env=False)

    def _post(
        self,
        path: str,
        document: Mapping[str, Any],
        *,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = API_TIMEOUT_SECONDS,
        preserve_http_status: bool = False,
    ) -> Dict[str, Any]:
        try:
            response = self._client.post(
                self._base_url + path,
                content=contracts.canonical_json(document).encode("utf-8"),
                headers={"content-type": "application/json", **(headers or {})},
                timeout=httpx.Timeout(float(timeout_seconds)),
            )
        except httpx.HTTPError as exc:
            response = getattr(exc, "response", None)
            raise RunnerError(
                "Arena API transport failure: %s" % type(exc).__name__,
                http_status=getattr(response, "status_code", None),
            ) from exc
        if response.status_code >= 500:
            failure_payload = None
            failure_body = getattr(response, "content", None)
            if (
                preserve_http_status
                and isinstance(failure_body, bytes)
                and len(failure_body) <= 4096
            ):
                try:
                    failure_payload = response.json()
                except (RecursionError, ValueError):
                    pass
            denial_code = (
                _known_claim_denial_code(failure_payload.get("code"))
                if isinstance(failure_payload, Mapping)
                else None
            )
            raise RunnerError(
                "Arena API failed: HTTP %d" % response.status_code,
                http_status=response.status_code,
                denial_code=denial_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RunnerError(
                "Arena API returned non-JSON",
                http_status=response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise RunnerError(
                "Arena API returned a non-object",
                http_status=response.status_code,
            )
        if response.status_code >= 400 and "status" not in payload:
            payload = {"status": "rejected", "http_status": response.status_code, "detail": payload.get("detail")}
        if preserve_http_status:
            return _HttpResponseDocument(payload, response.status_code)
        return payload

    def claim(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post(
            "/arena/v1/runs/claim",
            envelope,
            preserve_http_status=True,
        )

    def trajectory(
        self, run_id: str, lease_token: str,
        events: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9._:-]{1,200}", run_id
        ):
            raise RunnerError("run trajectory is unavailable")
        return self._post(
            "/arena/v1/runs/%s/trajectory" % run_id,
            {"events": list(events)},
            headers={"x-lab-arena-lease": lease_token},
            timeout_seconds=5.0,
        )

    def training_input(
        self, run_id: str, lease_token: str, document: Mapping[str, Any],
    ) -> Dict[str, Any]:
        # Generation and claim identity are taken only from the gateway row.
        training_input.object_ref(run_id, 1)  # Validate the path vocabulary.
        if len(contracts.canonical_json(document).encode("utf-8")) > training_input.MAX_INPUT_BYTES:
            raise RunnerError("training input exceeds capture limit")
        return self._post(
            "/arena/v1/runs/%s/training-input" % run_id, document,
            headers={"x-lab-arena-lease": lease_token},
            timeout_seconds=TRAINING_CAPTURE_TIMEOUT_SECONDS,
            preserve_http_status=True,
        )

    def training_observations(
        self, run_id: str, lease_token: str, document: Mapping[str, Any],
    ) -> Dict[str, Any]:
        training_input.object_ref(run_id, 1)
        if len(contracts.canonical_json(document).encode("utf-8")) > training_observations.MAX_OBJECT_BYTES:
            raise RunnerError("training observations exceed capture limit")
        return self._post(
            "/arena/v1/runs/%s/training-observations" % run_id, document,
            headers={"x-lab-arena-lease": lease_token},
            timeout_seconds=TRAINING_OBSERVATIONS_TIMEOUT_SECONDS,
            preserve_http_status=True,
        )

    def verify_training_observations(self, run_id: str, lease_token: str, payload_sha256: str) -> bool:
        training_input.object_ref(run_id, 1)
        if not isinstance(payload_sha256, str) or not contracts.SHA256_RE.fullmatch(payload_sha256):
            raise RunnerError("training observations digest is invalid")
        etag = '"' + payload_sha256 + '"'
        try:
            response = self._client.head(
                self._base_url + "/arena/v1/runs/%s/training-observations" % run_id,
                headers={"x-lab-arena-lease": lease_token, "If-Match": etag},
                timeout=httpx.Timeout(TRAINING_OBSERVATIONS_TIMEOUT_SECONDS),
            )
        except httpx.HTTPError as exc:
            raise RunnerError("training observations verification transport failure") from exc
        return response.status_code == 204 and response.headers.get("ETag") == etag

    def training_observation_segment(
        self, run_id: str, lease_token: str, document: Mapping[str, Any],
    ) -> Dict[str, Any]:
        training_input.object_ref(run_id, 1)
        if len(contracts.canonical_json(document).encode("utf-8")) > training_observations.MAX_SEGMENT_BYTES:
            raise RunnerError("training observation segment exceeds capture limit")
        return self._post(
            "/arena/v1/runs/%s/training-observation-segments" % run_id, document,
            headers={"x-lab-arena-lease": lease_token},
            timeout_seconds=10.0,
            preserve_http_status=True,
        )

    def provider(
        self, run_id: str, lease_token: str, frame: Mapping[str, Any],
        *, timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        requested_timeout = frame.get("timeout_ms")
        api_timeout_seconds = API_TIMEOUT_SECONDS
        operation_id = frame.get("operation_id")
        operation = (
            operations.get_operation(operation_id)
            if isinstance(operation_id, str)
            else None
        )
        if (
            operation is not None
            and isinstance(requested_timeout, int)
            and not isinstance(requested_timeout, bool)
        ):
            operation_timeout = min(
                requested_timeout / 1000.0,
                float(operation.timeout_seconds),
            )
            api_timeout_seconds = max(
                API_TIMEOUT_SECONDS,
                min(
                    MAX_PROVIDER_API_TIMEOUT_SECONDS,
                    operations.BUDGET_ADMISSION_MAX_SECONDS
                    + operation_timeout
                    + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                    + PROVIDER_API_TIMEOUT_GRACE_SECONDS,
                ),
            )
        if timeout_seconds is not None:
            api_timeout_seconds = min(api_timeout_seconds, max(0.001, timeout_seconds))
        return self._post(
            "/arena/v1/runs/%s/provider" % run_id,
            frame,
            headers={"x-lab-arena-lease": lease_token},
            timeout_seconds=api_timeout_seconds,
        )

    def signal_judgment(self, run_id: str, lease_token: str, request: Mapping[str, Any]) -> Dict[str, Any]:
        from lab_arena.signal_cache_control import validate_request
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", run_id):
            raise RunnerError("invalid signal cache run")
        return self._post(
            "/arena/v1/runs/%s/signal-judgment" % run_id, validate_request(request),
            headers={"x-lab-arena-lease": lease_token}, timeout_seconds=30.0,
        )

    def quota_usage(
        self,
        run_id: str,
        lease_token: str,
        *,
        include_sourcing_cost: bool = False,
    ) -> Dict[str, Any]:
        """Read one bounded active-lease quota snapshot."""

        if not isinstance(run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9._:-]{1,200}", run_id
        ):
            raise RunnerError("run quota is unavailable")
        try:
            with self._client.stream(
                "GET",
                self._base_url
                + "/arena/v1/runs/%s/quota" % run_id
                + (
                    "?include_sourcing_cost=true"
                    if include_sourcing_cost
                    else ""
                ),
                headers={"x-lab-arena-lease": lease_token},
                timeout=httpx.Timeout(API_TIMEOUT_SECONDS),
            ) as response:
                if response.status_code != 200:
                    raise RunnerError("run quota is unavailable")
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        if int(declared) > lab_arena_checkpoint.MAX_QUOTA_RESPONSE_BYTES:
                            raise RunnerError("run quota is unavailable")
                    except ValueError:
                        raise RunnerError("run quota is unavailable") from None
                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > lab_arena_checkpoint.MAX_QUOTA_RESPONSE_BYTES:
                        raise RunnerError("run quota is unavailable")
                    chunks.append(chunk)
        except httpx.HTTPError:
            raise RunnerError("run quota is unavailable") from None
        try:
            document = json.loads(b"".join(chunks).decode("utf-8"))
            return (
                lab_arena_checkpoint.validate_quota_cost_snapshot(document)
                if include_sourcing_cost
                else lab_arena_checkpoint.validate_quota_snapshot(document)
            )
        except (UnicodeDecodeError, ValueError, lab_arena_checkpoint.QuotaUnavailable):
            raise RunnerError("run quota is unavailable") from None

    def complete(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        result = self._post(
            "/arena/v1/runs/%s/complete" % envelope["body"]["run_id"],
            envelope, preserve_http_status=True,
        )
        if result.http_status in (408, 429):
            raise RunnerError(
                "Arena completion unavailable: HTTP %d" % result.http_status,
                http_status=result.http_status,
            )
        if result.http_status is not None and result.http_status >= 400:
            result["status"] = "rejected"
        return result

    def source(self, run_id: str, lease_token: str) -> bytes:
        """Download one bounded source archive under its active run lease."""

        retry_delays = iter(SOURCE_DOWNLOAD_RETRY_SECONDS)
        while True:
            # Discard all bytes and size accounting from a disconnected response.
            chunks = []
            total = 0
            try:
                with self._client.stream(
                    "GET",
                    self._base_url + "/arena/v1/runs/%s/source" % run_id,
                    headers={"x-lab-arena-lease": lease_token},
                    timeout=httpx.Timeout(API_TIMEOUT_SECONDS),
                ) as response:
                    if response.status_code != 200:
                        raise RunnerError(
                            "run source is unavailable: HTTP %d" % response.status_code
                        )
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            if int(declared) > source_bundle.MAX_SOURCE_ARCHIVE_BYTES:
                                raise RunnerError("run source exceeds the archive limit")
                        except ValueError as exc:
                            raise RunnerError("run source length is invalid") from exc
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > source_bundle.MAX_SOURCE_ARCHIVE_BYTES:
                            raise RunnerError("run source exceeds the archive limit")
                        chunks.append(chunk)
                return b"".join(chunks)
            except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                try:
                    delay = next(retry_delays)
                except StopIteration:
                    raise RunnerError("Arena API transport failure: %s" % type(exc).__name__) from exc
                time.sleep(delay)
            except httpx.HTTPError as exc:
                raise RunnerError("Arena API transport failure: %s" % type(exc).__name__) from exc

    def image_access(self, run_id: str, lease_token: str) -> Dict[str, Any]:
        """Fetch one bounded, transient scorer-image access document."""

        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", run_id):
            raise RunnerError("run id is invalid")
        try:
            with self._client.stream(
                "GET",
                self._base_url + "/arena/v1/runs/%s/image-access" % run_id,
                headers={"x-lab-arena-lease": lease_token},
                timeout=httpx.Timeout(API_TIMEOUT_SECONDS),
            ) as response:
                if response.status_code != 200:
                    raise RunnerError(
                        "run image access is unavailable: HTTP %d"
                        % response.status_code,
                        http_status=response.status_code,
                    )
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError:
                        raise RunnerError("run image access length is invalid") from None
                    if declared_size < 0 or declared_size > leased_images.MAX_IMAGE_ACCESS_DOCUMENT_BYTES:
                        raise RunnerError("run image access exceeds the document limit")
                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > leased_images.MAX_IMAGE_ACCESS_DOCUMENT_BYTES:
                        raise RunnerError("run image access exceeds the document limit")
                    chunks.append(chunk)
        except RunnerError:
            raise
        except httpx.HTTPError:
            # The lease token and returned signed URLs are bearer capabilities.
            raise RunnerError("Arena API image access transport failure") from None
        try:
            payload = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise RunnerError("Arena API returned invalid image access JSON") from None
        if not isinstance(payload, dict):
            raise RunnerError("Arena API returned a non-object image access document")
        if set(payload) != {
            "schema_version",
            "image_reference",
            "image_digest",
            "manifest_b64",
            "manifest_media_type",
            "blobs",
        } or payload.get("schema_version") != leased_images.IMAGE_ACCESS_SCHEMA_VERSION:
            raise RunnerError("Arena API returned an invalid image access document")
        return payload

    def round(self, round_id: str) -> Dict[str, Any]:
        return self._get("/arena/v1/rounds/%s" % round_id, "round %s" % round_id)

    def current(self) -> Dict[str, Any]:
        return self._get("/arena/v1/current", "current round")

    def _get(self, path: str, what: str) -> Dict[str, Any]:
        try:
            response = self._client.get(self._base_url + path)
        except httpx.HTTPError as exc:
            response = getattr(exc, "response", None)
            raise RunnerError(
                "Arena API transport failure: %s" % type(exc).__name__,
                http_status=getattr(response, "status_code", None),
            ) from exc
        if response.status_code != 200:
            raise RunnerError(
                "%s is unavailable: HTTP %d" % (what, response.status_code),
                http_status=response.status_code,
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RunnerError(
                "Arena API returned a non-object",
                http_status=response.status_code,
            )
        return payload

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Runner identity and image cache
# ---------------------------------------------------------------------------


@dataclass
class RunnerIdentity:
    hotkey: str
    sign: SignatureFn
    coldkey_owned_hotkeys: Sequence[str] = ()

    def __post_init__(self) -> None:
        contracts.require_hotkey(self.hotkey)


class ImageExporter(Protocol):
    """Populate ``target_dir/rootfs`` with the image named by reference and pinned by digest."""

    def __call__(self, image_reference: str, image_digest: str, target_dir: Path) -> None: ...


def _cache_path_bytes(path: Path) -> int:
    """Return allocated bytes once per inode without following symlinks."""

    total = 0
    seen = set()
    for directory, names, files in os.walk(path, followlinks=False):
        for name in list(names) + list(files):
            candidate = Path(directory) / name
            try:
                stat_result = candidate.lstat()
            except OSError:
                continue
            identity = (int(stat_result.st_dev), int(stat_result.st_ino))
            if identity in seen:
                continue
            seen.add(identity)
            blocks = int(getattr(stat_result, "st_blocks", 0)) * 512
            total += blocks if blocks > 0 else int(stat_result.st_size)
    return total


def _remove_cache_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    shutil.rmtree(path, ignore_errors=True)


class ImageCache:
    """A small LRU of materialized images; active root filesystems stay pinned."""

    def __init__(
        self,
        root: Path,
        exporter: ImageExporter,
        *,
        max_bytes: int = DEFAULT_IMAGE_CACHE_MAX_BYTES,
        max_entries: int = DEFAULT_IMAGE_CACHE_MAX_ENTRIES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise RunnerError("image cache byte limit is invalid")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise RunnerError("image cache entry limit is invalid")
        self._root = Path(root)
        self._exporter = exporter
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._ready: "OrderedDict[str, Path]" = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._in_use: Dict[str, int] = {}
        self._root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._load_existing_locked()
            if not self._ready:
                # An empty cache guarantees a cold export. Check its existing
                # space requirement before the runner can claim any leases.
                self._reserve_export_space_locked()

    def _load_existing_locked(self) -> None:
        existing = []
        for target in sorted(self._root.glob("sha256-*"), key=lambda item: item.name):
            marker = target / ".exported"
            try:
                digest = marker.read_text(encoding="utf-8").strip()
                marker_time = marker.stat().st_mtime_ns
            except (OSError, UnicodeError):
                _remove_cache_path(target)
                continue
            expected_name = "sha256-" + digest.rsplit("sha256:", 1)[-1]
            rootfs = target / "rootfs"
            if not IMAGE_DIGEST_RE.match(digest) or target.name != expected_name or rootfs.is_symlink() or not rootfs.is_dir():
                _remove_cache_path(target)
                continue
            existing.append((marker_time, target.name, digest, rootfs, _cache_path_bytes(target)))
        for _time, _name, digest, rootfs, size in sorted(existing):
            self._ready[digest] = rootfs
            self._sizes[digest] = size
        self._evict_locked()

    def _within_limits_locked(self) -> bool:
        return len(self._ready) <= self._max_entries and sum(self._sizes.values()) <= self._max_bytes

    def _evict_locked(self, *, protected: Sequence[str] = ()) -> None:
        protected_set = set(protected)
        while not self._within_limits_locked():
            victim = next(
                (digest for digest in self._ready if digest not in protected_set and self._in_use.get(digest, 0) == 0),
                None,
            )
            if victim is None:
                return
            rootfs = self._ready.pop(victim)
            self._sizes.pop(victim, None)
            self._in_use.pop(victim, None)
            _remove_cache_path(rootfs.parent)

    def _reserve_export_space_locked(self) -> None:
        # A cold export holds the compressed layer on disk while extracting it.
        # Make room before downloading, since ordinary cache eviction happens
        # only after a successful export.
        required = min(self._max_bytes, images.DEFAULT_MAX_ROOTFS_BYTES) + images.DEFAULT_MAX_IMAGE_BYTES
        try:
            free = shutil.disk_usage(self._root).free
        except OSError as exc:
            raise RunnerError("image cache disk space could not be checked") from exc
        while free < required:
            victim = next((digest for digest in self._ready if self._in_use.get(digest, 0) == 0), None)
            if victim is None:
                raise RunnerError("insufficient disk space for image export")
            rootfs = self._ready[victim]
            target = rootfs.parent
            _remove_cache_path(target)
            if target.exists() or target.is_symlink():
                raise RunnerError("image cache eviction failed")
            self._ready.pop(victim)
            self._sizes.pop(victim, None)
            self._in_use.pop(victim, None)
            try:
                free = shutil.disk_usage(self._root).free
            except OSError as exc:
                raise RunnerError("image cache disk space could not be checked") from exc

    def _rootfs_for_locked(
        self,
        image_digest: str,
        image_reference: str,
        exporter: Optional[ImageExporter] = None,
    ) -> Path:
        path = self._ready.get(image_digest)
        if path is not None and path.is_dir() and not path.is_symlink():
            self._ready.move_to_end(image_digest)
            return path
        if path is not None:
            self._ready.pop(image_digest, None)
            self._sizes.pop(image_digest, None)
            self._in_use.pop(image_digest, None)
        # Cache directories are keyed by the content digest alone.
        target = self._root / ("sha256-" + image_digest.rsplit("sha256:", 1)[1])
        if target.exists() or target.is_symlink():
            _remove_cache_path(target)
        self._reserve_export_space_locked()
        target.mkdir(parents=True)
        try:
            (exporter or self._exporter)(image_reference, image_digest, target)
            rootfs = target / "rootfs"
            if rootfs.is_symlink() or not rootfs.is_dir():
                raise RunnerError("image exporter produced no root filesystem")
            (target / ".exported").write_text(image_digest, encoding="utf-8")
            size = _cache_path_bytes(target)
            if size > self._max_bytes:
                raise RunnerError("image exceeds runner cache capacity")
        except Exception:
            _remove_cache_path(target)
            raise
        self._ready[image_digest] = rootfs
        self._sizes[image_digest] = size
        self._evict_locked(protected=(image_digest,))
        if not self._within_limits_locked():
            self._ready.pop(image_digest, None)
            self._sizes.pop(image_digest, None)
            _remove_cache_path(target)
            raise RunnerError("image cache capacity is in use")
        return rootfs

    def rootfs_for(
        self,
        image_digest: str,
        image_reference: str = "",
        *,
        exporter: Optional[ImageExporter] = None,
    ) -> Path:
        if not isinstance(image_digest, str) or not IMAGE_DIGEST_RE.match(image_digest):
            raise RunnerError("image digest is invalid")
        with self._lock:
            return self._rootfs_for_locked(image_digest, image_reference, exporter)

    @contextmanager
    def acquire(
        self,
        image_digest: str,
        image_reference: str = "",
        *,
        exporter: Optional[ImageExporter] = None,
    ) -> Iterator[Path]:
        """Pin one cached rootfs until the caller's sandbox has stopped."""

        if not isinstance(image_digest, str) or not IMAGE_DIGEST_RE.match(image_digest):
            raise RunnerError("image digest is invalid")
        with self._lock:
            rootfs = self._rootfs_for_locked(image_digest, image_reference, exporter)
            self._in_use[image_digest] = self._in_use.get(image_digest, 0) + 1
        try:
            yield rootfs
        finally:
            with self._lock:
                remaining = self._in_use.get(image_digest, 0) - 1
                if remaining > 0:
                    self._in_use[image_digest] = remaining
                else:
                    self._in_use.pop(image_digest, None)
                self._evict_locked()


class SourceFetcher(Protocol):
    def __call__(self, run_id: str, lease_token: str) -> bytes: ...


class DependencyInstaller(Protocol):
    def __call__(self, requirements_path: Path, target_dir: Path) -> None: ...


def _validated_requirements(path: Path) -> List[str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AgentDependencyError("requirements.txt is unreadable") from exc
    if len(data) > MAX_REQUIREMENTS_BYTES:
        raise AgentDependencyError("requirements.txt exceeds the size limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentDependencyError("requirements.txt must be UTF-8") from exc
    requirements = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        if not REQUIREMENT_RE.fullmatch(line):
            raise AgentDependencyError(
                "requirements.txt may contain only package names and version constraints"
            )
        requirements.append(line)
        if len(requirements) > MAX_REQUIREMENTS:
            raise AgentDependencyError("requirements.txt has too many packages")
    return requirements


def install_binary_requirements(requirements_path: Path, target_dir: Path) -> None:
    """Install wheels in bounded scratch space, then enforce final tree limits."""

    requirements = _validated_requirements(requirements_path)
    if not requirements:
        return
    target = Path(target_dir)
    if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
        raise RunnerError("dependency target must be an empty directory")
    staging = Path(tempfile.mkdtemp(prefix="lab-arena-deps-"))
    mounted = False
    unmounted = False
    failure: Optional[Exception] = None
    try:
        mount = subprocess.run(
            [
                "mount",
                "-t",
                "tmpfs",
                "-o",
                "size=%d,mode=0700,nosuid,nodev,noexec" % MAX_DEPENDENCY_WORKSPACE_BYTES,
                "tmpfs",
                str(staging),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DEPENDENCY_MOUNT_TIMEOUT_SECONDS,
            check=False,
        )
        if mount.returncode != 0:
            raise RunnerError("dependency filesystem mount failed")
        mounted = True
        install_target = staging / "target"
        install_target.mkdir(mode=0o700)
        home = staging / "home"
        home.mkdir(mode=0o700)
        temporary = staging / "tmp"
        temporary.mkdir(mode=0o700)
        environment = {
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(temporary),
        }
        command = [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--isolated",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "--no-compile",
            "--only-binary=:all:",
            "--index-url=https://pypi.org/simple",
            "--target",
            str(install_target),
            "--requirement",
            str(requirements_path),
        ]
        for install_attempt in range(2):
            if install_attempt:
                for retry_path in (install_target, temporary):
                    _remove_cache_path(retry_path)
                    if retry_path.exists() or retry_path.is_symlink():
                        raise DependencyInstallInfrastructureError(
                            "installer_error"
                        )
                    try:
                        retry_path.mkdir(mode=0o700)
                    except OSError as exc:
                        raise DependencyInstallInfrastructureError(
                            "installer_error"
                        ) from exc
            try:
                stderr_path = temporary / "pip-stderr.log"
                with stderr_path.open("wb") as captured_stderr:
                    result = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=captured_stderr,
                        env=environment,
                        timeout=DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
                        check=False,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                if install_attempt == 0:
                    continue
                raise DependencyInstallInfrastructureError(
                    _dependency_install_failure_kind(exc),
                    detail=_pip_failure_detail(stderr_path),
                ) from exc
            if result.returncode != 0:
                if _pip_stderr_has_host_marker(stderr_path):
                    raise DependencyInstallInfrastructureError(
                        "installer_error", detail=_pip_failure_detail(stderr_path)
                    )
                if not _pip_stderr_has_network_marker(stderr_path):
                    # pip maps its installation and network exceptions to the
                    # same exit code. Unclassified failures remain source-owned.
                    raise AgentDependencyError(
                        "binary dependency installation failed",
                        detail=_pip_failure_detail(stderr_path),
                    )
                if install_attempt == 0:
                    continue
                raise DependencyInstallInfrastructureError(
                    "network_error", detail=_pip_failure_detail(stderr_path)
                )
            break
        _lock_down_dependency_tree(install_target, site="install_staging")
        for child in install_target.iterdir():
            destination = target / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)
    except (OSError, subprocess.TimeoutExpired) as exc:
        failure = (
            AgentDependencyError("binary dependency installation failed")
            if mounted
            else RunnerError("dependency filesystem mount failed")
        )
        failure.__cause__ = exc
    except Exception as exc:
        failure = exc
    finally:
        if mounted:
            try:
                result = subprocess.run(
                    ["umount", str(staging)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=DEPENDENCY_MOUNT_TIMEOUT_SECONDS,
                    check=False,
                )
                unmounted = result.returncode == 0
                if not unmounted:
                    result = subprocess.run(
                        ["umount", "-l", str(staging)],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=DEPENDENCY_MOUNT_TIMEOUT_SECONDS,
                        check=False,
                    )
                    unmounted = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                unmounted = False
            if not unmounted:
                cleanup_failure = RunnerError("dependency filesystem cleanup failed")
                if failure is not None:
                    cleanup_failure.__cause__ = failure
                failure = cleanup_failure
        if not mounted or unmounted:
            shutil.rmtree(staging, ignore_errors=True)
    if failure is not None:
        raise failure


def _lock_down_dependency_tree(path: Path, *, site: str = "dependency_tree") -> None:
    regular_bytes = 0
    directory_bytes = 0
    count = 0
    directories = []
    for directory, names, files in os.walk(path, topdown=True, followlinks=False):
        directory_path = Path(directory)
        directories.append(directory_path)
        for name in list(names) + list(files):
            candidate = directory_path / name
            try:
                details = candidate.lstat()
            except OSError as exc:
                raise RunnerError("installed dependency is unreadable") from exc
            if not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode)):
                raise AgentDependencyError("installed dependency has an unsafe type")
            count += 1
            # Directory metadata varies by filesystem. Bound payload bytes
            # separately; directories still consume the existing entry limit.
            if stat.S_ISREG(details.st_mode):
                regular_bytes += int(details.st_size)
            else:
                directory_bytes += int(details.st_size)
            if count > MAX_DEPENDENCY_FILES or regular_bytes > MAX_DEPENDENCY_BYTES:
                threshold = "entries" if count > MAX_DEPENDENCY_FILES else "regular_bytes"
                raise AgentDependencyError(
                    "installed dependencies exceed the cache limit",
                    detail=(
                        f"dependency_limit site={site} threshold={threshold} "
                        f"entries_seen={count} regular_bytes={regular_bytes} "
                        f"directory_bytes={directory_bytes} "
                        f"max_entries={MAX_DEPENDENCY_FILES} "
                        f"max_regular_bytes={MAX_DEPENDENCY_BYTES}"
                    ),
                )
            if stat.S_ISREG(details.st_mode):
                os.chmod(candidate, 0o444)
    for directory in reversed(directories):
        os.chmod(directory, 0o555)


class SourceCache:
    """A bounded LRU keyed by the existing server-assigned submission id."""

    def __init__(
        self,
        root: Path,
        fetcher: SourceFetcher,
        *,
        dependency_installer: DependencyInstaller = install_binary_requirements,
        max_bytes: int = DEFAULT_SOURCE_CACHE_MAX_BYTES,
        max_entries: int = DEFAULT_SOURCE_CACHE_MAX_ENTRIES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise RunnerError("source cache byte limit is invalid")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise RunnerError("source cache entry limit is invalid")
        self._root = Path(root)
        self._fetcher = fetcher
        self._dependency_installer = dependency_installer
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._building: set[str] = set()
        self._ready: "OrderedDict[str, Tuple[Path, Path]]" = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._in_use: Dict[str, int] = {}
        self._root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._load_existing_locked()

    def _load_existing_locked(self) -> None:
        existing = []
        for target in sorted(self._root.glob("submission-*"), key=lambda item: item.name):
            marker = target / ".ready"
            archive = target / "source.tar.gz"
            source = target / "source"
            dependencies = target / "deps"
            try:
                marker_time = marker.stat().st_mtime_ns
                archive_size = archive.stat().st_size
            except OSError:
                _remove_cache_path(target)
                continue
            submission_id = target.name.removeprefix("submission-")
            if (
                not contracts.SUBMISSION_ID_RE.fullmatch(submission_id)
                or target.name != "submission-" + submission_id
                or not 1 <= archive_size <= source_bundle.MAX_SOURCE_ARCHIVE_BYTES
                or source.is_symlink()
                or not source.is_dir()
                or dependencies.is_symlink()
                or not dependencies.is_dir()
            ):
                _remove_cache_path(target)
                continue
            try:
                facts = source_bundle.validate_source_archive(archive.read_bytes())
            except (OSError, source_bundle.SourceBundleError):
                _remove_cache_path(target)
                continue
            if facts["source_size_bytes"] != archive_size:
                _remove_cache_path(target)
                continue
            existing.append(
                (marker_time, target.name, submission_id, source, dependencies, _cache_path_bytes(target))
            )
        for _time, _name, submission_id, source, dependencies, size in sorted(existing):
            self._ready[submission_id] = (source, dependencies)
            self._sizes[submission_id] = size
        self._evict_locked()

    def _within_limits_locked(self) -> bool:
        return len(self._ready) <= self._max_entries and sum(self._sizes.values()) <= self._max_bytes

    def _evict_locked(self, *, protected: Sequence[str] = ()) -> None:
        protected_set = set(protected)
        while not self._within_limits_locked():
            victim = next(
                (submission_id for submission_id in self._ready if submission_id not in protected_set and self._in_use.get(submission_id, 0) == 0),
                None,
            )
            if victim is None:
                return
            source, _dependencies = self._ready.pop(victim)
            self._sizes.pop(victim, None)
            self._in_use.pop(victim, None)
            _remove_cache_path(source.parent)

    def _cached_source_locked(
        self,
        submission_id: str,
        source_size_bytes: int,
    ) -> Optional[Tuple[Path, Path]]:
        cached = self._ready.get(submission_id)
        if cached is None:
            return None
        archive = cached[0].parent / "source.tar.gz"
        if archive.is_file() and archive.stat().st_size == source_size_bytes:
            self._ready.move_to_end(submission_id)
            return cached
        self._ready.pop(submission_id, None)
        self._sizes.pop(submission_id, None)
        _remove_cache_path(cached[0].parent)
        return None

    def _prepare_source(
        self,
        run_id: str,
        lease_token: str,
        source_ref: str,
        submission_id: str,
        source_size_bytes: int,
    ) -> Tuple[Tuple[Path, Path], int]:
        target = self._root / ("submission-" + submission_id)
        if target.exists() or target.is_symlink():
            _remove_cache_path(target)
        target.mkdir(parents=True, mode=0o700)
        archive_path = target / "source.tar.gz"
        source_path = target / "source"
        dependency_path = target / "deps"
        try:
            payload = bytes(self._fetcher(run_id, lease_token))
            if len(payload) != source_size_bytes:
                raise RunnerError("run source size does not match its lease")
            facts = source_bundle.validate_source_archive(payload)
            if facts["source_size_bytes"] != source_size_bytes:
                raise RunnerError("run source size does not match its lease")
            archive_path.write_bytes(payload)
            os.chmod(archive_path, 0o400)
            source_path.mkdir(mode=0o700)
            source_bundle.extract_source_archive(payload, source_path)
            dependency_path.mkdir(mode=0o700)
            requirements = source_path / "requirements.txt"
            if requirements.is_file():
                self._dependency_installer(requirements, dependency_path)
            _lock_down_dependency_tree(dependency_path, site="source_cache")
            (target / ".ready").touch(mode=0o400)
            size = _cache_path_bytes(target)
            if size > self._max_bytes:
                raise RunnerError("source exceeds runner cache capacity")
        except source_bundle.SourceBundleError as exc:
            _remove_cache_path(target)
            raise RunnerError("run source archive is invalid: %s" % exc.code) from exc
        except Exception:
            _remove_cache_path(target)
            raise
        return (source_path, dependency_path), size

    @contextmanager
    def acquire(
        self,
        run_id: str,
        lease_token: str,
        source_ref: str,
        submission_id: str,
        source_size_bytes: int,
    ) -> Iterator[Tuple[Path, Path]]:
        if not isinstance(source_ref, str) or not source_ref or len(source_ref) > 1024:
            raise RunnerError("source ref is invalid")
        if not isinstance(submission_id, str) or not contracts.SUBMISSION_ID_RE.fullmatch(submission_id):
            raise RunnerError("submission id is invalid")
        if (
            isinstance(source_size_bytes, bool)
            or not isinstance(source_size_bytes, int)
            or not 1 <= source_size_bytes <= source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        ):
            raise RunnerError("source size is invalid")
        must_prepare = False
        with self._condition:
            while submission_id in self._building:
                self._condition.wait()
            paths = self._cached_source_locked(submission_id, source_size_bytes)
            if paths is None:
                self._building.add(submission_id)
                must_prepare = True
            else:
                self._in_use[submission_id] = self._in_use.get(submission_id, 0) + 1
        if must_prepare:
            try:
                paths, size = self._prepare_source(
                    run_id,
                    lease_token,
                    source_ref,
                    submission_id,
                    source_size_bytes,
                )
                with self._condition:
                    self._ready[submission_id] = paths
                    self._sizes[submission_id] = size
                    self._evict_locked(protected=(submission_id,))
                    if not self._within_limits_locked():
                        self._ready.pop(submission_id, None)
                        self._sizes.pop(submission_id, None)
                        _remove_cache_path(paths[0].parent)
                        raise RunnerError("source cache capacity is in use")
                    self._in_use[submission_id] = 1
            finally:
                with self._condition:
                    self._building.discard(submission_id)
                    self._condition.notify_all()
        try:
            yield paths
        finally:
            with self._lock:
                remaining = self._in_use.get(submission_id, 0) - 1
                if remaining > 0:
                    self._in_use[submission_id] = remaining
                else:
                    self._in_use.pop(submission_id, None)
                self._evict_locked()


def registry_image_exporter(client: images.RegistryClient, *, rules: Optional[images.ImageRules] = None) -> ImageExporter:
    """Materialize a pinned image from the Arena registry with the hardened extractor (no Docker daemon)."""

    image_rules = rules or images.ImageRules()

    def export(image_reference: str, image_digest: str, target_dir: Path) -> None:
        try:
            reference = images.parse_reference(image_reference)
            if reference.digest != image_digest:
                raise RunnerError("lease image reference does not name the lease digest")
            images.materialize_rootfs(client, reference, target_dir, rules=image_rules)
        except images.ImageError as exc:
            raise RunnerError("image %s could not be materialized: %s" % (image_digest[:19], exc.rule_id)) from exc

    return export


# ---------------------------------------------------------------------------
# Per-run state and the worker socket
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    lease: Dict[str, Any]
    lease_token: str
    calls: List[Dict[str, Any]] = field(default_factory=list)
    provider_error_events: List[Dict[str, Any]] = field(default_factory=list)
    provider_error_events_dropped: int = 0
    decision_events: List[Dict[str, Any]] = field(default_factory=list)
    decision_events_omitted: int = 0
    decision_nonfinish_recorded: int = 0
    decision_finish_recorded: bool = False
    recovered_responses_retry_call_identities: set[str] = field(
        default_factory=set
    )
    worker_observations: training_observations.Buffer = field(default_factory=training_observations.Buffer)
    worker_observation_ordinal: int = 0
    worker_observations_pending: set[int] = field(default_factory=set)
    worker_observations_frozen: bool = False
    worker_handlers_inflight: int = 0
    training_control_frames: int = 0
    training_control_bytes: int = 0
    training_retry_tokens: Dict[str, Tuple[bytes, int]] = field(default_factory=dict)
    action_sequence: int = 0
    refusals: int = 0  # refused calls answered by the Arena for this run
    local_refusals: int = 0  # refused locally during an operation cooldown
    operation_refusals: Dict[Tuple[str, str], Tuple[int, str]] = field(
        default_factory=dict
    )
    operation_refusal_until: Dict[Tuple[str, str], float] = field(
        default_factory=dict
    )
    operation_probe_inflight: set[Tuple[str, str]] = field(default_factory=set)
    operation_latest_calls: Dict[Tuple[str, str], Dict[str, Any]] = field(
        default_factory=dict
    )
    operation_recovery_sequence: Dict[Tuple[str, str], int] = field(
        default_factory=dict
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    quota_request_count: int = 0
    quota_snapshot: Optional[Dict[str, Any]] = None
    quota_snapshot_at: float = 0.0
    quota_snapshot_inflight: bool = False
    quota_snapshot_generation: int = 0
    trusted_quota_failure: bool = False
    quota_failure_diagnostics: set[Tuple[str, bool]] = field(
        default_factory=set
    )
    quota_condition: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )


def _execution_output_observation(
    result: Optional[runtime.SandboxResult],
    output: Optional[Mapping[str, Any]],
    terminal: str,
    diagnostic: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Observe selected output before provider rules can clear it; retain no body."""

    state, code, count = "not_validated", None, None
    raw = result.output_bytes if result is not None else None
    if result is not None:
        if result.output_error:
            state, code = "unreadable", "sandbox_output_error"
        elif raw is None:
            state, code = "missing", "missing_output"
        elif terminal == "accepted" and output is not None:
            count = len(output["companies"])
            state = "validated_nonempty" if count else "validated_empty"
        elif diagnostic is not None and diagnostic.get("error_class") == "execution_output_invalid":
            state = "invalid"
            reason = diagnostic.get("reason")
            code = reason if reason in (
                "invalid_json", "output_schema_mismatch", "output_contract_violation",
            ) else "output_contract_violation"
    return {
        "state": state,
        "terminal_before_classification": terminal,
        "failure_code": code,
        "company_count": count,
        "output_bytes": len(raw) if raw is not None else None,
        "output_sha256": contracts.hash_bytes(raw) if raw is not None else None,
        # These bytes may be the last valid checkpoint, not the rejected final file.
        "checkpoint_output_invalid": bool(result and result.checkpoint_output_invalid),
    }


def _unresolved_operation_failures(
    latest_calls: Sequence[Mapping[str, Any]],
    recovered_identities: Collection[str],
) -> Tuple[int, int]:
    """Count each tool's latest actual credential or provider failure."""

    credentials = providers = 0
    for call in latest_calls:
        if call.get("call_identity") in recovered_identities:
            continue
        error_code = call.get("error_code")
        miner_account_status = (
            call.get("funding_source") == "miner_key"
            and call.get("provider_status") in (401, 402, 403)
        )
        if call.get("funding_source") == "miner_key" and error_code in (
            "miner_credentials_unavailable", "miner_provider_not_configured"
        ):
            credentials += 1
        elif error_code in ("broker_unavailable", "provider_unavailable") or (
            operations.provider_status_is_infrastructure(call.get("provider_status"))
            and error_code != "provider_request_refused"
            and not miner_account_status
        ):
            providers += 1
    return credentials, providers


def _record_latest_operation_call_locked(
    state: RunState, key: Tuple[str, str], call: Dict[str, Any], sequence: int,
) -> bool:
    """Order attempts by the worker's issued sequence, not optional call metadata."""

    previous = state.operation_latest_calls.get(key)
    if previous is not None and sequence < previous["action_sequence"]:
        return False
    state.operation_latest_calls[key] = {**call, "action_sequence": sequence}
    return True


def _provider_error_event(
    operation_id: str, action_sequence: int, exc: RunnerError,
) -> Dict[str, Any]:
    content: Dict[str, Any] = {
        "operation_id": operation_id,
        "action_sequence": action_sequence,
        "outcome": "unknown",
        "error_class": type(exc).__name__[:64],
        "error_code": "broker_unavailable",
    }
    operation = operations.get_operation(operation_id)
    if operation is not None:
        content["provider"] = operation.provider
    if exc.http_status is not None:
        content["http_status"] = exc.http_status
    if exc.denial_code is not None:
        content["denial_code"] = exc.denial_code
    return trajectory.event("runtime.provider_error", content)


def _buffer_provider_error_event_locked(
    state: RunState, provider_event: Dict[str, Any],
) -> None:
    """Bound the existing runtime diagnostics, including failed socket writes."""
    if len(state.provider_error_events) < MAX_BUFFERED_PROVIDER_ERROR_EVENTS:
        state.provider_error_events.append(provider_event)
        return
    state.provider_error_events_dropped += 1
    content = state.provider_error_events[-1]["content"]
    content["dropped"] = True
    content["dropped_count"] = state.provider_error_events_dropped


def _socket_write_failure_event(receipt: Mapping[str, Any], exc: BaseException) -> Dict[str, Any]:
    # Never retain exception prose, filenames or an arbitrary subclass name.
    category = next((kind.__name__ for kind in (
        BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError,
    ) if isinstance(exc, kind)), "other")
    number = getattr(exc, "errno", None)
    return trajectory.event("runtime.socket_write_failed", {
        "receipt_ordinal": receipt["ordinal"],
        "action_sequence": receipt["action_sequence"],
        "call_identity": receipt["call_identity"],
        "operation_id": receipt["operation_id"],
        "protocol": receipt["protocol"],
        "exception_class": category,
        "errno": number if type(number) is int and 0 <= number <= 65535 else None,
    })


def _provider_error_events(state: RunState) -> List[Dict[str, Any]]:
    with state.lock:
        return [
            {**item, "content": dict(item["content"])}
            for item in state.provider_error_events
        ]


def _decision_events_and_summary(
    state: RunState, observed_at: str,
) -> List[Dict[str, Any]]:
    """Snapshot model-reported decisions and one host capture summary."""

    with state.lock:
        decisions = [
            {**item, "content": dict(item["content"])}
            for item in state.decision_events
        ]
        omitted = state.decision_events_omitted
    return [
        *decisions,
        trajectory.event(
            "runtime.decision_capture",
            {
                "status": "provided" if decisions or omitted else "not_provided",
                "recorded": len(decisions),
                "omitted": omitted,
            },
            occurred_at=observed_at,
        ),
    ]


def _timestamp(clock: Callable[[], datetime]) -> str:
    return clock().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quota_snapshot_upstream_limit(wall_clock_seconds: int) -> int:
    """Bound authoritative reads for one signed sandbox lifetime."""

    if (
        isinstance(wall_clock_seconds, bool)
        or not isinstance(wall_clock_seconds, int)
        or wall_clock_seconds < 1
    ):
        raise RunnerError("quota snapshot wall clock is invalid")
    cache_intervals = (
        wall_clock_seconds * 1000 + QUOTA_SNAPSHOT_CACHE_MILLISECONDS - 1
    ) // QUOTA_SNAPSHOT_CACHE_MILLISECONDS
    return (
        QUOTA_SNAPSHOT_SCHEMA_VARIANTS * cache_intervals
        + QUOTA_SNAPSHOT_STARTUP_REQUESTS
    )


def _call_provider_matches_operation(call: Mapping[str, Any], operation_id: str) -> bool:
    """Bind broker receipts to their requested or reviewed effective provider."""

    if not isinstance(call, Mapping):
        return False
    operation = operations.get_operation(operation_id)
    if operation is None or operation_id in operations.CATALOG_OPERATIONS:
        return False
    if call.get("provider") == operation.provider:
        return True
    # Score reads keep their legacy operation id after the broker routes
    # them through Deepline. Require the complete reviewed route summary;
    # an arbitrary different provider cannot gain receipt authority.
    return bool(
        call.get("provider") == "deepline"
        and call.get("effective_operation_id") == scoring_provider_compat.EFFECTIVE_OPERATION_ID
        and (
            call.get("compatibility_version") == scoring_provider_compat.COMPATIBILITY_VERSION
            or (
                call.get("compatibility_version") == scoring_provider_compat.SOURCE_HTTP_COMPATIBILITY_VERSION
                and call.get("adapter") == "generic_http_raw_html"
            )
        )
        and isinstance(call.get("adapter"), str)
        and isinstance(call.get("tool"), str)
        and (operation_id, call.get("adapter"), call.get("tool")) in {
            ("scrapingdog.scrape", "firecrawl_raw_html", scoring_provider_compat.FIRECRAWL_TOOL),
            ("scrapingdog.scrape", "generic_http_raw_html", scoring_provider_compat.GENERIC_HTTP_TOOL),
            ("scrapingdog.scrape", "harvest_linkedin_post_html", scoring_provider_compat.HARVEST_POST_TOOL),
            *(("scrapingdog.scrape", "generic_ats_json:" + kind, scoring_provider_compat.GENERIC_HTTP_TOOL)
              for kind in ("ashby", "greenhouse", "greenhouse_board", "workday", "lever")),
            ("scrapingdog.x_post", "twitter_x_post", scoring_provider_compat.TWITTER_POST_TOOL),
            ("scrapingdog.linkedinjobs", "harvest_linkedin_job", scoring_provider_compat.HARVEST_JOB_TOOL),
            ("scrapingdog.profile_post", "harvest_linkedin_post", scoring_provider_compat.HARVEST_POST_TOOL),
        }
    )


@dataclass
class _WorkerSocketAdmission:
    permit: threading.BoundedSemaphore
    capture_only: bool
    completion_deadline: float
    read_deadline: Optional[float] = None
    provider_deadline: Optional[float] = None


class WorkerSocketServer:
    """Unix-socket bridge: operation frames in, provider responses out.

    Frames carry only the operation id, validated parameters, and a timeout;
    round, miner, stage, run, account, and lease identity come from the lease
    the worker holds, never from the sandbox.
    """

    def __init__(
        self,
        socket_path: Path,
        api: ArenaApiClient,
        state: RunState,
        *,
        max_connections: int = MAX_WORKER_CONNECTIONS,
        read_timeout_seconds: float = WORKER_SOCKET_READ_TIMEOUT_SECONDS,
        quota_snapshot_upstream_limit: Optional[int] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(max_connections, bool) or not isinstance(max_connections, int) or max_connections < 1:
            raise RunnerError("worker socket connection limit is invalid")
        if read_timeout_seconds <= 0:
            raise RunnerError("worker socket read timeout is invalid")
        if quota_snapshot_upstream_limit is None:
            quota_snapshot_upstream_limit = _quota_snapshot_upstream_limit(
                contracts.ICP_WALL_CLOCK_SECONDS
            )
        if (
            isinstance(quota_snapshot_upstream_limit, bool)
            or not isinstance(quota_snapshot_upstream_limit, int)
            or quota_snapshot_upstream_limit < 1
        ):
            raise RunnerError("worker quota snapshot limit is invalid")
        self._path = Path(socket_path)
        self._api = api
        self._state = state
        self._max_connections = max_connections
        self._read_timeout_seconds = float(read_timeout_seconds)
        self._quota_snapshot_upstream_limit = quota_snapshot_upstream_limit
        self._monotonic = monotonic
        self._server: Optional[socketserver.ThreadingUnixStreamServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._admissions: Dict[socket.socket, _WorkerSocketAdmission] = {}
        self._handler_context = threading.local()

    def _retain_handler_deadline(self, deadline: float, clock: Callable[[], float]) -> None:
        """Retain an existing dispatch window, without changing provider behavior."""

        admission = getattr(self._handler_context, "admission", None)
        if admission is None:
            return
        # Dispatch recovery tests can inject a separate monotonic clock. Keep
        # admission and drain deadlines in the real socket clock domain.
        socket_deadline = time.monotonic() + max(0.0, deadline - clock())
        with self._state.lock:
            if admission.provider_deadline is None:
                admission.provider_deadline = socket_deadline
            else:
                # Responses retries share the first operation window. A later
                # attempt must never give final capture a fresh lifetime.
                admission.provider_deadline = min(admission.provider_deadline, socket_deadline)
            admission.completion_deadline = admission.provider_deadline

    def _retain_socket_write_deadline(self, connection: socket.socket) -> None:
        """Capture a response write within the socket's unchanged timeout."""

        admission = getattr(self._handler_context, "admission", None)
        timeout = connection.gettimeout()
        if admission is not None and timeout is not None:
            with self._state.lock:
                admission.completion_deadline = time.monotonic() + timeout

    def _quota_snapshot(
        self, *, include_sourcing_cost: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Cache and single-flight passive reads without touching call counters."""

        state = self._state
        condition = state.quota_condition
        wait_deadline = time.monotonic() + API_TIMEOUT_SECONDS + 1.0

        def cached_view() -> Optional[Dict[str, Any]]:
            snapshot = state.quota_snapshot
            if snapshot is None:
                return None
            if include_sourcing_cost:
                return dict(snapshot) if "sourcing_cost" in snapshot else None
            return {
                "schema_version": (
                    lab_arena_checkpoint.QUOTA_SNAPSHOT_SCHEMA_VERSION
                ),
                "providers": snapshot["providers"],
            }

        def log_failure_once(reason: str) -> None:
            key = (reason, include_sourcing_cost)
            with condition:
                if key in state.quota_failure_diagnostics:
                    return
                state.quota_failure_diagnostics.add(key)
                read_count = state.quota_request_count
            _log_quota_snapshot_failure(
                reason=reason,
                read_count=read_count,
                read_limit=self._quota_snapshot_upstream_limit,
                include_sourcing_cost=include_sourcing_cost,
            )

        with condition:
            while True:
                if (
                    cached_view() is not None
                    and self._monotonic() - state.quota_snapshot_at
                    < QUOTA_SNAPSHOT_CACHE_SECONDS
                ):
                    return cached_view()
                if not state.quota_snapshot_inflight:
                    break
                generation = state.quota_snapshot_generation
                remaining = wait_deadline - time.monotonic()
                if remaining <= 0:
                    log_failure_once("upstream_exception")
                    return None
                condition.wait_for(
                    lambda: (
                        not state.quota_snapshot_inflight
                        or state.quota_snapshot_generation != generation
                    ),
                    timeout=remaining,
                )
                if state.quota_snapshot_generation == generation:
                    # The leader produced no usable upstream result within
                    # the same bounded API window. The diagnostic deliberately
                    # groups this wait timeout with upstream exceptions; both
                    # have identical fail-closed behavior and share no payload.
                    log_failure_once("upstream_exception")
                    return None
                if state.quota_snapshot is None:
                    # The single-flight leader already logged and published
                    # its fail-closed result. A joiner does not spend another
                    # upstream read merely because it arrived concurrently.
                    return None
                # Recheck the current cache and leader. When several v2
                # callers wait behind one v1 leader, only the first becomes
                # the v2 upgrade leader; the others join that new generation.
            if (
                state.quota_request_count
                >= self._quota_snapshot_upstream_limit
            ):
                log_failure_once("read_cap")
                return None
            # Only the single-flight leader consumes the finite upstream-read
            # allowance. Cached reads and waiters are passive local views.
            state.quota_request_count += 1
            state.quota_snapshot_inflight = True

        snapshot = None
        run_id = state.lease["run_id"]
        lease_token = state.lease_token
        failure_reason = None
        try:
            if include_sourcing_cost:
                document = self._api.quota_usage(
                    run_id, lease_token, include_sourcing_cost=True
                )
            else:
                document = self._api.quota_usage(run_id, lease_token)
        except Exception:
            failure_reason = "upstream_exception"
        else:
            try:
                snapshot = (
                    lab_arena_checkpoint.validate_quota_cost_snapshot(document)
                    if include_sourcing_cost
                    else lab_arena_checkpoint.validate_quota_snapshot(document)
                )
            except lab_arena_checkpoint.QuotaUnavailable:
                failure_reason = "invalid_snapshot"

        if failure_reason is not None:
            log_failure_once(failure_reason)

        with condition:
            state.quota_snapshot_inflight = False
            state.quota_snapshot_generation += 1
            if snapshot is None:
                if include_sourcing_cost:
                    # Cost visibility is optional. Drop a prior cost view so
                    # this request cannot reuse it, but preserve validated v1
                    # counters and their health classification.
                    if (
                        state.quota_snapshot is not None
                        and "sourcing_cost" in state.quota_snapshot
                    ):
                        state.quota_snapshot = {
                            "schema_version": (
                                lab_arena_checkpoint.QUOTA_SNAPSHOT_SCHEMA_VERSION
                            ),
                            "providers": state.quota_snapshot["providers"],
                        }
                else:
                    state.quota_snapshot = None
                    state.quota_snapshot_at = 0.0
                    state.trusted_quota_failure = True
            else:
                state.quota_snapshot = dict(snapshot)
                state.quota_snapshot_at = self._monotonic()
                state.trusted_quota_failure = False
            condition.notify_all()
            return cached_view()

    def _handle_quota_control(self, raw: bytes) -> Optional[bytes]:
        """Handle the exact non-provider control frame, if one was supplied."""

        try:
            frame = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(frame, Mapping) or frame.get(
            "schema_version"
        ) != lab_arena_checkpoint.QUOTA_CONTROL_SCHEMA_VERSION:
            return None
        include_cost = (
            dict(frame) == lab_arena_checkpoint.QUOTA_COST_CONTROL_FRAME
        )
        if (
            not include_cost
            and dict(frame) != lab_arena_checkpoint.QUOTA_CONTROL_FRAME
        ):
            return shim.encode_worker_error("invalid_frame")
        snapshot = self._quota_snapshot(include_sourcing_cost=include_cost)
        if snapshot is None:
            return shim.encode_worker_error("quota_unavailable")
        return contracts.canonical_json(snapshot).encode("utf-8")

    def _handle_signal_judgment_control(self, raw: bytes) -> Optional[bytes]:
        from lab_arena import signal_cache_control
        try:
            frame = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(frame, Mapping) or frame.get("schema_version") != signal_cache_control.SCHEMA:
            return None
        state = self._state
        try:
            if (len(raw) > signal_cache_control.MAX_BYTES
                or set(frame) != {"schema_version", "request"}
                or state.lease.get("kind") != "score"
                or (state.lease.get("scorer_policy") or {}).get("signal_judgment_policy") != signal_cache_control.POLICY):
                raise ValueError("signal cache control is not authorized")
            request = signal_cache_control.validate_request(frame["request"])
            result = self._api.signal_judgment(state.lease["run_id"], state.lease_token, request)
            encoded = contracts.canonical_json(result).encode("utf-8")
            if len(encoded) > signal_cache_control.MAX_BYTES:
                raise ValueError("signal cache response too large")
            return encoded
        except (RunnerError, ValueError, TypeError):
            return contracts.canonical_json({"status": "unavailable"}).encode("utf-8")

    def _handle_decision_control(self, raw: bytes) -> Optional[bytes]:
        """Capture one bounded model report without dispatching a provider call."""

        try:
            frame = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(frame, Mapping) or frame.get(
            "schema_version"
        ) != lab_arena_checkpoint.DECISION_SCHEMA_VERSION:
            return None
        if len(raw) > lab_arena_checkpoint.MAX_DECISION_FRAME_BYTES:
            return contracts.canonical_json({"recorded": False}).encode("utf-8")
        try:
            decision = lab_arena_checkpoint.validate_decision_frame(frame)
        except Exception:
            return contracts.canonical_json({"recorded": False}).encode("utf-8")

        state = self._state
        if str(state.lease.get("kind") or "execute") != "execute":
            return contracts.canonical_json({"recorded": False}).encode("utf-8")
        try:
            observed_at = _timestamp(lambda: datetime.now(timezone.utc))
            with state.lock:
                is_finish = decision["decision"] == "finish"
                if (
                    is_finish and state.decision_finish_recorded
                    or not is_finish
                    and state.decision_nonfinish_recorded
                    >= MAX_BUFFERED_DECISION_EVENTS - 1
                ):
                    state.decision_events_omitted += 1
                    return contracts.canonical_json({"recorded": False}).encode(
                        "utf-8"
                    )
                sequence = len(state.decision_events)
                content = {
                    "source": "model_reported",
                    "sequence": sequence,
                    "after_action_sequence": state.action_sequence - 1,
                    "objective": decision["objective"],
                    "evidence": list(decision["evidence"]),
                    "rationale": decision["rationale"],
                    "next_action": decision["next_action"],
                    "decision": decision["decision"],
                }
                if "candidate" in decision:
                    content["candidate"] = decision["candidate"]
                redacted = trajectory.sanitize_content(
                    content, secrets=(state.lease_token,)
                )
                state.decision_events.append(
                    trajectory.event(
                        "runtime.decision", redacted, occurred_at=observed_at
                    )
                )
                if is_finish:
                    state.decision_finish_recorded = True
                else:
                    state.decision_nonfinish_recorded += 1
        except Exception:
            try:
                with state.lock:
                    state.decision_events_omitted += 1
            except Exception:
                pass
            return contracts.canonical_json({"recorded": False}).encode("utf-8")
        return contracts.canonical_json({"recorded": True}).encode("utf-8")

    def _handle_training_control(self, raw: bytes, *, replayed: bool = False) -> Optional[bytes]:
        """Store only explicit sandbox reports under this worker's active lease."""
        if len(raw) > lab_arena_checkpoint.MAX_TRAINING_FRAME_BYTES:
            self._mark_observations_incomplete()
            return contracts.canonical_json({"recorded": False}).encode("utf-8")
        try:
            frame = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return None
        if not isinstance(frame, Mapping) or frame.get("schema_version") != lab_arena_checkpoint.TRAINING_SCHEMA_VERSION:
            return None
        unavailable = contracts.canonical_json({"recorded": False}).encode("utf-8")
        try:
            frame = lab_arena_checkpoint.validate_training_frame(frame)
            if str(self._state.lease.get("kind") or "execute") != "execute":
                return unavailable
            data = dict(frame["data"])
            if frame["kind"] == "web_fetch":
                body = data["body"].encode("utf-8")
                data.update(body_sha256="sha256:" + hashlib.sha256(body).hexdigest(),
                            body_bytes=len(body), stored_body_sha256=None,
                            stored_body_bytes=None, body_ref=None)
            elif frame["kind"] == "web_fetch_binary":
                body = base64.b64decode(data["body_b64"], validate=True)
                data.update(body_sha256="sha256:" + hashlib.sha256(body).hexdigest(),
                            body_bytes=len(body), stored_body_sha256=None,
                            stored_body_bytes=None, body_ref=None,
                            privacy_review="unreviewed_binary")
            with self._state.lock:
                buffer = self._state.worker_observations
                if self._state.worker_observations_frozen:
                    return unavailable
                retry_token = frame.get("retry_token")
                if retry_token is not None:
                    previous = self._state.training_retry_tokens.get(retry_token)
                    if previous is not None:
                        if previous[0] != hashlib.sha256(raw).digest():
                            buffer.failures.add("capture_incomplete")
                            return unavailable
                        return contracts.canonical_json({"recorded": True,
                            "observation_id": previous[1]}).encode("utf-8")
                if frame["kind"] == "response_received":
                    matched = [row for row in buffer.response_receipts if row["call_identity"] == data["call_identity"]
                               and row["status"] == data["status"] and row["body_sha256"] == data["body_sha256"]
                               and row["body_bytes"] == data["body_bytes"]]
                    if len(matched) != 1:
                        buffer.failures.add("capture_incomplete")
                        return unavailable
                references = frame["depends_on"]
                known_calls = {row.get("call_identity") for row in self._state.calls}
                for reference in references:
                    if reference.startswith("agent:"):
                        if (not reference[6:].isdigit() or int(reference[6:]) >= len(buffer.agent_observations)):
                            buffer.failures.add("capture_incomplete")
                            return unavailable
                    elif reference not in known_calls:
                        buffer.failures.add("capture_incomplete")
                        return unavailable
                value = {"id": len(buffer.agent_observations), "kind": frame["kind"],
                         "observed_at": _timestamp(lambda: datetime.now(timezone.utc)),
                         "after_action_sequence": (None if replayed else
                                                   self._state.action_sequence - 1 if self._state.action_sequence else None),
                         "depends_on": references, "data": data,
                         "privacy": {"redacted": False, "redactions": []}}
                identity = buffer.append_agent_observation(value, secrets=(
                    self._state.lease_token,
                    *(os.environ.get(name, "") for name in (
                        "OPENROUTER_API_KEY", "DEEPLINE_API_KEY", "SCRAPINGDOG_API_KEY")),
                ))
                if identity is not None and retry_token is not None:
                    self._state.training_retry_tokens[retry_token] = (hashlib.sha256(raw).digest(), identity)
            if identity is None:
                return unavailable
            return contracts.canonical_json({"recorded": True, "observation_id": identity}).encode("utf-8")
        except Exception:
            self._mark_observations_incomplete()
            return unavailable

    def _mark_socket_delivery(
        self, report: Mapping[str, Any], *, written: bool, error: BaseException | None = None,
    ) -> None:
        ordinal = report.get("receipt_ordinal")
        if type(ordinal) is not int:
            return
        try:
            with self._state.lock:
                if not self._state.worker_observations_frozen:
                    self._state.worker_observations.mark_socket_write(
                        ordinal, written_at=_timestamp(lambda: datetime.now(timezone.utc)) if written else None,
                    )
                    if not written and error is not None:
                        receipt = next(row for row in self._state.worker_observations.response_receipts
                                       if row["ordinal"] == ordinal)
                        _buffer_provider_error_event_locked(
                            self._state, _socket_write_failure_event(receipt, error),
                        )
        except Exception:
            self._mark_observations_incomplete()

    def _allow_training_frame(self, size: int) -> bool:
        """Cap optional capture traffic before reading its large body."""
        with self._state.lock:
            if (self._state.training_control_frames >= MAX_TRAINING_CONTROL_FRAMES
                    or self._state.training_control_bytes + size > MAX_TRAINING_CONTROL_BYTES):
                self._state.worker_observations.failures.add("capture_incomplete")
                return False
            self._state.training_control_frames += 1
            self._state.training_control_bytes += size
            return True

    @staticmethod
    def _temporary_hold(
        document: Mapping[str, Any], operation_id: str, action_sequence: int
    ) -> bool:
        """Accept only the broker's proved pre-dispatch billing hold."""

        if not isinstance(document, Mapping):
            return False
        call = document.get("call")
        if not (
            document.get("status") == 502
            and isinstance(call, Mapping)
            and call.get("operation_id") == operation_id
            and call.get("action_sequence") == action_sequence
            and call.get("error_code") == "provider_unavailable"
            and call.get("outcome") == "not_dispatched"
            and call.get("reason") == "provider_cost_uncertain"
            and call.get("idempotent") is False
            and call.get("provider_status") is None
        ):
            return False
        try:
            body = json.loads(
                base64.b64decode(str(document.get("body_b64")), validate=True)
            )
        except (TypeError, ValueError):
            return False
        return body == {"error": {"code": "provider_unavailable"}}

    def _cancelled(
        self, cancel_requested: Optional[Callable[[], bool]]
    ) -> bool:
        if self._stopping.is_set():
            return True
        if cancel_requested is None:
            return False
        try:
            return bool(cancel_requested())
        except Exception:
            return True

    @staticmethod
    def _deepline_uncertain_identity(
        document: Mapping[str, Any], operation_id: str, action_sequence: int
    ) -> Optional[str]:
        """Recover only a bound uncertainty or a billed, missing Deepline reply."""

        if not isinstance(document, Mapping) or set(document) != {
            "status", "headers", "body_b64", "call"
        }:
            return None
        call = document.get("call")
        settled_lost_response = (
            document.get("status") == 502
            and isinstance(call, Mapping)
            and call.get("outcome") == "settled"
            and call.get("deepline_response_missing") is True
            and type(call.get("actual_microusd")) is int
            and call["actual_microusd"] >= 0
            and isinstance(call.get("transport_error_class"), str)
            and bool(call["transport_error_class"])
            and call.get("transport_recovery") is None
            and call.get("provider_status") == 502
        )
        if not (
            type(document.get("status")) is int
            and document["status"] in (502, 409)
            and isinstance(document.get("headers"), Mapping)
            and isinstance(call, Mapping)
            and call.get("operation_id") == operation_id
            and call.get("provider") == "deepline"
            and type(call.get("action_sequence")) is int
            and call["action_sequence"] == action_sequence
            and (call.get("outcome") == "uncertain" or settled_lost_response)
            and call.get("error_code") == (
                "provider_unavailable" if document["status"] == 502 else "call_uncertain"
            )
            and isinstance(call.get("call_identity"), str)
            and contracts.SHA256_RE.fullmatch(call["call_identity"]) is not None
        ):
            return None
        try:
            body = json.loads(base64.b64decode(str(document["body_b64"]), validate=True))
        except (TypeError, ValueError, UnicodeDecodeError):
            return None
        if body != {"error": {"code": call["error_code"]}}:
            return None
        if (call.get("deepline_terminal_failure") is True
            and type(call.get("provider_status")) is int
            and 400 <= call["provider_status"] <= 599
            and call.get("transport_error_class") is None
            and call.get("deepline_response_missing") is not True):
            # A broker-proved completed failure cannot yield a missing success.
            # Its unknown charge still follows the ordinary delayed audit.
            return None
        return call["call_identity"]

    def _recover_deepline_response(
        self, document: Dict[str, Any], frame: Mapping[str, Any], deadline: float,
        cancel_requested: Optional[Callable[[], bool]],
    ) -> Dict[str, Any]:
        """Make one same-action recovery lookup within the original API window."""

        identity = self._deepline_uncertain_identity(
            document, frame["operation_id"], frame["action_sequence"]
        )
        remaining = deadline - self._monotonic()
        if identity is None or remaining <= 0 or self._cancelled(cancel_requested):
            return document
        # Native timeout is outside the provider body. Compatibility routes
        # can embed it in their payload, so preserve those frames exactly.
        # The HTTP request cap below still uses the original remaining window.
        recovery_frame = dict(frame)
        if frame["operation_id"] == "deepline.execute":
            recovery_frame["timeout_ms"] = min(
                frame["timeout_ms"], max(1, int(remaining * 1000))
            )
        try:
            if isinstance(self._api, HttpArenaApiClient):
                recovered = self._api.provider(
                    self._state.lease["run_id"], self._state.lease_token,
                    recovery_frame, timeout_seconds=remaining,
                )
            else:
                recovered = self._api.provider(
                    self._state.lease["run_id"], self._state.lease_token, recovery_frame,
                )
        except RunnerError:
            return document
        if self._monotonic() > deadline:
            return document
        if not isinstance(recovered, Mapping) or set(recovered) != {
            "status", "headers", "body_b64", "call"
        }:
            return document
        call = recovered.get("call")
        if not (
            type(recovered.get("status")) is int and recovered["status"] == 200
            and isinstance(recovered.get("headers"), Mapping)
            and isinstance(call, Mapping)
            and call.get("operation_id") == frame["operation_id"]
            and call.get("provider") == "deepline"
            and type(call.get("action_sequence")) is int
            and call["action_sequence"] == frame["action_sequence"]
            and call.get("call_identity") == identity
            and call.get("outcome") in ("settled", "uncertain")
            and (
                document["call"].get("outcome") != "settled"
                or (
                    call.get("outcome") == "settled"
                    and type(call.get("actual_microusd")) is int
                    and call["actual_microusd"]
                    == document["call"]["actual_microusd"]
                )
            )
            and call.get("error_code") is None
        ):
            return document
        try:
            if not isinstance(recovered.get("body_b64"), str):
                return document
            base64.b64decode(recovered["body_b64"], validate=True)
        except (TypeError, ValueError):
            return document
        return dict(recovered)

    def _dispatch_once(
        self,
        operation_id: str,
        parameters: Mapping[str, Any],
        timeout_ms: int,
        *,
        cancel_requested: Optional[Callable[[], bool]] = None,
        observation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        state = self._state
        operation = operations.get_operation(operation_id)
        deadline = self._monotonic() + max(1, int(timeout_ms)) / 1000.0 + (
            MAX_PROVIDER_API_TIMEOUT_SECONDS - MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS
        )
        self._retain_handler_deadline(deadline, self._monotonic)
        # Compatibility routes and the native execute route share the actual
        # Deepline tool's guard. Catalog reads and other providers stay separate.
        tool = (
            operation.deepline_tool
            if operation is not None and operation.provider == "deepline"
            else None
        )
        if operation_id == "deepline.execute":
            tool = str(parameters.get("tool") or "")
        refusal_key = (
            ("deepline.execute", tool) if tool else (operation_id, "")
        )
        with state.lock:
            sequence = state.action_sequence
            state.action_sequence += 1
            if observation is not None:
                observation["action_sequence"] = sequence
                observation["contributes_to_provider_call_count"] = False
            refusal_count, refusal_error = state.operation_refusals.get(
                refusal_key, (0, "call_refused")
            )
            probe = refusal_count >= MAX_REFUSED_FRAMES
            if probe:
                if (
                    refusal_key in state.operation_probe_inflight
                    or self._monotonic() < state.operation_refusal_until.get(
                        refusal_key, 0.0
                    )
                ):
                    # Refuse only this operation. Keep its original error and
                    # never start an automatic paid retry.
                    state.local_refusals += 1
                    if observation is not None:
                        observation["kind"] = "local_refusal"
                        observation["contributes_to_provider_call_count"] = False
                    return refusal_error, None
                # Reserve one model-requested probe until its actual result.
                state.operation_probe_inflight.add(refusal_key)
        # Calls admitted before the threshold can still complete concurrently.
        # Keep their original frames and accounting; do not serialize or cancel
        # an already dispatched request to enforce an exact cutoff.
        try:
            return self._dispatch_admitted(
                operation_id, parameters, timeout_ms, state, sequence,
                refusal_key, probe, deadline, cancel_requested, observation,
            )
        finally:
            if probe:
                with state.lock:
                    state.operation_probe_inflight.discard(refusal_key)
                    if (
                        refusal_key in state.operation_refusals
                        and state.operation_refusal_until.get(refusal_key, 0.0)
                        <= self._monotonic()
                    ):
                        state.operation_refusal_until[refusal_key] = (
                            self._monotonic() + REFUSAL_PROBE_COOLDOWN_SECONDS
                        )

    def _dispatch_admitted(
        self,
        operation_id: str,
        parameters: Mapping[str, Any],
        timeout_ms: int,
        state: RunState,
        sequence: int,
        refusal_key: Tuple[str, str],
        probe: bool,
        deadline: float,
        cancel_requested: Optional[Callable[[], bool]],
        observation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        frame = {"operation_id": operation_id, "parameters": dict(parameters), "timeout_ms": int(timeout_ms), "action_sequence": sequence}
        while True:
            if self._cancelled(cancel_requested):
                return "worker_unavailable", None
            try:
                document = self._api.provider(
                    state.lease["run_id"], state.lease_token, frame
                )
            except RunnerError as exc:
                if observation is not None:
                    observation.update(kind="transport_failure", exception_class=type(exc).__name__)
                # A gateway failure can occur before dispatch, after dispatch,
                # or after settlement. Never replay an unknown outcome.
                provider_event = None
                observation_error = None
                try:
                    provider_event = _provider_error_event(
                        operation_id, sequence, exc
                    )
                except Exception as observation_exc:
                    observation_error = type(observation_exc).__name__
                with state.lock:
                    failed_call = {
                        "operation_id": operation_id,
                        "action_sequence": sequence,
                        "outcome": "unknown",
                        "error_code": "broker_unavailable",
                    }
                    state.calls.append(failed_call)
                    if observation is not None:
                        observation["contributes_to_provider_call_count"] = True
                    _record_latest_operation_call_locked(
                        state, refusal_key, failed_call, sequence
                    )
                    if provider_event is not None:
                        try:
                            _buffer_provider_error_event_locked(
                                state, provider_event
                            )
                        except Exception as observation_exc:
                            observation_error = type(observation_exc).__name__
                if observation_error is not None:
                    try:
                        print(
                            "Arena provider error trajectory capture failed: %s"
                            % observation_error,
                            file=sys.stderr,
                            flush=True,
                        )
                    except Exception:
                        pass
                return "worker_unavailable", None
            if not self._temporary_hold(document, operation_id, sequence):
                break
            # The database created no reservation or provider request. Keep
            # the exact frame and sequence, but stop before a new API request
            # when the original socket caller has gone away.
            if self._stopping.wait(
                TEMPORARY_HOLD_RETRY_SECONDS
            ) or self._cancelled(cancel_requested):
                # Retain one terminal diagnostic for the hold that consumed
                # the caller's remaining window. Intermediate polls are not
                # provider calls and stay out of the run summary.
                with state.lock:
                    held_call = dict(document["call"])
                    state.calls.append(held_call)
                    if observation is not None:
                        observation["contributes_to_provider_call_count"] = True
                    _record_latest_operation_call_locked(
                        state, refusal_key, held_call, sequence
                    )
                return "worker_unavailable", None
        document = self._recover_deepline_response(
            document, frame, deadline, cancel_requested
        )
        def invalid_broker_response() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
            if observation is not None:
                observation["kind"] = "invalid_response"
            with state.lock:
                failed_call = {
                    "operation_id": operation_id,
                    "action_sequence": sequence,
                    "outcome": "unknown",
                    "error_code": "broker_unavailable",
                }
                state.calls.append(failed_call)
                if observation is not None:
                    observation["contributes_to_provider_call_count"] = True
                _record_latest_operation_call_locked(
                    state, refusal_key, failed_call, sequence
                )
            return "worker_unavailable", None

        if not isinstance(document, Mapping) or set(document) != {
            "status",
            "headers",
            "body_b64",
            "call",
        }:
            return invalid_broker_response()
        response_headers = document.get("headers")
        if not isinstance(response_headers, Mapping) or not isinstance(
            document.get("call"), Mapping
        ):
            return invalid_broker_response()
        trusted_names = [
            name
            for name in response_headers
            if isinstance(name, str)
            and name.lower() == operations.TRUSTED_RESPONSE_URL_HEADER
        ]
        if len(trusted_names) > 1:
            return invalid_broker_response()
        if trusted_names:
            if operation_id != shim.PAGE_FETCH_OPERATION:
                return invalid_broker_response()
            try:
                response_url = operations.validate_https_url(
                    response_headers[trusted_names[0]],
                    max_length=2000,
                    field="response_url",
                )
            except operations.OperationError:
                return invalid_broker_response()
            response_headers = dict(response_headers)
            response_headers.pop(trusted_names[0])
            response_headers[operations.TRUSTED_RESPONSE_URL_HEADER] = response_url
            document = {**document, "headers": response_headers}
        call = dict(document["call"])
        with state.lock:
            state.calls.append(call)
            if observation is not None:
                observation["contributes_to_provider_call_count"] = True
            latest = _record_latest_operation_call_locked(
                state, refusal_key, call, sequence
            )
            refused = (
                call.get("error_code") in (
                    "budget_refused", "budget_exhausted",
                    "miner_credentials_unavailable", "miner_provider_not_configured",
                ) or call.get("outcome") == "refused"
            )
            if refused:
                state.refusals += 1
            newer_recovery = sequence < state.operation_recovery_sequence.get(
                refusal_key, -1
            )
            if refused and not newer_recovery:
                count, _previous_error = state.operation_refusals.get(
                    refusal_key, (0, "call_refused")
                )
                error_code = call.get("error_code")
                state.operation_refusals[refusal_key] = (
                    max(count + 1, MAX_REFUSED_FRAMES) if probe else count + 1,
                    (
                        error_code
                        if isinstance(error_code, str) and error_code in HTTP_ERROR_STATUS
                        else "call_refused"
                    ) if latest else _previous_error,
                )
                if state.operation_refusals[refusal_key][0] >= MAX_REFUSED_FRAMES:
                    state.operation_refusal_until[refusal_key] = (
                        self._monotonic() + REFUSAL_PROBE_COOLDOWN_SECONDS
                    )
            elif latest and (
                (
                    type(document["status"]) is int
                    and 200 <= document["status"] < 300
                    and call.get("error_code") is None
                )
                or (
                    call.get("provider_status") == document["status"]
                    and (
                        document["status"] in (400, 404, 422)
                        and call.get("error_code") not in (
                            "broker_unavailable", "provider_unavailable"
                        )
                        or document["status"] == 403
                        and call.get("error_code") == "provider_request_refused"
                    )
                )
            ):
                # A complete request-specific error also proves the old
                # credential refusal is stale. Billing may remain uncertain.
                state.operation_recovery_sequence[refusal_key] = sequence
                state.operation_refusals.pop(refusal_key, None)
                state.operation_refusal_until.pop(refusal_key, None)
            elif latest and probe:
                state.operation_refusal_until[refusal_key] = (
                    self._monotonic() + REFUSAL_PROBE_COOLDOWN_SECONDS
                )
        document = {
            **document,
            "headers": self._socket_response_headers(
                document, operation_id, sequence
            ),
        }
        return None, document

    @staticmethod
    def _responses_rate_limit_delay(call: Mapping[str, Any], retry: int) -> Optional[float]:
        common = (
            call.get("operation_id") == "openrouter.responses"
            and call.get("provider") == "openrouter"
            and call.get("error_code") == "provider_unavailable"
            and type(call.get("status")) is int
            and call.get("status") == 502
            and type(call.get("provider_status")) is int
            and call.get("provider_status") == 429
            and call.get("idempotent", False) is False
            and 0 <= retry < RESPONSES_RATE_LIMIT_RETRIES
        )
        proved_free = (
            call.get("funding_source") in ("host", "miner_key")
            and call.get("outcome") == "settled"
            and type(call.get("actual_microusd")) is int
            and call.get("actual_microusd") == 0
        )
        completed_price_unknown = (
            call.get("funding_source") in ("host", "miner_key")
            and call.get("outcome") == "uncertain"
            and call.get("completed_rate_limit_retryable") is True
            and "actual_microusd" not in call
            and type(call.get("reserved_microusd")) is int
            and call.get("reserved_microusd") == 0
            and isinstance(call.get("call_identity"), str)
            and contracts.SHA256_RE.fullmatch(call["call_identity"]) is not None
        )
        if not (common and (proved_free or completed_price_unknown)):
            return None
        if "retry_after_seconds" in call:
            hint = call["retry_after_seconds"]
            if type(hint) is not int or not 0 <= hint <= 3600:
                return None
            delay = float(hint)
        else:
            delay = RESPONSES_RATE_LIMIT_BACKOFF_SECONDS[retry]
        return delay + (
            1 + secrets.randbelow(RESPONSES_RATE_LIMIT_JITTER_MILLISECONDS)
        ) / 1000.0

    @staticmethod
    def _settled_responses_success_identity(
        document: Mapping[str, Any],
    ) -> Optional[str]:
        """Identify one exact successful response delivered after a hidden retry."""

        call = document.get("call")
        if not (
            type(document.get("status")) is int
            and document.get("status") == 200
            and isinstance(call, Mapping)
            and call.get("operation_id") == "openrouter.responses"
            and call.get("provider") == "openrouter"
            and call.get("funding_source") in ("host", "miner_key")
            and call.get("outcome") == "settled"
            and type(call.get("status")) is int
            and call.get("status") == 200
            and type(call.get("provider_status")) is int
            and call.get("provider_status") == 200
            and type(call.get("actual_microusd")) is int
            and call.get("actual_microusd") >= 0
            and call.get("error_code") is None
            and call.get("idempotent", False) is False
        ):
            return None
        identity = call.get("call_identity")
        if not isinstance(identity, str) or contracts.SHA256_RE.fullmatch(identity) is None:
            return None
        return identity

    def _dispatch(
        self,
        operation_id: str,
        parameters: Mapping[str, Any],
        timeout_ms: int,
        *,
        cancel_requested: Optional[Callable[[], bool]] = None,
        observation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Bridge one operation, retrying only proved free Responses throttles."""

        state = self._state
        if operation_id != "openrouter.responses" or str(
            state.lease.get("kind") or "execute"
        ) != "execute":
            return self._dispatch_once(
                operation_id,
                parameters,
                timeout_ms,
                cancel_requested=cancel_requested,
                observation=observation,
            )
        deadline = time.monotonic() + max(1, int(timeout_ms)) / 1000.0 + (
            MAX_PROVIDER_API_TIMEOUT_SECONDS - MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS
        )
        self._retain_handler_deadline(deadline, time.monotonic)
        attempt_timeout_ms = timeout_ms
        hidden_retry_call_identities: List[str] = []
        for retry in range(RESPONSES_RATE_LIMIT_RETRIES + 1):
            if retry and self._stopping.is_set():
                return error, document
            error, document = self._dispatch_once(
                operation_id,
                parameters,
                attempt_timeout_ms,
                cancel_requested=cancel_requested,
                observation=observation,
            )
            if error or document is None:
                return error, document
            recovered_by = self._settled_responses_success_identity(document)
            if (
                hidden_retry_call_identities
                and recovered_by is not None
                and recovered_by not in hidden_retry_call_identities
            ):
                with state.lock:
                    state.recovered_responses_retry_call_identities.update(
                        hidden_retry_call_identities
                    )
                hidden_retry_call_identities.clear()
            if retry == RESPONSES_RATE_LIMIT_RETRIES:
                return error, document
            delay = (
                self._responses_rate_limit_delay(document["call"], retry)
                if type(document.get("status")) is int
                and document.get("status") == 502
                else None
            )
            if delay is None:
                return error, document
            remaining_provider_seconds = deadline - time.monotonic() - delay - (
                MAX_PROVIDER_API_TIMEOUT_SECONDS - MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS
            )
            if remaining_provider_seconds < RESPONSES_RATE_LIMIT_MIN_PROVIDER_SECONDS:
                return error, document
            if self._stopping.wait(delay):
                return error, document
            remaining_provider_seconds = deadline - time.monotonic() - (
                MAX_PROVIDER_API_TIMEOUT_SECONDS - MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS
            )
            if remaining_provider_seconds < RESPONSES_RATE_LIMIT_MIN_PROVIDER_SECONDS:
                return error, document
            retry_identity = document["call"].get("call_identity")
            if (
                isinstance(retry_identity, str)
                and contracts.SHA256_RE.fullmatch(retry_identity) is not None
            ):
                hidden_retry_call_identities.append(retry_identity)
            attempt_timeout_ms = min(
                int(timeout_ms), max(1, int(remaining_provider_seconds * 1000))
            )
        raise AssertionError("unreachable Responses retry loop")

    @staticmethod
    def _socket_response_headers(
        document: Mapping[str, Any], operation_id: str, action_sequence: int
    ) -> Dict[str, Any]:
        """Add host-bound call proofs after removing untrusted copies."""

        headers = {
            name: value
            for name, value in dict(document["headers"]).items()
            if not (
                isinstance(name, str)
                and name.lower() in (
                    SETTLED_MICROUSD_HEADER, CALL_IDENTITY_HEADER,
                )
            )
        }
        call = document.get("call")
        provider_matches = isinstance(call, Mapping) and _call_provider_matches_operation(call, operation_id)
        # A valid managed call has a host-issued identity even when its result
        # is an error or its provider cost has not settled. Local refusals have
        # no call identity and never gain one here.
        if (
            isinstance(call, Mapping)
            and call.get("operation_id") == operation_id
            and type(call.get("action_sequence")) is int
            and call["action_sequence"] == action_sequence
            and provider_matches
            and operation_id not in operations.CATALOG_OPERATIONS
            and type(document.get("status")) is int
            and 100 <= document["status"] <= 599
            and isinstance(call.get("call_identity"), str)
            and contracts.SHA256_RE.fullmatch(call["call_identity"]) is not None
        ):
            headers[CALL_IDENTITY_HEADER] = call["call_identity"]
        if operation_id != "deepline.execute":
            return headers
        call = document.get("call")
        if not (
            isinstance(call, Mapping)
            and call.get("operation_id") == operation_id
            and call.get("provider") == "deepline"
            and type(call.get("action_sequence")) is int
            and call.get("action_sequence") == action_sequence
            and isinstance(call.get("call_identity"), str)
            and contracts.SHA256_RE.fullmatch(call["call_identity"]) is not None
        ):
            return headers
        headers[CALL_IDENTITY_HEADER] = call["call_identity"]
        try:
            body = json.loads(
                base64.b64decode(str(document["body_b64"]), validate=True)
            )
        except (TypeError, ValueError, UnicodeDecodeError):
            return headers
        if not isinstance(body, Mapping):
            return headers
        # Provider billing can remain pending after the gateway settles the
        # exact charge. The bound ledger proof is independent of those fields;
        # leave the provider body unchanged and expose the confirmed amount.
        if not (
            call.get("outcome") == "settled"
            and type(call.get("actual_microusd")) is int
            and call.get("actual_microusd") >= 0
        ):
            return headers
        headers[SETTLED_MICROUSD_HEADER] = str(call["actual_microusd"])
        return headers

    def _mark_observations_incomplete(self) -> None:
        try:
            with self._state.lock:
                self._state.worker_observations.failures.add("capture_incomplete")
        except Exception:
            pass

    def _begin_observation(self, *args: Any) -> Optional[Dict[str, Any]]:
        try:
            return self._begin_observation_unchecked(*args)
        except Exception:
            try:
                with self._state.lock:
                    self._state.worker_observations.failures.add("capture_incomplete")
            except Exception:
                pass
            return None

    def _begin_observation_unchecked(self, operation_id: str, parameters: Mapping[str, Any], timeout_ms: int, protocol: str) -> Dict[str, Any]:
        state = self._state
        with state.lock:
            ordinal = state.worker_observation_ordinal
            state.worker_observation_ordinal += 1
            state.worker_observations_pending.add(ordinal)
        return {"ordinal": ordinal, "action_sequence": None,
                "operation_id": operation_id, "request": {"admitted_parameters": dict(parameters), "timeout_ms": int(timeout_ms)},
                "protocol": protocol, "contributes_to_provider_call_count": False}

    def _dispatch_observed(self, *args: Any, observation: Optional[Dict[str, Any]], **kwargs: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            return self._dispatch(*args, observation=observation, **kwargs)
        except Exception as exc:
            if observation is not None:
                observation.update(kind="transport_failure", exception_class=type(exc).__name__)
            self._finish_observation(observation, None)
            raise

    def _finish_observation(self, observation: Optional[Dict[str, Any]], *args: Any, **kwargs: Any) -> None:
        if observation is None:
            return
        try:
            self._finish_observation_unchecked(observation, *args, **kwargs)
        except Exception:
            try:
                with self._state.lock:
                    self._state.worker_observations.failures.add("capture_incomplete")
                    self._state.worker_observations_pending.discard(observation["ordinal"])
            except Exception:
                pass

    def _finish_observation_unchecked(self, observation: Dict[str, Any], response: Optional[Dict[str, Any]], *, error: Optional[str] = None, document: Optional[Mapping[str, Any]] = None) -> None:
        # Evidence failures cannot change the response, retry, quota or accounting.
        state = self._state
        try:
            call = (document or {}).get("call") or {}
            valid_response = False
            if response is not None and "status" in response:
                try:
                    shim.parse_worker_response(response)
                    valid_response = True
                except shim.ShimError as exc:
                    observation.update(kind="invalid_response", exception_class=type(exc).__name__)
            receiptable_call = (
                response is not None and valid_response
                and type(response.get("status")) is int and 100 <= response["status"] <= 599
                and _call_provider_matches_operation(call, observation["operation_id"])
                and call.get("operation_id") == observation["operation_id"]
                and type(call.get("action_sequence")) is int
                and call["action_sequence"] == observation["action_sequence"]
                and isinstance(call.get("call_identity"), str)
                and contracts.SHA256_RE.fullmatch(call["call_identity"])
                and observation["operation_id"] not in operations.CATALOG_OPERATIONS
            )
            normal_paid_success = (receiptable_call and not error
                and 200 <= response["status"] < 300
                and call.get("outcome") == "settled" and not call.get("error_code"))
            if not normal_paid_success:
                outcome = {"protocol": observation["protocol"]}
                if response is not None:
                    outcome["response"] = response
                for key, value in (("exception_class", observation.get("exception_class")), ("error_code", error)):
                    if value is not None:
                        outcome[key] = value
                kind = observation.get("kind") or ("local_catalog" if observation["operation_id"] in operations.CATALOG_OPERATIONS else "nonstandard_response")
                if (kind == "nonstandard_response" and receiptable_call
                        and call.get("outcome") == "not_dispatched"
                        and call.get("error_code") == "provider_unavailable"
                        and isinstance(call.get("reason"), str)
                        and call["reason"] in training_observations.BROKER_NOT_DISPATCHED_REASONS
                        and training_observations._expected_not_dispatched_response(response)):
                    # This host broker disposition describes this call only.
                    # It does not prove the ledger or provider accounting.
                    outcome["broker_disposition"] = {
                        "outcome": "not_dispatched", "reason": call["reason"],
                    }
                prepared = state.worker_observations.prepare_observation({
                    **{key: observation[key] for key in ("ordinal", "action_sequence", "operation_id", "request", "contributes_to_provider_call_count")},
                    "kind": kind, "outcome": outcome,
                    "call_identity": call.get("call_identity"),
                    "privacy": {"redacted": False, "redactions": []},
                }, secrets=(state.lease_token,))
            with state.lock:
                if state.worker_observations_frozen:
                    return
                if receiptable_call:
                    raw_body = base64.b64decode(response["body_b64"], validate=True)
                    recorded = state.worker_observations.append_response_receipt({
                        "ordinal": observation["ordinal"], "action_sequence": observation["action_sequence"],
                        "operation_id": observation["operation_id"], "call_identity": call["call_identity"],
                        "protocol": observation["protocol"], "status": response["status"],
                        "body_sha256": "sha256:" + hashlib.sha256(raw_body).hexdigest(),
                        "body_bytes": len(raw_body), "write_state": "prepared", "socket_written_at": None,
                    })
                    if recorded:
                        observation["response_receipt_ordinal"] = observation["ordinal"]
                if not normal_paid_success:
                    state.worker_observations.append_prepared(prepared, local_refusal=kind == "local_refusal")
                state.worker_observations_pending.discard(observation["ordinal"])
        except Exception:
            with state.lock:
                state.worker_observations.failures.add("capture_incomplete")
                state.worker_observations_pending.discard(observation["ordinal"])

    def _observation_text(self, raw: bytes) -> Any:
        try:
            text = raw.decode("utf-8")
            try:
                return json.loads(text)
            except ValueError:
                return text
        except Exception:
            try:
                with self._state.lock:
                    self._state.worker_observations.failures.add("capture_incomplete")
            except Exception:
                pass
            return "[UNAVAILABLE_REQUEST_ENCODING]"

    def _record_parse_observation(self, request: Mapping[str, Any], response: Dict[str, Any], protocol: str, code: str) -> None:
        try:
            observation = self._begin_observation("unknown", request, 0, protocol)
            if observation is not None:
                observation.update(kind="local_parse_error", contributes_to_provider_call_count=False)
            self._finish_observation(observation, response, error=code)
        except Exception:
            with self._state.lock:
                self._state.worker_observations.failures.add("capture_incomplete")

    def handle_frame(
        self,
        raw: bytes,
        *,
        cancel_requested: Optional[Callable[[], bool]] = None,
        delivery_report: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """The judge shim's transport: one length-prefixed operation frame."""

        control_response = self._handle_signal_judgment_control(raw)
        if control_response is not None:
            return control_response
        control_response = self._handle_quota_control(raw)
        if control_response is not None:
            return control_response
        control_response = self._handle_decision_control(raw)
        if control_response is not None:
            return control_response
        try:
            operation_id, parameters, timeout_ms = shim.decode_operation_frame(
                raw, deepline_catalog=self._state.lease.get("deepline_catalog"),
                allow_full_thinking=self._state.lease.get("kind") == "execute"
                and self._state.lease.get("reasoning_capture_policy")
                == shim.FULL_THINKING_POLICY,
            )
        except shim.OperationFrameError as exc:
            code = str(exc) if str(exc) in shim.FRAME_ERROR_CODES else "invalid_frame"
            self._record_parse_observation({"raw_frame": self._observation_text(raw)}, {"error": code}, "frame", code)
            return shim.encode_worker_error(code)
        except operations.OperationError as exc:
            code = getattr(exc, "code", "invalid_request")
            code = code if code in shim.FRAME_ERROR_CODES else "invalid_request"
            self._record_parse_observation({"raw_frame": self._observation_text(raw)}, {"error": code}, "frame", code)
            return shim.encode_worker_error(code)
        observation = self._begin_observation(operation_id, parameters, timeout_ms, "frame")
        error, document = self._dispatch_observed(
            operation_id,
            parameters,
            timeout_ms,
            cancel_requested=cancel_requested,
            observation=observation,
        )
        if error:
            self._finish_observation(observation, {"error": error}, error=error)
            return shim.encode_worker_error(error)
        response = {
            "status": document["status"],
            "headers": document["headers"],
            "body_b64": document["body_b64"],
        }
        self._finish_observation(observation, response, document=document)
        if delivery_report is not None and observation is not None and "response_receipt_ordinal" in observation:
            delivery_report["receipt_ordinal"] = observation["response_receipt_ordinal"]
        return contracts.canonical_json(response).encode("utf-8")

    def handle_http(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        *,
        cancel_requested: Optional[Callable[[], bool]] = None,
        delivery_report: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """The miner contract: a provider's own HTTP request, sent over the socket without a credential."""

        try:
            operation_id, parameters = operations.match_request(
                method, url, body, headers,
                deepline_catalog=self._state.lease.get("deepline_catalog"),
                allow_full_thinking=self._state.lease.get("kind") == "execute"
                and self._state.lease.get("reasoning_capture_policy")
                == shim.FULL_THINKING_POLICY,
            )
        except operations.OperationError as exc:
            code = getattr(exc, "code", "invalid_request")
            status, payload = HTTP_ERROR_STATUS.get(code, 400), _http_error_body(code)
            self._record_parse_observation({"method": method, "url": url, "headers": dict(headers), "body_text": self._observation_text(body)}, {"status": status, "headers": {}, "body_b64": base64.b64encode(payload).decode("ascii")}, "http", code)
            return status, {}, payload
        operation_timeout_ms = operations.get_operation(operation_id).timeout_seconds * 1000
        observation = self._begin_observation(operation_id, parameters, operation_timeout_ms, "http")
        error, document = self._dispatch_observed(
            operation_id,
            parameters,
            operation_timeout_ms,
            cancel_requested=cancel_requested,
            observation=observation,
        )
        if error:
            status, payload = HTTP_ERROR_STATUS.get(error, 400), _http_error_body(error)
            self._finish_observation(observation, {"status": status, "headers": {}, "body_b64": base64.b64encode(payload).decode("ascii")}, error=error)
            return status, {}, payload
        response_headers = {
            str(name): str(value)
            for name, value in dict(document.get("headers") or {}).items()
            if str(name).lower()
            not in (
                operations.TRUSTED_RESPONSE_URL_HEADER,
            )
        }
        try:
            payload = base64.b64decode(str(document["body_b64"]), validate=True)
        except (ValueError, TypeError):
            payload = _http_error_body("worker_unavailable")
            if observation is not None:
                observation["kind"] = "invalid_response"
            self._finish_observation(observation, {"status": 503, "headers": {}, "body_b64": base64.b64encode(payload).decode("ascii")}, error="worker_unavailable", document=document)
            return 503, {}, payload
        self._finish_observation(observation, {"status": int(document["status"]), "headers": response_headers, "body_b64": base64.b64encode(payload).decode("ascii")}, document=document)
        if delivery_report is not None and observation is not None and "response_receipt_ordinal" in observation:
            delivery_report["receipt_ordinal"] = observation["response_receipt_ordinal"]
        return int(document["status"]), response_headers, payload

    def start(self) -> None:
        self._stopping.clear()
        server_self = self
        slots = threading.BoundedSemaphore(self._max_connections)
        capture_slot = threading.BoundedSemaphore(1)
        admissions = self._admissions

        def connection_closed(connection: socket.socket) -> bool:
            if server_self._stopping.is_set():
                return True
            try:
                readable, _, _ = select.select([connection], [], [], 0)
            except (OSError, ValueError):
                return True
            # The complete request has already been read. Readability now is
            # EOF or unexpected pipelined input; both cancel this one-shot RPC.
            return bool(readable)

        class BoundedServer(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

            def verify_request(self, request: Any, client_address: Any) -> bool:
                if slots.acquire(blocking=False):
                    permit, capture_only = slots, False
                elif capture_slot.acquire(blocking=False):
                    # One capture writer must make progress while all eight
                    # provider handlers are busy. This permit grants no dispatch.
                    permit, capture_only = capture_slot, True
                else:
                    return False
                try:
                    read_deadline = time.monotonic() + (
                        min(server_self._read_timeout_seconds,
                            lab_arena_checkpoint.TRAINING_SOCKET_TIMEOUT_SECONDS)
                        if capture_only else server_self._read_timeout_seconds
                    )
                    admission = _WorkerSocketAdmission(
                        permit, capture_only=capture_only,
                        completion_deadline=read_deadline,
                        read_deadline=read_deadline if capture_only else None,
                    )
                    with server_self._state.lock:
                        admissions[request] = admission
                except BaseException:
                    permit.release()
                    raise
                return True

            def admission_for(self, request: socket.socket) -> _WorkerSocketAdmission:
                with server_self._state.lock:
                    return admissions[request]

            def release_request(self, request: socket.socket) -> None:
                with server_self._state.lock:
                    admission = admissions.pop(request)
                    server_self._state.worker_handlers_inflight -= 1
                    admission.permit.release()

            def process_request(self, request: Any, client_address: Any) -> None:
                with server_self._state.lock:
                    server_self._state.worker_handlers_inflight += 1
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self.release_request(request)
                    raise

            def process_request_thread(self, request: Any, client_address: Any) -> None:
                server_self._handler_context.admission = self.admission_for(request)
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    del server_self._handler_context.admission
                    self.release_request(request)

        class HttpBridge(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "LabArenaWorker/1"
            sys_version = ""

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                return

            def _bridge(self) -> None:
                self.close_connection = True
                if self.headers.get("Transfer-Encoding"):
                    self._reject(400, "invalid_request")
                    return
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    self._reject(400, "invalid_request")
                    return
                if length < 0 or length > shim.MAX_FRAME_BYTES:
                    self._reject(413, "request_too_large")
                    return
                body = self.rfile.read(length) if length else b""
                host = (self.headers.get("Host") or "").strip()
                if not host or "/" in host or any(ch.isspace() for ch in host):
                    self._reject(400, "invalid_request")
                    return
                headers = {name: value for name, value in self.headers.items()}
                delivery_report: Dict[str, Any] = {}
                status, response_headers, payload = server_self.handle_http(
                    self.command,
                    "https://" + host + self.path,
                    body,
                    headers,
                    cancel_requested=lambda: connection_closed(self.connection),
                    delivery_report=delivery_report,
                )
                try:
                    self._answer(status, response_headers, payload)
                except BaseException as exc:
                    server_self._mark_socket_delivery(delivery_report, written=False, error=exc)
                    raise
                server_self._mark_socket_delivery(delivery_report, written=True)

            def _reject(self, status: int, code: str) -> None:
                payload = _http_error_body(code)
                server_self._mark_observations_incomplete()
                server_self._record_parse_observation(
                    {"method": self.command, "path": self.path, "headers": dict(self.headers)},
                    {"status": status, "headers": {}, "body_b64": base64.b64encode(payload).decode("ascii")},
                    "http", code,
                )
                self._answer(status, {}, payload)

            def send_error(self, code: int, message: Any = None, explain: Any = None) -> None:
                # BaseHTTPRequestHandler can refuse a request before _bridge.
                server_self._mark_observations_incomplete()
                super().send_error(code, message, explain)

            def _answer(self, status: int, headers: Mapping[str, str], payload: bytes) -> None:
                self.send_response(status)
                for name, value in headers.items():
                    if name.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                server_self._retain_socket_write_deadline(self.connection)
                self.end_headers()
                if self.command != "HEAD":
                    server_self._retain_socket_write_deadline(self.connection)
                    self.wfile.write(payload)
                self.wfile.flush()

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _bridge

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection = self.request
                admission = self.server.admission_for(connection)
                connection.settimeout(server_self._read_timeout_seconds)
                if not admission.capture_only:
                    try:
                        first = connection.recv(1, socket.MSG_PEEK)
                    except OSError:
                        return
                    if first and first[0] in HTTP_FIRST_BYTES:
                        try:
                            HttpBridge(connection, self.client_address, self.server)
                        except (OSError, ValueError):
                            return
                        return
                try:
                    header = _recv_exact(connection, 4, deadline=admission.read_deadline)
                    declared = int.from_bytes(header, "big")
                    training_frame = bool(declared & 0x80000000)
                    if admission.capture_only and not training_frame:
                        return
                    size = declared & 0x7fffffff if training_frame else declared
                    delivery_report: Dict[str, Any] = {}
                    frame_limit = lab_arena_checkpoint.MAX_TRAINING_FRAME_BYTES if training_frame else shim.MAX_FRAME_BYTES
                    if size < 2 or size > frame_limit:
                        server_self._mark_observations_incomplete()
                        server_self._record_parse_observation({"declared_frame_size": size}, {"error": "frame_too_large"}, "frame", "frame_too_large")
                        payload = shim.encode_worker_error("frame_too_large")
                    elif training_frame and not server_self._allow_training_frame(size):
                        payload = contracts.canonical_json({"recorded": False}).encode("utf-8")
                    elif training_frame:
                        # The extended prefix is reserved for optional capture;
                        # it cannot dispatch provider or other control work.
                        payload = server_self._handle_training_control(
                            _recv_exact(connection, size, deadline=admission.read_deadline)
                        )
                        if payload is None:
                            server_self._mark_observations_incomplete()
                            payload = contracts.canonical_json({"recorded": False}).encode("utf-8")
                    else:
                        payload = server_self.handle_frame(
                            _recv_exact(connection, size),
                            cancel_requested=lambda: connection_closed(connection),
                            delivery_report=delivery_report,
                        )
                    try:
                        server_self._retain_socket_write_deadline(connection)
                        connection.sendall(len(payload).to_bytes(4, "big") + payload)
                    except OSError as exc:
                        server_self._mark_socket_delivery(delivery_report, written=False, error=exc)
                        raise
                    server_self._mark_socket_delivery(delivery_report, written=True)
                except OSError:
                    return

        if self._path.exists():
            self._path.unlink()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._server = BoundedServer(str(self._path), Handler)
        os.chmod(self._path, 0o666)
        self._thread = threading.Thread(target=self._server.serve_forever, name="lab-arena-worker-socket", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._path.exists():
            self._path.unlink()

    def wait_for_handlers(self) -> bool:
        """Drain admitted handlers only within their existing remaining windows."""

        while True:
            with self._state.lock:
                if self._state.worker_handlers_inflight == 0:
                    return True
                deadline = max(admission.completion_deadline
                               for admission in self._admissions.values())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))


def _http_error_body(code: str) -> bytes:
    return json.dumps({"error": {"code": str(code)}}, sort_keys=True).encode("utf-8")


def _recv_exact(connection: socket.socket, size: int, *, deadline: Optional[float] = None) -> bytes:
    output = bytearray()
    while len(output) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker capture read deadline")
            connection.settimeout(remaining)
        chunk = connection.recv(min(65536, size - len(output)))
        if not chunk:
            raise OSError("connection closed")
        output.extend(chunk)
    return bytes(output)


# ---------------------------------------------------------------------------
# Executing one assignment
# ---------------------------------------------------------------------------


class SandboxRuntime(Protocol):
    def run_icp(self, spec: runtime.SandboxSpec, **kwargs: Any) -> runtime.SandboxResult: ...


@dataclass
class RunnerConfig:
    # None follows the Arena's current round (production); a value pins one round.
    round_id: Optional[str]
    identity: RunnerIdentity
    api: ArenaApiClient
    sandbox_runtime: SandboxRuntime
    image_cache: ImageCache
    source_cache: SourceCache
    work_dir: Path
    max_parallel_runs: int = DEFAULT_MAX_PARALLEL_RUNS
    slot_ceiling: int = contracts.RUNNER_SLOT_CEILING
    wall_clock_seconds: int = contracts.ICP_WALL_CLOCK_SECONDS
    # Waits between completion retries after a transport or server failure.
    completion_retry_seconds: Tuple[float, ...] = COMPLETION_RECOVERY_RETRY_SECONDS
    # Execution and scoring sandboxes can end while their last provider request
    # is still settling. This fixed 672-second retry-wait budget covers the
    # full 665-second provider API deadline, including host overhead. The retry
    # waits and bounded API calls use less than the minimum 25-minute lease TTL.
    accounting_open_retry_seconds: Tuple[float, ...] = COMPLETION_RECOVERY_RETRY_SECONDS
    evaluation_date: str = ""  # fallback only; every lease names the round's evaluation date
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    socket_root: Path = Path(DEFAULT_SOCKET_ROOT)
    agent_entrypoint_path: Path = AGENT_ENTRYPOINT_PATH
    checkpoint_module_path: Path = CHECKPOINT_MODULE_PATH
    automatic_capture_path: Path = AUTOMATIC_CAPTURE_PATH
    codex_module_path: Path = CODEX_MODULE_PATH
    web_bridge_path: Path = WEB_BRIDGE_PATH
    # Production always supplies verified slots. Direct test configurations may
    # omit the pool but cannot execute new proxy-enabled assignments.
    proxy_worker_pool: Any = None
    # Waits between retries of the same signed claim after a transport or
    # server failure. The service replays a committed claim by request ID.
    claim_retry_seconds: Tuple[float, ...] = (2.0, 5.0)
    # Reuse the process's normal idle claim cadence while active leases leave
    # spare capacity. A retry may become eligible without a local future ending.
    claim_poll_seconds: float = 30.0
    workspace_cache_lock: Any = None
    operational_logger: Any = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_parallel_runs, bool)
            or not isinstance(self.max_parallel_runs, int)
            or not 1 <= self.max_parallel_runs <= contracts.RUNNER_SLOT_CEILING
        ):
            raise RunnerError(
                "max_parallel_runs must be between 1 and %d"
                % contracts.RUNNER_SLOT_CEILING
            )
        self.work_dir = Path(self.work_dir)
        self.socket_root = Path(self.socket_root)
        self.agent_entrypoint_path = Path(self.agent_entrypoint_path)
        self.checkpoint_module_path = Path(self.checkpoint_module_path)
        self.automatic_capture_path = Path(self.automatic_capture_path)
        self.codex_module_path = Path(self.codex_module_path)
        self.web_bridge_path = Path(self.web_bridge_path)
        self.socket_root.mkdir(parents=True, exist_ok=True)


@contextmanager
def _attempt_web_egress(server: Any, worker: Any) -> Iterator[Any]:
    """Release an exit only after all attempt connections are proved closed."""
    try:
        server.start()
        yield server
    finally:
        try:
            server.stop()
        except BaseException:
            worker.quarantine()
            raise


def _record_trajectory(
    config: RunnerConfig, lease: Mapping[str, Any], lease_token: str,
    events: Sequence[Mapping[str, Any]],
) -> None:
    """Best-effort bounded delivery, independent of completion and scoring.

    Replay the same event IDs once on transport failure. A diagnostics outage
    must not turn completed model work into a failed assignment.
    """
    send = getattr(config.api, "trajectory", None)
    if not callable(send):  # embedded clients can predate this optional API
        return
    try:
        batches: List[List[Mapping[str, Any]]] = [[]]
        size = 20
        for item in events:
            # Match the server's conservative escaped-JSON size check.
            item_size = len(json.dumps(item, separators=(",", ":")).encode("utf-8")) + 1
            if batches[-1] and (len(batches[-1]) >= 32 or size + item_size > 48_000):
                batches.append([])
                size = 20
            batches[-1].append(item)
            size += item_size
        deadline = time.monotonic() + 15.0
        for batch in batches:
            if not batch:
                continue
            for attempt in range(2):
                if time.monotonic() >= deadline:
                    raise RunnerError("trajectory upload deadline")
                try:
                    response = send(str(lease["run_id"]), lease_token, batch)
                except Exception:
                    if attempt == 0:
                        continue
                    raise
                if not isinstance(response, Mapping) or response.get("accepted") != len(batch):
                    raise RunnerError("trajectory upload rejected")
                break
    except Exception as exc:
        # No response body, lease, model text or exception prose in host logs.
        print("Arena trajectory upload failed: %s" % type(exc).__name__, file=sys.stderr, flush=True)


def _log_training_capture_failure(kind: str, exc: Exception) -> None:
    """Keep capture diagnostics bounded and free of exception or response text."""
    def class_name(value: BaseException) -> str:
        name = type(value).__name__
        return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name) else "unknown"

    cause = exc.__cause__
    transport = class_name(cause) if isinstance(cause, httpx.HTTPError) else "-"
    status = _bounded_http_status(getattr(exc, "http_status", None))
    print(
        "Arena training %s capture failed: exception_class=%s http_status=%s transport_cause=%s"
        % (kind if kind in {"input", "observations"} else "unknown",
           class_name(exc), status if status is not None else "-", transport),
        file=sys.stderr, flush=True,
    )


def _record_training_input(
    config: RunnerConfig, lease: Mapping[str, Any], lease_token: str,
    document: Mapping[str, Any],
) -> None:
    """Best-effort input evidence; capture cannot change run/scoring results."""
    send = getattr(config.api, "training_input", None)
    if not callable(send):
        return
    try:
        for attempt in range(2):
            try:
                response = send(str(lease["run_id"]), lease_token, document)
            except Exception:
                if attempt == 0:
                    continue
                raise
            if not isinstance(response, Mapping) or response.get("status") != "stored":
                raise RunnerError("training input capture rejected", http_status=getattr(response, "http_status", None))
            break
    except Exception as exc:
        try:
            _log_training_capture_failure("input", exc)
        except Exception:
            pass


def _record_training_observations(config: RunnerConfig, lease: Mapping[str, Any], lease_token: str, state: RunState) -> Dict[str, Any]:
    summary = {"status": "unavailable", "observation_count": 0,
               "local_refusal_count": 0, "complete": False}
    try:
        with state.lock:
            state.worker_observations_frozen = True
            if state.worker_observations_pending or state.worker_handlers_inflight:
                state.worker_observations.failures.add("capture_incomplete")
            document = state.worker_observations.document()
            document["failures"] = training_observations.failure_tags_for_lease(
                document["failures"], lease)
        summary.update(observation_count=document["observation_count"],
                       local_refusal_count=document["local_refusal_count"], complete=document["complete"])
        send = getattr(config.api, "training_observations", None)
        if not callable(send):
            return summary
        expected_digest = contracts.document_hash(training_observations._validate_payload(
            document, secrets=(lease_token,),
        ))
        for attempt in range(2):
            try:
                result = send(str(lease["run_id"]), lease_token, document)
                if (not isinstance(result, Mapping) or result.get("status") != "stored"
                        or getattr(result, "http_status", 200) != 200
                        or result.get("payload_sha256") != expected_digest):
                    raise RunnerError("training observations rejected", http_status=getattr(result, "http_status", None))
                return {**summary, "status": "stored", "payload_sha256": result["payload_sha256"]}
            except Exception:
                if attempt:
                    verify = getattr(config.api, "verify_training_observations", None)
                    if callable(verify) and verify(str(lease["run_id"]), lease_token, expected_digest) is True:
                        return {**summary, "status": "stored", "payload_sha256": expected_digest}
                    raise
    except Exception as exc:
        try:
            _log_training_capture_failure("observations", exc)
        except Exception:
            pass
    return summary


class _ObservationJournal:
    """Best-effort serial uploads of bounded, immutable worker-only deltas."""

    FLUSH_SECONDS = 10.0
    STOP_TIMEOUT_SECONDS = 30.0
    FINAL_FLUSH_ATTEMPTS = 2

    def __init__(self, api: ArenaApiClient, lease: Mapping[str, Any], token: str, state: RunState) -> None:
        self.api, self.lease, self.token, self.state = api, lease, token, state
        self.stop_event = threading.Event()
        self.stop_deadline: Optional[float] = None
        self.thread = threading.Thread(target=self._work, name="arena-observation-journal", daemon=True)
        self.number = 0
        self.previous_sha256: Optional[str] = None
        self.cumulative_bytes = 0
        self.last_failures: Tuple[str, ...] = ()
        self.observation_cursor = self.receipt_cursor = self.agent_cursor = 0
        self.prepared_receipts: set[int] = set()
        self.pending: Optional[Tuple[dict, str, int, int, int, set[int]]] = None

    def start(self) -> None:
        if (not self.stop_event.is_set()
                and callable(getattr(self.api, "training_observation_segment", None))
                and isinstance(self.lease.get("claim_request_hash"), str)
                and contracts.SHA256_RE.fullmatch(self.lease["claim_request_hash"])
                and type(self.lease.get("lease_generation")) is int):
            self.thread.start()

    def stop(self) -> None:
        # The same uploader owns the final delta and any ambiguous retry.
        # An already admitted segment has its existing ten-second transport
        # window; at most two further requests fit this fixed stop allowance.
        if self.stop_deadline is None:
            self.stop_deadline = time.monotonic() + self.STOP_TIMEOUT_SECONDS
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(max(0.0, self.stop_deadline - time.monotonic()))

    def _flush_stopping(self) -> None:
        for _ in range(self.FINAL_FLUSH_ATTEMPTS):
            # Do not give a final segment a fresh lifetime after the stop
            # allowance. A timed-out uploader may finish its immutable send,
            # but cannot prepare another delta after the snapshot freezes.
            if (not self.stop_event.is_set() or self.stop_deadline is None
                    or self.stop_deadline - time.monotonic() < TRAINING_CAPTURE_TIMEOUT_SECONDS):
                return
            with self.state.lock:
                if self.state.worker_observations_frozen:
                    return
            if self.pending is None and not self._prepare():
                return
            with self.state.lock:
                if self.state.worker_observations_frozen:
                    return
            if self.stop_deadline - time.monotonic() < TRAINING_CAPTURE_TIMEOUT_SECONDS:
                return
            self._send_pending()

    def _prepare(self) -> bool:
        if self.number >= training_observations.MAX_SEGMENTS:
            return False
        with self.state.lock:
            if self.state.worker_observations_frozen:
                return False
            buffer = self.state.worker_observations
            observations = buffer.observations[self.observation_cursor:self.observation_cursor + 64]
            receipts = [dict(row) for row in buffer.response_receipts[self.receipt_cursor:self.receipt_cursor + 64]]
            agents = buffer.agent_observations[self.agent_cursor:self.agent_cursor + 32]
            changed = [dict(row) for row in buffer.response_receipts
                       if row["ordinal"] in self.prepared_receipts and row["write_state"] != "prepared"][:64]
            counts = (buffer.attempted, buffer.local_refusals,
                      buffer.response_receipt_count, buffer.agent_observation_count)
            failures = training_observations.failure_tags_for_lease(buffer.failures, self.lease)
        if not (observations or receipts or agents or changed or tuple(failures) != self.last_failures):
            return False
        while True:
            payload = {
                "observation_from": self.observation_cursor, "observations": observations,
                "receipt_from": self.receipt_cursor, "response_receipts": receipts,
                "receipt_updates": changed, "agent_from": self.agent_cursor,
                "agent_observations": agents, "observation_count": counts[0],
                "local_refusal_count": counts[1], "response_receipt_count": counts[2],
                "agent_observation_count": counts[3], "failures": failures,
            }
            try:
                cleaned_payload = training_observations._clean_segment_payload(
                    payload, secrets=(self.token,))
                segment, _ = training_observations.build_segment(
                    str(self.lease["run_id"]), str(self.lease.get("kind") or "execute"), cleaned_payload,
                    lease_generation=self.lease["lease_generation"],
                    claim_request_hash=self.lease["claim_request_hash"],
                    number=self.number, previous_sha256=self.previous_sha256,
                    cumulative_bytes=self.cumulative_bytes + len(
                        contracts.canonical_json(cleaned_payload).encode("utf-8")),
                    secrets=(self.token,),
                )
                break
            except Exception:
                # A single validated event fits the 16 MiB segment cap.
                largest = max((observations, receipts, agents, changed),
                              key=lambda rows: sum(len(contracts.canonical_json(row)) for row in rows))
                if not largest:
                    return False
                if len(largest) == 1:
                    if sum(bool(rows) for rows in (observations, receipts, agents, changed)) == 1:
                        return False
                    largest.clear()
                else:
                    del largest[len(largest) // 2:]
        document = {"number": self.number, "previous_sha256": self.previous_sha256,
                    "cumulative_bytes": segment["cumulative_bytes"],
                    "payload": segment["payload"]}
        prepared = (self.prepared_receipts | {row["ordinal"] for row in receipts
                                                if row["write_state"] == "prepared"}) - {
            row["ordinal"] for row in changed}
        self.pending = (document, segment["segment_sha256"], len(observations),
                        len(receipts), len(agents), prepared)
        return True

    def _work(self) -> None:
        try:
            self._upload_loop()
        finally:
            try:
                self._flush_stopping()
            except Exception:
                pass  # Capture faults cannot change completion or shutdown.

    def _upload_loop(self) -> None:
        while not self.stop_event.wait(self.FLUSH_SECONDS):
            while not self.stop_event.is_set():
                if self.pending is None:
                    try:
                        if not self._prepare():
                            break
                    except Exception:
                        return  # Capture faults cannot stop the run.
                if self.stop_event.is_set():
                    break  # The bounded terminal flush owns this pending send.
                with self.state.lock:
                    if self.state.worker_observations_frozen:
                        return
                if not self._send_pending():
                    break  # Retry the exact same bytes at the next tick.
                with self.state.lock:
                    buffer = self.state.worker_observations
                    more = (len(buffer.observations) > self.observation_cursor
                            or len(buffer.response_receipts) > self.receipt_cursor
                            or len(buffer.agent_observations) > self.agent_cursor)
                if not more:
                    break

    def _send_pending(self) -> bool:
        if self.pending is None:
            return False
        document, digest, observations, receipts, agents, prepared = self.pending
        try:
            result = self.api.training_observation_segment(
                str(self.lease["run_id"]), self.token, document)
            if (not isinstance(result, Mapping) or result.get("status") != "stored"
                    or result.get("number") != self.number
                    or result.get("segment_sha256") != digest):
                raise RunnerError("training observation journal rejected")
        except Exception:
            return False
        self.observation_cursor += observations
        self.receipt_cursor += receipts
        self.agent_cursor += agents
        self.prepared_receipts = prepared
        self.previous_sha256 = digest
        self.cumulative_bytes = document["cumulative_bytes"]
        self.last_failures = tuple(document["payload"]["failures"])
        self.number += 1
        self.pending = None
        return True


def _runtime_log_events(
    result: Optional[runtime.SandboxResult], lease_token: str, observed_at: str,
) -> List[Dict[str, Any]]:
    if result is None:
        return []
    events = []
    for stream in ("stdout", "stderr"):
        raw = getattr(result, stream)[:runtime.MAX_LOG_BYTES]
        # Redact before splitting so credentials cannot span stored chunks.
        text = trajectory.redact_text(raw.decode("utf-8", errors="replace"), secrets=(lease_token,))
        encoded = text.encode("utf-8")
        truncated = bool(getattr(result, stream + "_truncated")) or len(encoded) > runtime.MAX_LOG_BYTES
        text = encoded[:runtime.MAX_LOG_BYTES].decode("utf-8", errors="replace")
        sequence = 0
        while text:
            count = min(1024, len(text))
            while len(json.dumps(text[:count]).encode("utf-8")) > 6000:
                count //= 2
            chunk, text = text[:count], text[count:]
            events.append(trajectory.event(
                "runtime." + stream,
                {"text": chunk, "sequence": sequence,
                 "truncated": truncated,
                 "captured_bytes": len(raw)},
                occurred_at=observed_at,
            ))
            sequence += 1
    return events


def _record_runtime_failure(
    config: RunnerConfig,
    lease: Mapping[str, Any],
    lease_token: str,
    state: RunState,
    exc: BaseException,
    *,
    failure_stage: str,
    result: Optional[runtime.SandboxResult] = None,
    status: str = "abandoned",
) -> None:
    """Best-effort private failure boundary without exception prose."""

    if failure_stage not in ("setup", "runtime", "cleanup"):
        failure_stage = "runtime"
    observed_at = _timestamp(config.clock)
    try:
        error_content: Dict[str, Any] = {
            "status": status,
            "failure_stage": failure_stage,
            "error_class": type(exc).__name__[:64],
        }
        if isinstance(exc, RuntimeHostError):
            error_content["runtime_host_reason"] = exc.reason
            if exc.reason in ("sandbox_launch_failed", "sandbox_launcher_signaled", "sandbox_startup_timeout"):
                error_content.update({
                    "launch_exit_code": exc.launch_exit_code,
                    "launch_timed_out": exc.launch_timed_out,
                    "launch_stderr_truncated": exc.launch_stderr_truncated,
                    "launch_stderr": trajectory.redact_text(
                        exc.launch_stderr, secrets=(lease_token,),
                    ),
                })
        if result is not None:
            error_content.update({
                "resource_summary": {
                    "wall_seconds": float(result.wall_seconds),
                    "cpu_seconds": float(result.cpu_seconds),
                    "max_rss_bytes": int(result.max_rss_bytes),
                    "stdout_bytes": len(result.stdout),
                    "stderr_bytes": len(result.stderr),
                    "provider_call_count": len(state.calls),
                },
                "exit_code": result.exit_code,
                "timed_out": bool(result.timed_out),
            })
        events = [
            *_provider_error_events(state),
            *(
                _decision_events_and_summary(state, observed_at)
                if str(state.lease.get("kind") or "execute") == "execute"
                else []
            ),
            trajectory.event(
                "runtime.error",
                error_content,
                occurred_at=observed_at,
            ),
            *_runtime_log_events(result, lease_token, observed_at),
        ]
    except Exception as observation_exc:
        try:
            print(
                "Arena runtime error trajectory capture failed: %s"
                % type(observation_exc).__name__,
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        return
    _record_trajectory(config, lease, lease_token, events)


def _record_cleanup_failure(
    config: RunnerConfig,
    lease: Mapping[str, Any],
    lease_token: str,
    exc: BaseException,
) -> None:
    """Best-effort secondary cleanup boundary without replacing the cause."""

    try:
        event = trajectory.event(
            "runtime.cleanup_error",
            {
                "failure_stage": "cleanup",
                "error_class": type(exc).__name__[:64],
            },
            occurred_at=_timestamp(config.clock),
        )
    except Exception as observation_exc:
        try:
            print(
                "Arena cleanup trajectory capture failed: %s"
                % type(observation_exc).__name__,
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        return
    _record_trajectory(config, lease, lease_token, [event])


def _runtime_source_metadata(lease: Mapping[str, Any]) -> Dict[str, str]:
    """Small informational fields shared by start events and completions."""

    return {
        **runtime_version.SOURCE_METADATA,
        "gateway_claim_source_commit": runtime_version.commit_or_unknown(
            lease.get("gateway_source_commit")
        ),
        **reviewer_runtime.audit_metadata(lease),
    }


def _runtime_started_content(
    lease: Mapping[str, Any], *, wall_clock_limit_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    content: Dict[str, Any] = {
        "status": "starting",
        "runtime": "runsc",
        **_runtime_source_metadata(lease),
    }
    # Bind this recorded source identity to the exact lease, including a
    # reclaimed run with the same hotkey and attempt.
    generation = lease.get("lease_generation")
    if type(generation) is int and generation > 0:
        content["lease_generation"] = generation
    if wall_clock_limit_seconds is not None:
        content["wall_clock_limit_seconds"] = wall_clock_limit_seconds
    if lease.get("kind") == "score" and isinstance(lease.get("image_reference"), str):
        content["scorer_image_reference"] = lease["image_reference"]
    return content


class AssignmentExecutor:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config

    def execute(
        self, lease: Mapping[str, Any], lease_token: str, icp: Mapping[str, Any],
        *, before_finished: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run one ICP; let its owner retain the signed result before telemetry.

        Runner requires ``before_finished`` for production delivery. Direct
        embedded callers can inspect the returned envelope without delivery.
        """

        config = self._config
        state = RunState(lease=dict(lease), lease_token=lease_token)
        started_at = _timestamp(config.clock)
        run_dir: Optional[Path] = None
        socket_dir: Optional[Path] = None
        try:
            kind = str(lease.get("kind") or "execute")
            scoring_run = kind == "score"
            checkpoint_policy = lease.get("checkpoint_deadline_policy")
            if (
                checkpoint_policy is not None
                and checkpoint_policy not in contracts.CHECKPOINT_DEADLINE_PROFILES
            ):
                raise RunnerError("lease checkpoint deadline policy is unsupported")
            if checkpoint_policy is not None:
                checkpoint_wall, checkpoint_lease = (
                    contracts.CHECKPOINT_DEADLINE_PROFILES[checkpoint_policy]
                )
                if (
                    lease.get("icp_wall_clock_seconds") != checkpoint_wall
                    or lease.get("lease_ttl_seconds") != checkpoint_lease
                ):
                    raise RunnerError(
                        "lease checkpoint deadline differs from its signed round"
                    )
            signed_wall = lease.get(
                "scoring_wall_clock_seconds"
                if scoring_run else "icp_wall_clock_seconds"
            )
            if signed_wall is not None and (
                isinstance(signed_wall, bool)
                or not isinstance(signed_wall, int)
                or signed_wall < 30
            ):
                raise RunnerError("lease sandbox duration is invalid")
            wall_clock_seconds = (
                signed_wall if signed_wall is not None else (
                    contracts.SCORING_WALL_CLOCK_SECONDS if scoring_run
                    else config.wall_clock_seconds
                )
            )
            run_dir = Path(
                tempfile.mkdtemp(prefix="run-", dir=str(config.work_dir))
            )
            input_dir = run_dir / "input"
            output_dir = run_dir / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            # Keep the Unix socket under the short dedicated root.
            socket_dir = Path(
                tempfile.mkdtemp(prefix="la", dir=str(config.socket_root))
            )
            socket_path = socket_dir / runtime.SANDBOX_SOCKET_NAME
            if len(str(socket_path).encode("utf-8")) > MAX_SOCKET_PATH_BYTES:
                raise RunnerError(
                    "worker socket path exceeds %d bytes; set a shorter socket_root"
                    % MAX_SOCKET_PATH_BYTES
                )
            server = WorkerSocketServer(
                socket_path,
                config.api,
                state,
                quota_snapshot_upstream_limit=_quota_snapshot_upstream_limit(
                    wall_clock_seconds
                ),
            )
        except Exception as exc:
            try:
                _record_trajectory(config, lease, lease_token, [trajectory.event(
                    "runtime.started",
                    _runtime_started_content(lease),
                    occurred_at=started_at,
                )])
            except Exception as observation_exc:
                try:
                    print(
                        "Arena runtime start trajectory capture failed: %s"
                        % type(observation_exc).__name__,
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:
                    pass
            _record_training_observations(config, lease, lease_token, state)
            _record_runtime_failure(
                config, lease, lease_token, state, exc, failure_stage="setup"
            )
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)
            if socket_dir is not None:
                shutil.rmtree(socket_dir, ignore_errors=True)
            raise

        try:
            journal = _ObservationJournal(config.api, lease, lease_token, state)
        except Exception:
            journal = None

        def valid_checkpoint(candidate: bytes) -> bool:
            try:
                output_document_from_bytes(
                    candidate,
                    expected_schema_version=contact_policy.output_schema(lease),
                    require_intent_dates=(
                        lease.get("integrity_policy") != integrity.POLICY
                    ),
                )
            except (OutputInvalid, RecursionError):
                return False
            return True
        terminal = "judge_error" if scoring_run else "model_error"
        failure_diagnostic: Optional[Dict[str, str]] = None
        execution_output_diagnostic: Optional[Dict[str, str]] = None
        failure_detail: Any = ""
        provider_classification_ready = False
        output_document: Optional[Dict[str, Any]] = None
        result: Optional[runtime.SandboxResult] = None
        execution_diagnostic: Optional[Dict[str, Any]] = None
        checkpoint_transition: Optional[Dict[str, Any]] = None
        web_server = None
        worker = None
        runtime_entered = False
        primary_exception: Optional[BaseException] = None
        runtime_failure: Optional[Tuple[BaseException, str]] = None
        runtime_failure_status = "abandoned"
        evaluation_date = str(lease.get("evaluation_date") or config.evaluation_date)
        try:
            _record_trajectory(config, lease, lease_token, [trajectory.event(
                "runtime.started", _runtime_started_content(
                    lease, wall_clock_limit_seconds=wall_clock_seconds,
                ),
                occurred_at=started_at,
            )])
            if scoring_run:
                # A scoring assignment runs the Arena judge image on one accepted
                # output; the judge's provider calls cross the same socket.
                input_document = scoring.build_scoring_input(
                    scored_run_id=str(lease["scored_run_id"]), icp=icp, companies=list((lease.get("scored_output") or {}).get("companies") or []),
                    policy=lease["scorer_policy"], evaluation_date=evaluation_date,
                    contact_source_evidence=lease.get("contact_source_evidence"),
                    company_judgment_cache=lease.get("company_judgment_cache"),
                    prior_terminal_breakdowns=lease.get("prior_terminal_breakdowns"),
                )
                extra_environment = {shim.TRUSTED_SCORER_ENV: "1"}
            else:
                input_document = {
                    "schema_version": "leadpoet.lab_arena.icp_input.v1",
                    "icp": (
                        integrity.agent_visible_icp(icp, contacts_required=lease.get("contact_policy") == "contacts_v1")
                        if lease.get("integrity_policy") == "arena_integrity_v1" else dict(icp)
                    ),
                    "evaluation_date": evaluation_date,
                    "output_schema_version": contact_policy.output_schema(lease),
                    "company_limit": int(icp.get("max_companies") or 5),
                    "provider_operations": sorted(operations.OPERATIONS),
                }
                # harness.run_icp receives only this nested object. Announce
                # the same pinned schema that validates its returned companies.
                input_document["icp"]["output_schema_version"] = (
                    input_document["output_schema_version"]
                )
                extra_environment = {}
                if quality_policy.enabled(lease):
                    input_document["company_quality_policy"] = quality_policy.POLICY
                    input_document["company_requirements"] = {
                        "company_linkedin": "matching LinkedIn company page required",
                        "state": "headquarters state required for United States companies; full name or abbreviation, including DC",
                    }
                    # The public harness receives only document['icp']; carry
                    # the announced requirements through that actual boundary.
                    input_document["icp"].update({
                        "company_quality_policy": quality_policy.POLICY,
                        "company_requirements": dict(input_document["company_requirements"]),
                    })
                if intent_details_policy.enabled(lease):
                    input_document["intent_details_policy"] = (
                        intent_details_policy.POLICY
                    )
                    input_document["icp"]["intent_details_policy"] = (
                        intent_details_policy.POLICY
                    )
                if lease.get("scrapingdog_configured") is True:
                    extra_environment["SCRAPINGDOG_API_KEY"] = operations.SCRAPINGDOG_RUNTIME_HANDLE
            if lease.get("deepline_catalog") is not None:
                input_document["deepline_catalog"] = lease["deepline_catalog"]
                if not scoring_run:
                    input_document["provider_operations"] = sorted(
                        set(operations.OPERATIONS) | set(operations.CATALOG_OPERATIONS)
                    )
            if (lease.get("kind") == "execute"
                    and lease.get("reasoning_capture_policy") == shim.FULL_THINKING_POLICY):
                # The shim needs a local parse hint; the worker's lease remains
                # the authority for the same request after it crosses the socket.
                extra_environment[shim.REASONING_POLICY_ENV] = shim.FULL_THINKING_POLICY
            (input_dir / runtime.INPUT_FILE_NAME).write_text(json.dumps(input_document, sort_keys=True), encoding="utf-8")
            _record_training_input(config, lease, lease_token, input_document)
            staged_agent_entrypoint = (
                None
                if scoring_run
                else _stage_agent_entrypoint(config.agent_entrypoint_path, run_dir)
            )
            staged_checkpoint_module = (
                None if scoring_run else _stage_agent_entrypoint(
                    config.checkpoint_module_path, run_dir,
                    filename="lab_arena_checkpoint.py",
                )
            )
            staged_automatic_capture = (
                None if scoring_run else _stage_agent_entrypoint(
                    config.automatic_capture_path, run_dir,
                    filename="automatic_capture.py",
                )
            )
            # Both assignment kinds use the service-selected trusted Python
            # image. Execute assignments add the admitted source bundle under
            # read-only mounts; no miner image metadata is accepted.
            image_reference = str(lease.get("image_reference") or "")
            _check_runtime_image(image_reference, str(lease["image_digest"]))
            image_exporter = None
            if leased_images.is_ecr_reference(image_reference):
                image_exporter = leased_images.leased_image_exporter(
                    config.api,
                    str(lease["run_id"]),
                    lease_token,
                )
            with ExitStack() as resources:
                if image_exporter is None:
                    image_context = config.image_cache.acquire(
                        str(lease["image_digest"]),
                        image_reference,
                    )
                else:
                    image_context = config.image_cache.acquire(
                        str(lease["image_digest"]),
                        image_reference,
                        exporter=image_exporter,
                    )
                rootfs = resources.enter_context(image_context)
                source_dir = dependency_dir = None
                if not scoring_run:
                    source_dir, dependency_dir = resources.enter_context(
                        config.source_cache.acquire(
                            str(lease["run_id"]),
                            lease_token,
                            lease.get("source_ref"),
                            lease.get("submission_id"),
                            lease.get("source_size_bytes"),
                        )
                    )
                staged_web_bridge = None
                staged_codex_module = None
                proxy_execution = (
                    lease.get("parallel_twenty_icp_execution") is True
                    or lease.get("execution_sequence_policy")
                    == contracts.BASELINE_SCORED_FIRST_POLICY
                )
                if not scoring_run and proxy_execution:
                    if config.proxy_worker_pool is None:
                        raise RunnerError("verified proxy worker pool is required")
                    from lab_arena.web_egress import WebEgressServer
                    worker = resources.enter_context(config.proxy_worker_pool.acquire(timeout=5.0))
                    staged_web_bridge = _stage_agent_entrypoint(
                        config.web_bridge_path, run_dir, filename="web-egress-bridge.py"
                    )
                    staged_codex_module = _stage_agent_entrypoint(
                        config.codex_module_path, run_dir, filename="lab_arena_codex.py"
                    )
                    web_server = resources.enter_context(_attempt_web_egress(WebEgressServer(
                        socket_dir / runtime.SANDBOX_WEB_SOCKET_NAME,
                        proxy_url=worker.proxy_url,
                    ), worker))
                spec = runtime.SandboxSpec(
                    sandbox_id="arena-%s" % contracts.document_hash(lease["run_id"])[7:39],
                    rootfs_path=rootfs,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    socket_path=socket_path,
                    source_dir=source_dir,
                    dependency_dir=dependency_dir,
                    agent_entrypoint_path=(
                        staged_agent_entrypoint
                    ),
                    checkpoint_module_path=staged_checkpoint_module,
                    automatic_capture_path=staged_automatic_capture,
                    codex_module_path=staged_codex_module,
                    web_bridge_path=staged_web_bridge,
                    entry_command=runtime.SCORER_ENTRY_COMMAND if scoring_run else runtime.AGENT_ENTRY_COMMAND,
                    working_dir=runtime.SCORER_WORKING_DIR if scoring_run else runtime.AGENT_WORKING_DIR,
                    evaluation_date=evaluation_date,
                    random_seed=int(contracts.document_hash(lease["assignment_id"])[7:15], 16) % (2 ** 32),
                    wall_clock_seconds=wall_clock_seconds,
                    checkpoint_deadline_policy=(
                        checkpoint_policy if not scoring_run else None
                    ),
                    checkpoint_validator=(
                        valid_checkpoint if checkpoint_policy and not scoring_run
                        else None
                    ),
                    extra_environment=extra_environment,
                )
                server.start()
                if journal is not None:
                    try:
                        journal.start()
                    except Exception:
                        pass  # Optional evidence cannot change model execution.
                runtime_entered = True
                result = config.sandbox_runtime.run_icp(spec)
                if not scoring_run:
                    execution_diagnostic = _execution_diagnostic_from_stderr(
                        result.stderr
                    )
                    checkpoint_transition = _checkpoint_transition_from_stderr(
                        result.stderr
                    )
            if result.timed_out and not (
                checkpoint_policy in contracts.CHECKPOINT_DEADLINE_PROFILES
                and not scoring_run and result.output_bytes is not None
            ):
                terminal = "judge_timeout" if scoring_run else "model_timeout"
                if scoring_run:
                    failure_diagnostic = {
                        "stage": "sandbox",
                        "error_class": "judge_timeout",
                    }
                    if not result.output_error and result.output_bytes is not None:
                        try:
                            checkpoint = scoring.scoring_output_from_bytes(result.output_bytes)
                        except scoring.ScoringError:
                            checkpoint = None
                        output_document = _partial_company_judgment_output(checkpoint, lease)
                        if output_document is not None:
                            output_document = scoring.build_scoring_failure(
                                output_document["scored_run_id"], "judge_timeout",
                                detail=output_document.get("detail", ""),
                                reason=output_document.get("reason", ""),
                                **{key: output_document[key] for key in
                                   ("company_judgments", "terminal_breakdowns")
                                   if key in output_document},
                            )
                            if "reason" in output_document:
                                failure_diagnostic["reason"] = output_document["reason"]
            else:
                if result.output_error or result.output_bytes is None:
                    terminal = "judge_error" if scoring_run else ("invalid_output" if result.output_error else "model_error")
                    if scoring_run:
                        failure_diagnostic = {
                            "stage": "sandbox_output",
                            "error_class": (
                                "sandbox_output_error"
                                if result.output_error
                                else "missing_output"
                            ),
                        }
                        failure_detail = result.output_error or ""
                    elif result.output_error:
                        execution_output_diagnostic = {
                            "stage": "sandbox_output",
                            "error_class": "sandbox_output_error",
                        }
                elif scoring_run:
                    try:
                        output_document = scoring.scoring_output_from_bytes(result.output_bytes)
                    except scoring.ScoringError as exc:
                        terminal = "judge_error"
                        failure_diagnostic = {
                            "stage": "scoring_output",
                            "error_class": "scoring_output_invalid",
                        }
                        failure_detail = str(exc)
                    else:
                        if "failure" in output_document:
                            terminal = str(output_document["failure"])
                            failure_diagnostic = {
                                "stage": "scorer",
                                "error_class": terminal,
                                **(
                                    {"reason": output_document["reason"]}
                                    if scoring._validated_failure_reason(
                                        output_document.get("reason")
                                    )
                                    else {}
                                ),
                            }
                            failure_detail = output_document.get("detail", "")
                            output_document = _partial_company_judgment_output(output_document, lease)
                        else:
                            terminal = "accepted"
                else:
                    try:
                        output_document = output_document_from_bytes(
                            result.output_bytes,
                            expected_schema_version=contact_policy.output_schema(lease),
                            require_intent_dates=(
                                lease.get("integrity_policy") != integrity.POLICY
                            ),
                        )
                    except OutputInvalid as exc:
                        terminal = "invalid_output"
                        execution_output_diagnostic = {
                            "stage": "sandbox_output",
                            "error_class": "execution_output_invalid",
                            "reason": output_invalid_reason(exc),
                        }
                    else:
                        terminal = "accepted"
            provider_classification_ready = True
        except DependencyInstallInfrastructureError as exc:
            if scoring_run:  # the trusted scorer has no submitted dependency tree
                primary_exception = exc
                runtime_failure = (exc, "setup")
                raise
            terminal = "provider_error"
            output_document = None
            failure_diagnostic = {
                "stage": "provider_call",
                "error_class": "provider_unavailable",
                "reason": "provider_error",
            }
            _log_dependency_failure(str(lease["run_id"]), exc)
            _record_trajectory(config, lease, lease_token, [trajectory.event(
                "runtime.error",
                {"status": terminal, "failure_stage": "setup",
                 "error_class": type(exc).__name__,
                 "failure_kind": exc.failure_kind,
                 "diagnostic_detail": exc.diagnostic_detail},
                occurred_at=_timestamp(config.clock),
            )])
        except AgentDependencyError as exc:
            if scoring_run:  # the trusted scorer has no submitted dependency tree
                primary_exception = exc
                runtime_failure = (exc, "setup")
                raise
            terminal = "model_error"
            output_document = None
            _log_dependency_failure(str(lease["run_id"]), exc)
            _record_trajectory(config, lease, lease_token, [trajectory.event(
                "runtime.error",
                {"status": terminal, "failure_stage": "setup",
                 "error_class": type(exc).__name__,
                 "failure_kind": "source_error",
                 "diagnostic_detail": exc.diagnostic_detail},
                occurred_at=_timestamp(config.clock),
            )])
        except Exception as exc:
            primary_exception = exc
            if (
                isinstance(exc, runtime.SandboxCleanupError)
                and exc.result is not None
            ):
                result = exc.result
            runtime_failure = (
                exc,
                "cleanup"
                if isinstance(exc, runtime.SandboxCleanupError)
                else "runtime" if runtime_entered else "setup",
            )
            interrupted_partial = None
            if (
                scoring_run
                and isinstance(exc, RuntimeHostError)
                and exc.reason == "sandbox_launcher_signaled"
                and exc.launch_exit_code is not None
                and exc.launch_exit_code < 0
                and not exc.launch_timed_out
                and isinstance(exc.checkpoint_bytes, bytes)
                and len(exc.checkpoint_bytes) <= scoring.MAX_SCORING_OUTPUT_BYTES
            ):
                try:
                    checkpoint = scoring.scoring_output_from_bytes(exc.checkpoint_bytes)
                except (scoring.ScoringError, RecursionError):
                    checkpoint = None
                interrupted_partial = _partial_company_judgment_output(checkpoint, lease)
            if interrupted_partial is None:
                raise
            # Retain the host error during finalization. Report only its failed,
            # lease-bound terminal company subset through normal completion.
            # A later worker cleanup failure must still raise, never complete.
            primary_exception = None
            runtime_failure_status = "judge_error"
            terminal = "judge_error"
            output_document = interrupted_partial
            provider_classification_ready = True
            failure_diagnostic = {"stage": "sandbox", "error_class": "judge_error"}
        finally:
            pending_exception = primary_exception
            stop_exception = None
            try:
                server.stop()
            except Exception as exc:
                stop_exception = exc
            else:
                # stop() closes ingress but daemon handlers may still finish a
                # settled call. Keep the journal and final buffer open only
                # until the already-admitted handlers' existing deadlines.
                # This extra evidence wait cannot turn a valid result into an
                # abandoned run; stop() failure keeps its existing fatal path.
                try:
                    server.wait_for_handlers()
                except Exception:
                    with state.lock:
                        state.worker_observations.failures.add("capture_incomplete")
            if not scoring_run and result is not None:
                # Runtime copied these bounded bytes before cleanup. Import
                # after admitted socket handlers settle, including a cleanup
                # error carrying a result, before final journal/snapshot freeze.
                if result.automatic_capture_handoff_invalid:
                    with state.lock:
                        state.worker_observations.failures.add("capture_incomplete")
                for raw in result.automatic_capture_handoff:
                    if not server._allow_training_frame(len(raw)):
                        break
                    reply = server._handle_training_control(raw, replayed=True)
                    try:
                        recorded = json.loads(reply).get("recorded") is True
                    except (TypeError, ValueError, UnicodeError):
                        recorded = False
                    if not recorded:
                        with state.lock:
                            state.worker_observations.failures.add("capture_incomplete")
                capture_summary = result.automatic_capture_summary
                if capture_summary is not None:
                    # The exit report remains evidence, never run authority.
                    with state.lock:
                        state.worker_observations.append_agent_observation({
                            "id": len(state.worker_observations.agent_observations),
                            "kind": "state_snapshot", "observed_at": _timestamp(config.clock),
                            "after_action_sequence": None,
                            "depends_on": [], "data": {
                                "name": "automatic_capture_summary", "value": capture_summary,
                            },
                            "privacy": {"redacted": False, "redactions": []},
                        }, secrets=(state.lease_token,))
            if journal is not None:
                try:
                    journal.stop()
                except Exception:
                    pass
            observation_capture = _record_training_observations(config, lease, lease_token, state)
            if runtime_failure is not None:
                # Authenticated abandonment can immediately expire this lease.
                # Drain handlers and store the bounded capture before that signal.
                _record_runtime_failure(
                    config, lease, lease_token, state, runtime_failure[0],
                    failure_stage=runtime_failure[1], result=result,
                    status=(runtime_failure_status if stop_exception is None else "abandoned"),
                )
            shutil.rmtree(run_dir, ignore_errors=True)
            shutil.rmtree(socket_dir, ignore_errors=True)
            if stop_exception is not None:
                if pending_exception is None:
                    _record_runtime_failure(
                        config,
                        lease,
                        lease_token,
                        state,
                        stop_exception,
                        failure_stage="cleanup",
                        result=result,
                    )
                    raise stop_exception
                _record_cleanup_failure(
                    config, lease, lease_token, stop_exception
                )
                try:
                    print(
                        "Arena worker cleanup failed: %s"
                        % type(stop_exception).__name__,
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:
                    pass
        execution_output_observation = None
        if not scoring_run:
            try:
                execution_output_observation = _execution_output_observation(
                    result, output_document, terminal, execution_output_diagnostic,
                )
            except Exception:
                # Optional private diagnostics must not alter classification or completion.
                pass
        if provider_classification_ready:
            # A shared host key or account failure is infrastructure when it
            # prevents an output. An agent that handles the failure and still
            # returns a valid output has completed the assignment.
            with state.lock:
                completed_calls = tuple(state.calls)
                recovered_responses_retries = frozenset(
                    state.recovered_responses_retry_call_identities
                )
                latest_operation_calls = tuple(
                    state.operation_latest_calls.values()
                )
            unresolved_credentials, unresolved_providers = (
                _unresolved_operation_failures(
                    latest_operation_calls, recovered_responses_retries
                )
            )
            # A scorer can recover the same tool before a later judge failure.
            # Keep execution's historical account-failure precedence unchanged.
            credential_calls = latest_operation_calls if scoring_run else completed_calls
            miner_credentials_failed = any(
                call.get("funding_source") == "miner_key"
                and call.get("error_code") == "miner_credentials_unavailable"
                and (not scoring_run or call.get("call_identity") not in recovered_responses_retries)
                for call in credential_calls
            )
            miner_scoring_funding_failed = scoring_run and any(
                call.get("funding_source") == "miner_key"
                and call.get("error_code") in ("budget_refused", "budget_exhausted")
                for call in completed_calls
            )
            infrastructure_failures = tuple(
                call for call in completed_calls
                if call.get("call_identity") not in recovered_responses_retries
                and (
                    call.get("error_code") in (
                        "broker_unavailable",
                        "provider_unavailable",
                    )
                    or (
                        operations.provider_status_is_infrastructure(
                            call.get("provider_status")
                        )
                        and call.get("error_code") != "provider_request_refused"
                        and not (
                            call.get("funding_source") == "miner_key"
                            and call.get("provider_status") in (401, 402, 403)
                        )
                    )
                )
            )
            provider_infrastructure_failed = bool(infrastructure_failures)
            budget_stop_sequences = [
                call.get("action_sequence")
                for call in completed_calls
                if call.get("error_code") == "budget_refused"
                and call.get("outcome") == "refused"
                and call.get("reason")
                in contracts.PER_ICP_POLICY_BUDGET_STOP_REASONS
                and type(call.get("action_sequence")) is int
            ]
            infrastructure_sequences = [
                call.get("action_sequence") for call in infrastructure_failures
                if type(call.get("action_sequence")) is int
            ]
            per_icp_budget_stop_ended_attempt = (
                not scoring_run
                and lease.get("sourcing_cost_eligibility_policy")
                in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
                and bool(budget_stop_sequences)
                and len(infrastructure_sequences) == len(infrastructure_failures)
                and max(budget_stop_sequences)
                > max(infrastructure_sequences, default=-1)
            )
            with state.quota_condition:
                trusted_quota_failure = state.trusted_quota_failure
            if (
                miner_credentials_failed or miner_scoring_funding_failed
            ) and terminal != "accepted":
                terminal = "credential_error"
                output_document = _partial_company_judgment_output(output_document, lease) if scoring_run else None
            elif not scoring_run and trusted_quota_failure and terminal != "accepted":
                terminal = "provider_error"
                output_document = None
            elif per_icp_budget_stop_ended_attempt and terminal != "accepted":
                terminal = "budget_exhausted"
                output_document = None
            elif provider_infrastructure_failed and terminal != "accepted":
                terminal = "judge_error" if scoring_run else "provider_error"
                output_document = _partial_company_judgment_output(output_document, lease) if scoring_run else None
                if scoring_run and failure_diagnostic is None:
                    # A previous page-fetch failure must not replace the
                    # trusted scorer's more specific terminal diagnostic.
                    failure_diagnostic = {
                        "stage": "provider_call",
                        "error_class": "provider_unavailable",
                        "reason": "provider_error",
                    }
                    failure_detail = ""
            if (
                not scoring_run
                and terminal == "accepted"
                and output_document is not None
                and not output_document["companies"]
            ):
                # An empty document cannot be a completed research result if
                # the latest actual attempt on a requested tool still failed.
                # Successful later calls on that same tool clear this signal;
                # unrelated healthy tools do not.
                if unresolved_credentials:
                    terminal = "credential_error"
                    output_document = None
                elif unresolved_providers and not per_icp_budget_stop_ended_attempt:
                    terminal = "provider_error"
                    output_document = None
            if not scoring_run:
                if (
                    terminal == "accepted"
                    and result is not None
                    and result.checkpoint_output_invalid
                ):
                    failure_diagnostic = {
                        "stage": "sandbox_output",
                        "error_class": "execution_output_invalid",
                        "reason": "invalid_final_checkpoint",
                    }
                elif (
                    terminal in ("model_error", "model_timeout")
                    and result is not None
                    and result.checkpoint_output_invalid
                ):
                    failure_diagnostic = {
                        "stage": "sandbox_output",
                        "error_class": "execution_output_invalid",
                        "reason": "no_valid_checkpoint",
                    }
                elif terminal == "invalid_output":
                    failure_diagnostic = execution_output_diagnostic
                elif terminal == "provider_error" and trusted_quota_failure:
                    failure_diagnostic = {
                        "stage": "provider_call",
                        "error_class": "provider_unavailable",
                        "reason": "provider_error",
                    }
                else:
                    # Other provider and credential overrides own the final failure.
                    failure_diagnostic = None
        finished_at = _timestamp(config.clock)
        round_id = str(lease.get("round_id") or config.round_id or "")
        with state.lock:
            refusal_count = state.refusals
            local_refusal_count = state.local_refusals
            latest_operation_calls = tuple(state.operation_latest_calls.values())
            recovered_responses_retries = frozenset(
                state.recovered_responses_retry_call_identities
            )
        unresolved_credentials, unresolved_providers = (
            _unresolved_operation_failures(
                latest_operation_calls, recovered_responses_retries
            )
        )
        run_result = {
            "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
            "resource_summary": {
                "wall_seconds": float(result.wall_seconds) if result else 0.0,
                "cpu_seconds": float(result.cpu_seconds) if result else 0.0,
                "max_rss_bytes": int(result.max_rss_bytes) if result else 0,
                "stdout_bytes": len(result.stdout) if result else 0,
                "stderr_bytes": len(result.stderr) if result else 0,
                "provider_call_count": len(state.calls),
                "provider_refusal_count": refusal_count,
                "provider_local_refusal_count": local_refusal_count,
                "worker_observations": observation_capture,
                "provider_unresolved_failure_count": (
                    unresolved_credentials + unresolved_providers
                ),
                **_runtime_source_metadata(lease),
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "terminal_status": terminal,
        }
        if web_server is not None and worker is not None:
            counters = web_server.summary
            run_result["resource_summary"]["web_egress"] = {
                "policy_version": contracts.PROXY_EXECUTION_VERSION,
                "worker_slot": worker.slot_index,
                "exit_fingerprint": worker.exit_ip_fingerprint,
                "connection_count": counters["accepted_connection_count"],
                "upload_bytes": counters["client_to_web_bytes"],
                "download_bytes": counters["web_to_client_bytes"],
                "failure_count": counters["failure_count"],
                "active_limit_rejection_count": counters["active_limit_rejection_count"],
                "total_limit_rejection_count": counters["total_limit_rejection_count"],
                "byte_limit_rejection_count": counters["byte_limit_rejection_count"],
                "cleanup_block_rejection_count": counters["cleanup_block_rejection_count"],
            }
        if terminal == "credential_error" and any(
            call.get("call_identity") not in recovered_responses_retries
            and provider_credit_failure_reason(call) is not None
            for call in latest_operation_calls
        ):
            failure_diagnostic = {
                "stage": "provider_call",
                "error_class": "provider_unavailable",
                "reason": "provider_credit_exhausted",
            }
            run_result["failure_diagnostic"] = failure_diagnostic
        elif terminal == "credential_error" and any(
            call.get("call_identity") not in recovered_responses_retries
            and provider_authentication_failure_reason(call) is not None
            for call in latest_operation_calls
        ):
            failure_diagnostic = {
                "stage": "provider_call",
                "error_class": "provider_unavailable",
                "reason": "provider_authentication_failed",
            }
            run_result["failure_diagnostic"] = failure_diagnostic
        elif scoring_run and terminal in ("judge_error", "judge_timeout"):
            # Only fixed, contract-validated codes are persisted. The bounded,
            # redacted detail stays in the private operator log.
            if failure_diagnostic is None:
                failure_diagnostic = {
                    "stage": "scorer",
                    "error_class": terminal,
                }
            run_result["failure_diagnostic"] = failure_diagnostic
            _log_judge_failure(
                str(lease["run_id"]),
                stage=failure_diagnostic["stage"],
                error_class=failure_diagnostic["error_class"],
                detail=failure_detail,
            )
        elif not scoring_run and failure_diagnostic is not None:
            run_result["failure_diagnostic"] = failure_diagnostic
        if (
            not scoring_run
            and terminal == "accepted"
            and output_document is not None
            and checkpoint_transition is not None
            and checkpoint_transition["final_count"]
            == len(output_document["companies"])
            and checkpoint_transition["final_sha256"]
            == _checkpoint_output_sha256(output_document)
        ):
            # The model owns the historical classification. The trusted host
            # binds only its final count and hash, so this stays informational.
            run_result["checkpoint_transition"] = checkpoint_transition
        if not scoring_run and result is not None and (
            execution_diagnostic is not None
            or result.timed_out
            or result.exit_code not in (None, 0)
        ):
            _log_execution_diagnostic(
                str(lease["run_id"]), execution_diagnostic, result
            )
        judge_summary = None
        if (
            scoring_run and terminal == "judge_error"
            and failure_diagnostic is not None
            and failure_diagnostic.get("stage") == "scorer"
        ):
            try:
                judge_summary = _judge_exhausted_summary(failure_detail)
            except Exception:
                # Optional telemetry cannot change a completed score run.
                pass
        if scoring_run and output_document is not None and terminal in ("judge_error", "judge_timeout"):
            # Provider accounting can refine a timeout to infrastructure error.
            # Its retained partial envelope must name the final signed cause.
            output_document = dict(output_document, failure=terminal)
        body = {"run_id": lease["run_id"], "result": run_result, "output": output_document, "lease_token": lease_token}
        finished_event = trajectory.event(
            "runtime.finished",
            {"status": terminal, "resource_summary": run_result["resource_summary"],
             "started_at": started_at, "finished_at": finished_at,
             "exit_code": result.exit_code if result else None,
             "timed_out": bool(result and result.timed_out),
             "company_count": len((output_document or {}).get("companies") or []),
             "failure_diagnostic": failure_diagnostic,
             **({"judge_failure_summary": judge_summary} if judge_summary else {})},
            occurred_at=finished_at,
        )
        if execution_output_observation is not None:
            try:
                finished_event = trajectory.validate_event({
                    **finished_event,
                    "content": {**finished_event["content"], "execution_output": execution_output_observation},
                })
            except trajectory.TrajectoryError:
                # Keep the original lifecycle event if optional metadata reaches its cap.
                pass
        envelope = contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE,
            round_id=round_id,
            hotkey=config.identity.hotkey,
            body=body,
            timestamp=int(config.clock().timestamp()),
            sign_message=config.identity.sign,
        )
        if before_finished is not None:
            before_finished(envelope)
        _record_trajectory(config, lease, lease_token, [
            *_provider_error_events(state), *(
                _decision_events_and_summary(state, finished_at)
                if not scoring_run else []
            ), finished_event, *_runtime_log_events(result, lease_token, finished_at)])
        return envelope


def _partial_company_judgment_output(document, lease):
    """Keep only a failed, lease-bound subset; never infer an accepted ICP."""
    from lab_arena import company_judgments, quality_policy

    if (not isinstance(document, dict) or "failure" not in document
        or document.get("scored_run_id") != lease.get("scored_run_id")
        or not (document.get("company_judgments") or document.get("terminal_breakdowns"))):
        return None
    try:
        output = scoring.validate_scoring_output_document(document)
        if "terminal_breakdowns" in output:
            if (lease.get("kind") != "score"
                or company_judgments.enabled(lease)):
                return None
            scoring.validate_terminal_breakdowns_for_work_item(
                output["terminal_breakdowns"],
                icp=lease["icp"],
                companies=(lease.get("scored_output") or {}).get("companies") or [],
                policy=lease["scorer_policy"],
            )
        else:
            company_judgments.validate_new_company_judgments(
                output["company_judgments"],
                lease_context=lease.get("company_judgment_cache") or {},
                require_complete=False,
            )
    except (scoring.ScoringError, company_judgments.CompanyJudgmentError,
            contracts.ArenaContractError, KeyError, TypeError, ValueError):
        return None
    return output


def _check_runtime_image(image_reference: str, image_digest: str) -> None:
    """Require one pinned service-selected Python root filesystem."""

    if not image_reference:
        raise RunnerError("lease carries no image reference")
    reference = images.parse_reference(image_reference)
    if reference.digest != image_digest:
        raise RunnerError("lease image reference does not name the lease digest")


# ---------------------------------------------------------------------------
# Claim loop
# ---------------------------------------------------------------------------

# Keep polling unfinished judging after a stage closes. The gateway remains
# responsible for deciding which execution or recovery assignment is eligible.
WORKING_STATUSES = (
    "stage1", "stage1_scoring", "stage1_judged", "stage1_scored",
    "stage2", "stage2_closed", "stage2_scoring", "stage2_judged", "scored",
)


class Runner:
    # Keep an unproved worker drain alive in this process.  The validator may
    # drop its local Runner reference after a close error; its cache lease must
    # still block maintenance until a successful drain or process exit.
    _unproved_closes: set["Runner"] = set()

    def __init__(self, config: RunnerConfig) -> None:
        self._config = config
        self._executor = AssignmentExecutor(config)
        self._slots = threading.BoundedSemaphore(config.max_parallel_runs)
        self._pool = ThreadPoolExecutor(max_workers=config.max_parallel_runs, thread_name_prefix="lab-arena-slot")
        self.completed: List[Dict[str, Any]] = []
        self.abandoned = 0
        self._pinned = config.round_id is not None
        self._round_ids: List[str] = [config.round_id] if config.round_id else []
        self._pending_completion_cursor = ""
        self._pending_completion_next_poll = 0.0
        # A timed-out claim can commit after all short retries fail. Preserve
        # its exact signature/hash across outer polls, without a second allocator.
        # This is process-local: it does not recover claims lost before restart.
        self._pending_claims: Dict[str, Dict[str, Any]] = {}
        self._report_capacity()

    def _capacity_snapshot(self) -> Tuple[int, int]:
        # ready_slots records startup inventory. This reports the current
        # claim ceiling after permanent worker quarantine; active_runs is separate.
        pool = self._config.proxy_worker_pool
        quarantined = pool.quarantined if pool is not None else 0
        capacity = self._config.max_parallel_runs
        if pool is not None:
            capacity = min(capacity, pool.capacity - quarantined)
        return max(0, capacity), quarantined

    def _report_capacity(self) -> int:
        usable, quarantined = self._capacity_snapshot()
        log = self._config.operational_logger
        if log is not None:
            log.state(usable_slots=usable, quarantined_slots=quarantined)
        return usable

    def _report_claim_status(self, status: Any, round_id: Optional[str] = None) -> None:
        log = self._config.operational_logger
        if log is not None:
            log.state(
                last_claim_status=(
                    status if isinstance(status, str)
                    and status in validator_events.CLAIM_STATUSES else "other"
                ),
                last_claim_round_id=round_id,
            )

    @property
    def round_id(self) -> Optional[str]:
        """The first followed round (the pinned round, or the oldest running round)."""

        return self._round_ids[0] if self._round_ids else None

    @property
    def round_ids(self) -> List[str]:
        return list(self._round_ids)

    def refresh_round(self) -> Optional[str]:
        """Follow every running round the Arena reports; a new round is verified before any claim.

        Rounds overlap, so a runner started without ``--round-id`` asks
        ``/arena/v1/current`` at every idle poll for the rounds with work
        (executing or scoring), adopts each new round's configuration once,
        and drops rounds that ended. The daily rounds roll over without a
        restart.
        """

        if self._pinned:
            return self.round_id
        config = self._config
        if config.operational_logger is not None:
            config.operational_logger.state(scoring_state="discovering")
        current = config.api.current()
        rows = current.get("running_rounds") if isinstance(current, Mapping) else None
        if not isinstance(rows, list):
            row = current.get("round") if isinstance(current, Mapping) else None
            rows = [row] if isinstance(row, Mapping) else []
        wanted = []
        for row in rows:
            if isinstance(row, Mapping) and row.get("round_id") and str(row.get("status") or "") in WORKING_STATUSES:
                wanted.append(str(row["round_id"]))
        for round_id in wanted:
            if round_id not in self._round_ids:
                config.api.round(round_id)
        self._round_ids = wanted
        if config.operational_logger is not None:
            config.operational_logger.state(progress=True, last_poll_at=config.clock().isoformat())
            config.operational_logger.recovered("round_discovery")
        return self.round_id

    def claim_one(self, round_id: Optional[str] = None) -> Dict[str, Any]:
        config = self._config
        round_id = round_id or self.round_id
        if round_id is None:
            self._report_claim_status("no_open_round")
            return {"status": "no_open_round"}
        capacity = self._report_capacity()
        if capacity < 1:
            self._report_claim_status("no_local_capacity", round_id)
            return {"status": "no_pending"}
        envelope = self._pending_claims.get(round_id)
        if envelope is None:
            envelope = contracts.build_signed_request(
                scope=contracts.SCOPE_CLAIM,
                round_id=round_id,
                hotkey=config.identity.hotkey,
                body={"declared_parallelism": capacity,
                      "proxy_execution_version": contracts.PROXY_EXECUTION_VERSION,
                      "output_schema_versions": sorted(
                          SUPPORTED_OUTPUT_SCHEMA_VERSIONS
                      ),
                      # The singular field keeps compatibility with a 45-minute
                      # service. New services use the full supported profile list.
                      "checkpoint_deadline_policy": contracts.CHECKPOINT_DEADLINE_POLICY,
                      "checkpoint_deadline_policies": list(
                          contracts.CHECKPOINT_DEADLINE_POLICIES
                      )},
                timestamp=int(config.clock().timestamp()),
                sign_message=config.identity.sign,
            )
            self._pending_claims[round_id] = envelope
        retry_delays = iter(tuple(config.claim_retry_seconds))
        recovery_envelope = None
        while True:
            now = int(config.clock().timestamp())
            recovering = abs(now - envelope["timestamp"]) > contracts.REQUEST_TIMESTAMP_WINDOW_SECONDS
            request = envelope
            if recovering:
                # The fresh signature authorizes resolution. The original
                # signed envelope remains the only allocator identity and token.
                if (recovery_envelope is None or abs(now - recovery_envelope["timestamp"])
                    > contracts.REQUEST_TIMESTAMP_WINDOW_SECONDS):
                    recovery_envelope = contracts.build_signed_request(
                        scope=contracts.SCOPE_CLAIM, round_id=round_id,
                        hotkey=config.identity.hotkey,
                        body={"recover_claim": envelope}, timestamp=now,
                        sign_message=config.identity.sign,
                    )
                request = recovery_envelope
            try:
                response = config.api.claim(request)
                if not isinstance(response, Mapping):
                    raise RunnerError("Arena claim returned a non-object")
                status = response.get("status")
                if recovering:
                    recovery_status = response.get("claim_recovery")
                    if (response.get("claim_recovery_request_id") != envelope["request_id"]
                        or recovery_status not in ("resolved", "unresolved")
                        or (recovery_status == "resolved" and status not in ("leased", "no_work"))
                        or (recovery_status == "unresolved" and status in ("leased", "no_work"))):
                        # An older gateway rejects this recovery form before
                        # allocation. Its response cannot resolve the old ID.
                        raise RunnerError("Arena claim recovery unavailable",
                            http_status=getattr(response, "http_status", None))
                if status == "leased":
                    if (any(not isinstance(response.get(key), str) or not response[key].strip()
                            for key in ("run_id", "lease_token"))
                        or not isinstance(response.get("icp"), Mapping)):
                        raise RunnerError("Arena claim returned an incomplete lease")
                    if (response.get("request_id", envelope["request_id"]) != envelope["request_id"]
                        or response.get("round_id", round_id) != round_id):
                        raise RunnerError("Arena claim lease does not match its request")
                if not recovering and status != "leased" and (
                    abs(int(config.clock().timestamp()) - envelope["timestamp"]) > contracts.REQUEST_TIMESTAMP_WINDOW_SECONDS
                    or response.get("code") in (
                        "request_invalid:request timestamp outside window",
                        "contract:request timestamp outside window",
                    )
                ):
                    raise RunnerError("pending claim timestamp expired; allocation remains unresolved")
                if ((not recovering and isinstance(status, str) and status in _DEFINITE_CLAIM_STATUSES)
                    or (recovering and response.get("claim_recovery") == "resolved")):
                    self._pending_claims.pop(round_id)
                self._report_claim_status(response.get("status"), round_id)
                if (config.operational_logger is not None and isinstance(response.get("status"), str)
                    and response["status"] in (_IDLE_CLAIM_STATUSES | {"leased"})):
                    config.operational_logger.recovered("claim", round_id=round_id)
                return response
            except RunnerError as exc:
                if config.operational_logger is not None:
                    config.operational_logger.error("claim", exc, round_id=round_id,
                        http_status=exc.http_status, denial_code=_known_claim_denial_code(exc.denial_code))
                try:
                    delay = next(retry_delays)
                except StopIteration:
                    self._report_claim_status("request_failed", round_id)
                    raise exc from None
                time.sleep(max(0.0, float(delay)))

    def _usable_parallelism(self) -> int:
        """Return the current claim limit after permanent route quarantine."""
        return self._capacity_snapshot()[0]

    def _pending_completion_dir(self, *, create: bool = False) -> Optional[Path]:
        directory = self._config.work_dir / "pending-completions"
        if create:
            self._config.work_dir.mkdir(parents=True, exist_ok=True)
            directory.mkdir(mode=0o700, exist_ok=True)
        try:
            details = directory.lstat()
        except FileNotFoundError:
            return None
        if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != 0o700):
            raise RunnerError("pending completion directory is not private")
        return directory

    @staticmethod
    def _sync_completion_dir(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def _lock_completion(self, path: Path) -> Iterator[Optional[int]]:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            yield None
            return
        try:
            details = os.fstat(descriptor)
            if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != 0o600
                or details.st_size > contracts.COMPLETION_REQUEST_LIMITS.max_total_bytes):
                raise RunnerError("pending completion file is invalid")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield None
                return
            # A previous uploader can acknowledge/unlink after this file was
            # opened. Do not upload its old inode after acquiring the lock.
            try:
                current = path.lstat()
            except FileNotFoundError:
                yield None
                return
            if (current.st_dev, current.st_ino) != (details.st_dev, details.st_ino):
                yield None
                return
            yield descriptor
        finally:
            os.close(descriptor)

    def _read_pending_completion(self, descriptor: int, path: Path) -> Dict[str, Any]:
        from lab_arena.chain import wallet_signature_verifier
        maximum = contracts.COMPLETION_REQUEST_LIMITS.max_total_bytes
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as source:
            raw = source.read(maximum + 1)
        if len(raw) > maximum:
            raise RunnerError("pending completion file is too large")
        try:
            envelope = json.loads(raw.decode("utf-8"))
            envelope = contracts.validate_signed_request(
                envelope, expected_scope=contracts.SCOPE_COMPLETE,
                now=envelope["timestamp"], verify_signature=wallet_signature_verifier,
                expected_round_id=self._config.round_id,
            )
            if (envelope["hotkey"] != self._config.identity.hotkey
                or path.name != envelope["request_id"] + ".json"
                or set(envelope["body"]) != {"run_id", "lease_token", "result", "output"}
                or not isinstance(envelope["body"]["run_id"], str)
                or not envelope["body"]["run_id"]
                or not isinstance(envelope["body"]["lease_token"], str)
                or not envelope["body"]["lease_token"]):
                raise RunnerError("pending completion binding is invalid")
            contracts.validate_run_result(envelope["body"]["result"])
        except (ValueError, TypeError, KeyError, RecursionError, ArenaContractError):
            raise RunnerError("pending completion document is invalid") from None
        return envelope

    @contextmanager
    def _retain_completion(self, envelope: Mapping[str, Any]) -> Iterator[Path]:
        contracts.check_strict_document(envelope, contracts.COMPLETION_REQUEST_LIMITS)
        if (envelope.get("hotkey") != self._config.identity.hotkey
            or envelope.get("scope") != contracts.SCOPE_COMPLETE
            or (self._config.round_id is not None and envelope.get("round_id") != self._config.round_id)):
            raise RunnerError("completion identity or round is invalid")
        if not isinstance(envelope.get("request_id"), str) or not contracts.HEX32_RE.fullmatch(envelope["request_id"]):
            raise RunnerError("completion request identity is invalid")
        directory = self._pending_completion_dir(create=True)
        path = directory / (envelope["request_id"] + ".json")
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(raw) > contracts.COMPLETION_REQUEST_LIMITS.max_total_bytes:
            raise RunnerError("completion document is too large")
        descriptor, temporary = tempfile.mkstemp(prefix=".completion-", dir=directory)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with os.fdopen(os.dup(descriptor), "wb") as destination:
                destination.write(raw)
                destination.flush()
                os.fsync(destination.fileno())
            try:
                # Publish a fully written locked inode without overwriting an
                # earlier immutable envelope or creating a separate lock file.
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                with self._lock_completion(path) as existing:
                    if existing is None:
                        raise RunnerError("completion delivery is already active")
                    saved = self._read_pending_completion(existing, path)
                    if any(saved[key] != envelope[key] for key in ("scope", "round_id", "hotkey", "request_id", "body")):
                        raise RunnerError("completion request identity conflict")
                    yield path
            else:
                os.unlink(temporary)
                self._sync_completion_dir(directory)
                yield path
        finally:
            os.close(descriptor)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _acknowledge_completion(self, path: Path, result: Mapping[str, Any]) -> bool:
        status = result.get("status")
        if status in {"accepted", "failed"}:
            path.unlink()
        elif status in {"stale", "rejected"}:
            # Stale/denied leases cannot be replayed, but their completed paid
            # evidence is not proved persisted. Keep it privately for audit.
            archive = path.with_suffix("." + status + ".json")
            try:
                os.link(path, archive, follow_symlinks=False)
            except FileExistsError:
                source_details, saved_details = path.lstat(), archive.lstat()
                if ((source_details.st_dev, source_details.st_ino)
                    != (saved_details.st_dev, saved_details.st_ino)):
                    raise RunnerError("completion archive already exists") from None
            path.unlink()
        else:
            return False
        self._sync_completion_dir(path.parent)
        return True

    def _replay_pending_completions(self) -> None:
        if time.monotonic() < self._pending_completion_next_poll:
            return
        cadence = max(5.0, float(self._config.claim_poll_seconds))
        self._pending_completion_next_poll = time.monotonic() + cadence
        directory = self._pending_completion_dir()
        if directory is None:
            return
        paths = sorted((path for path in directory.iterdir()
                        if re.fullmatch(r"[0-9a-f]{32}\.json", path.name)), key=lambda path: path.name)
        ordered = ([path for path in paths if path.name > self._pending_completion_cursor]
                   + [path for path in paths if path.name <= self._pending_completion_cursor])
        for path in ordered[:1]:
            self._pending_completion_cursor = path.name
            try:
                with self._lock_completion(path) as descriptor:
                    if descriptor is None:
                        continue
                    envelope = self._read_pending_completion(descriptor, path)
                    refreshed = self._completion_envelope_for_attempt(envelope)
                    result = self._config.api.complete(refreshed)
                    if self._acknowledge_completion(path, result):
                        self.completed.append({"run_id": envelope["body"]["run_id"], "result": result})
                        if (result.get("status") in ("accepted", "failed")
                            and self._config.operational_logger is not None):
                            self._config.operational_logger.recovered(
                                "completion", run_id=envelope["body"].get("run_id"),
                                round_id=envelope.get("round_id"))
            except Exception as exc:
                # A bad file, denial or outage cannot erase the authenticated
                # receipt, resign unverified data, or stop later replay items.
                print("Lab Arena pending completion deferred: %s" % type(exc).__name__,
                      file=sys.stderr, flush=True)
            finally:
                self._pending_completion_next_poll = time.monotonic() + cadence

    def _run_lease(self, lease: Mapping[str, Any]) -> None:
        log = self._config.operational_logger
        phase = "execution"
        try:
            if lease.get("claim_recovery") == "resolved":
                try:
                    expires = datetime.fromisoformat(str(lease["lease_expires_at"]).replace("Z", "+00:00"))
                    if expires.tzinfo is None or expires <= self._config.clock():
                        raise ValueError("recovered lease expired")
                except (KeyError, TypeError, ValueError, OverflowError):
                    raise RunnerError("recovered lease expired or invalid before dispatch") from None
            def completion_started() -> None:
                nonlocal phase
                phase = "completion"

            result = self._execute_and_complete(lease, before_completion=completion_started)
            self.completed.append({"run_id": lease["run_id"], "result": result})
            if log is not None:
                log.state(progress=True, last_completion_at=self._config.clock().isoformat())
                log.recovered("execution", run_id=lease.get("run_id"), round_id=lease.get("round_id") or self.round_id)
        except Exception as exc:  # the attempt fails closed; the service expires the lease
            self.abandoned += 1
            if log is not None:
                log.error(phase, exc, run_id=lease.get("run_id"), round_id=lease.get("round_id") or self.round_id)
            if isinstance(exc, RuntimeHostError):
                detail = runtime_host_diagnostic(exc)
            elif isinstance(exc, leased_images.LeasedImageError):
                # The exporter retains only fixed, URL-free failure reasons.
                detail = "%s: %s" % (type(exc).__name__, str(exc))
            else:
                detail = type(exc).__name__
            print(
                "Lab Arena run abandoned: %s" % (
                    trajectory.redact_text(
                        runtime_host_private_diagnostic(exc),
                        secrets=(str(lease.get("lease_token") or ""),),
                    ) if isinstance(exc, RuntimeHostError) else detail
                ),
                file=sys.stderr,
                flush=True,
            )
            self.completed.append(
                {
                    "run_id": lease.get("run_id"),
                    "error": type(exc).__name__,
                    "detail": (detail if isinstance(exc, RuntimeHostError) else str(exc))[:200],
                }
            )
        finally:
            if log is not None:
                log.activity(-1)
            self._slots.release()
            self._report_capacity()

    def _execute_and_complete(
        self, lease: Mapping[str, Any], *, before_completion: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Retain the signed result before terminal telemetry and through delivery."""
        with ExitStack() as retained:
            pending_path = None
            pending_hash = None

            def before_finished(envelope: Mapping[str, Any]) -> None:
                nonlocal pending_path, pending_hash
                if before_completion is not None:
                    before_completion()
                if pending_path is not None:
                    raise RunnerError("completion retention was repeated")
                pending_path = retained.enter_context(self._retain_completion(envelope))
                pending_hash = contracts.document_hash(envelope)

            envelope = self._executor.execute(
                lease, str(lease["lease_token"]), lease["icp"],
                before_finished=before_finished,
            )
            if pending_path is None or contracts.document_hash(envelope) != pending_hash:
                raise RunnerError("completion differs from its retained receipt")
            # Keep the retained inode locked through terminal telemetry and
            # delivery so a concurrent replay cannot take over between them.
            result = self._deliver_completion_with_retries(envelope)
            self._acknowledge_completion(pending_path, result)
            return result

    def _complete_with_retries(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        with self._retain_completion(envelope) as path:
            result = self._deliver_completion_with_retries(envelope)
            self._acknowledge_completion(path, result)
            return result

    def _deliver_completion_with_retries(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """Keep the exact completion through bounded availability failures.

        The envelope is idempotent, so a lost response or a transient object
        store failure on the Arena costs a retry, not the whole sandbox run. An
        ``accounting_open`` document is retried separately while an in-flight
        provider call settles. One elapsed-time bound covers both retry paths.
        Other response documents and permanent HTTP denials are never retried.
        The service owns renewed lease expiry and returns ``stale`` if it ends.
        """

        failure_delays = iter(tuple(self._config.completion_retry_seconds))
        accounting_delays = iter(
            tuple(self._config.accounting_open_retry_seconds)
        )
        deadline = time.monotonic() + COMPLETION_RECOVERY_SECONDS
        while True:
            if time.monotonic() + API_TIMEOUT_SECONDS > deadline:
                raise RunnerError("completion recovery deadline exceeded")
            envelope = self._completion_envelope_for_attempt(envelope)
            try:
                result = self._config.api.complete(envelope)
            except Exception as exc:
                if not isinstance(exc, RunnerError) or not _retryable_claim_availability(exc):
                    raise
                if self._config.operational_logger is not None:
                    self._config.operational_logger.error("completion", exc,
                        run_id=envelope["body"].get("run_id"), round_id=envelope.get("round_id"))
                try:
                    delay = next(failure_delays)
                except StopIteration:
                    raise exc from None
            else:
                status = result.get("status")
                safe_status = (
                    status
                    if status
                    in ("accepted", "failed", "stale", "accounting_open", "rejected")
                    else "other"
                )
                print(
                    "Lab Arena completion status: %s" % safe_status,
                    flush=True,
                )
                if status != "accounting_open":
                    if self._config.operational_logger is not None:
                        log = self._config.operational_logger
                        if status == "rejected":
                            log.error("completion", RunnerError("completion denied"), reason="completion_denied",
                                run_id=envelope["body"].get("run_id"), round_id=envelope.get("round_id"))
                        elif status in ("accepted", "failed"):
                            log.recovered("completion", run_id=envelope["body"].get("run_id"), round_id=envelope.get("round_id"))
                    return result
                try:
                    delay = next(accounting_delays)
                except StopIteration:
                    raise RunnerError("completion remained accounting_open")
            delay = max(0.0, float(delay))
            if time.monotonic() + delay + API_TIMEOUT_SECONDS > deadline:
                raise RunnerError("completion recovery deadline exceeded")
            time.sleep(delay)

    def _completion_envelope_for_attempt(
        self, envelope: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Refresh an old completion signature without changing its identity or body."""

        now = int(self._config.clock().timestamp())
        if (
            abs(now - int(envelope["timestamp"]))
            < COMPLETION_SIGNATURE_REFRESH_AGE_SECONDS
        ):
            return envelope
        return contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE,
            round_id=str(envelope["round_id"]),
            hotkey=str(envelope["hotkey"]),
            body=envelope["body"],
            timestamp=now,
            request_id=str(envelope["request_id"]),
            sign_message=self._config.identity.sign,
        )

    def run_once(self, *, max_claims: int = 1000, stop_event: Any = None) -> int:
        """Refill free local slots until no lease remains or ``max_claims`` is met."""

        if stop_event is not None and stop_event.is_set():
            return 0
        self._replay_pending_completions()
        log = self._config.operational_logger
        pickup_failed = False
        if not self._pinned:
            try:
                self.refresh_round()
            except RunnerError as exc:
                if log is not None:
                    log.state(scoring_state="unavailable")
                    log.error("round_discovery", exc, http_status=exc.http_status,
                              denial_code=_known_claim_denial_code(exc.denial_code))
                _log_pickup_failure(
                    phase="round_discovery",
                    reason="request_failed",
                    http_status=exc.http_status,
                    denial_code=exc.denial_code,
                )
                return 0  # the Arena or the round is unavailable: poll again later
        taken = 0
        futures = set()

        def wait_for_completion(*, rescan_on_timeout: bool = False) -> bool:
            while futures:
                done, pending = wait(
                    futures,
                    timeout=self._config.claim_poll_seconds,
                    return_when=FIRST_COMPLETED,
                )
                self._replay_pending_completions()
                if done:
                    futures.clear()
                    futures.update(pending)
                    for future in done:
                        future.result()
                    return True
                if stop_event is not None and stop_event.is_set():
                    return False
                if rescan_on_timeout:
                    return True
            return False

        round_ids = list(self._round_ids)
        if not round_ids:
            self._report_claim_status("no_open_round")
        round_index = 0
        rescan_while_active = False
        while round_index < len(round_ids):
            round_id = round_ids[round_index]
            rescan_after_completion = False
            # Oldest round first: its deadline is nearer. Each round is claimed
            # until it has nothing to lease or this call reaches its claim cap.
            while taken < max_claims:
                if stop_event is not None and stop_event.is_set():
                    break
                self._replay_pending_completions()
                done = {future for future in futures if future.done()}
                futures.difference_update(done)
                for future in done:
                    future.result()
                if done and round_index > 0:
                    rescan_after_completion = True
                    break
                usable_parallelism = self._report_capacity()
                if usable_parallelism < 1:
                    self._report_claim_status("no_local_capacity", round_id)
                    break
                if not self._slots.acquire(blocking=False):
                    if not wait_for_completion():
                        break
                    if round_index > 0:
                        rescan_after_completion = True
                        break
                    continue
                if len(futures) >= usable_parallelism:
                    self._slots.release()
                    if not wait_for_completion():
                        break
                    if round_index > 0:
                        rescan_after_completion = True
                        break
                    continue
                try:
                    if log is not None:
                        log.state(scoring_state="claiming")
                    response = self.claim_one(round_id)
                    if log is not None:
                        status = response.get("status")
                        log.state(progress=isinstance(status, str) and status in (_IDLE_CLAIM_STATUSES | {"leased"}),
                                  last_poll_at=self._config.clock().isoformat())
                except RunnerError as exc:
                    pickup_failed = True
                    if log is not None:
                        log.error("claim", exc, round_id=round_id, http_status=exc.http_status,
                                  denial_code=_known_claim_denial_code(exc.denial_code))
                    self._slots.release()
                    _log_pickup_failure(
                        phase="claim",
                        reason="request_failed",
                        http_status=exc.http_status,
                        denial_code=exc.denial_code,
                    )
                    rescan_while_active = rescan_while_active or (
                        bool(futures) and _retryable_claim_availability(exc)
                    )
                    break
                if response.get("status") != "leased":
                    self._slots.release()
                    response_status = response.get("status")
                    idle = (
                        isinstance(response_status, str)
                        and response_status in _IDLE_CLAIM_STATUSES
                    )
                    if idle and log is not None:
                        log.recovered("claim", round_id=round_id)
                    if not idle:
                        pickup_failed = True
                        if log is not None:
                            log.error("claim", RunnerError("claim denied"), reason="claim_denied",
                                round_id=round_id, http_status=getattr(response, "http_status", None),
                                denial_code=_known_claim_denial_code(response.get("code")))
                        _log_pickup_failure(
                            phase="claim",
                            reason="claim_denied",
                            http_status=getattr(response, "http_status", None),
                            denial_code=response.get("code"),
                        )
                    rescan_while_active = rescan_while_active or (
                        isinstance(response_status, str)
                        and response_status in _ACTIVE_RESCAN_CLAIM_STATUSES
                        and bool(futures)
                    )
                    break
                taken += 1
                if log is not None:
                    log.recovered("claim", round_id=round_id)
                    log.activity(1)
                try:
                    future = self._pool.submit(self._run_lease, response)
                except BaseException:
                    if log is not None:
                        log.activity(-1)
                    raise
                futures.add(future)
            if rescan_after_completion:
                # A retry in an older round may have become eligible while
                # this round was busy. Keep all other active futures running.
                round_index = 0
                continue
            round_index += 1
            if (
                round_index == len(round_ids)
                and rescan_while_active
                and taken < max_claims
                and futures
                and not (stop_event is not None and stop_event.is_set())
                and wait_for_completion(rescan_on_timeout=True)
            ):
                round_index = 0
                rescan_while_active = False
            elif round_index == len(round_ids):
                break
        # Keep servicing completed receipts while the last healthy sandbox
        # runs. A busy run_once must not defer delivery until lease expiry.
        while futures:
            if not wait_for_completion():
                for future in futures:
                    future.result()
                break
        if log is not None:
            self._report_capacity()
            log.state(scoring_state="unavailable" if pickup_failed else "idle", progress=not pickup_failed)
        return taken

    def close(self) -> None:
        # A failed drain leaves worker state unproved. Keep the shared cache
        # lease until shutdown succeeds so maintenance cannot remove its files.
        try:
            self._pool.shutdown(wait=True)
        except BaseException:
            if self._config.workspace_cache_lock is not None:
                self._unproved_closes.add(self)
            raise
        self._unproved_closes.discard(self)
        lock = self._config.workspace_cache_lock
        if lock is not None:
            self._config.workspace_cache_lock = None
            lock.close()
