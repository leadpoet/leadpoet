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
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

import httpx

from lab_arena import integrity, contact_policy, intent_details_policy, quality_policy
from lab_arena import contracts, images, lab_arena_checkpoint, leased_images, operations, runtime, scoring, shim, source_bundle
from lab_arena.contracts import ArenaContractError
from lab_arena.output import (
    OutputInvalid,
    SUPPORTED_OUTPUT_SCHEMA_VERSIONS,
    output_document_from_bytes,
    output_invalid_reason,
)
from lab_arena.runtime_host import DEFAULT_RUNNER_SOCKET_ROOT, RuntimeHostError, runtime_host_diagnostic

DEFAULT_MAX_PARALLEL_RUNS = 8  # compatibility for direct test/embedded configurations
MAX_PARALLEL_ENV = "LAB_ARENA_MAX_PARALLEL_RUNS"
DEFAULT_SOCKET_ROOT = str(DEFAULT_RUNNER_SOCKET_ROOT)
AGENT_ENTRYPOINT_PATH = Path(__file__).with_name("agent_entrypoint.py").resolve()
CHECKPOINT_MODULE_PATH = Path(__file__).with_name("lab_arena_checkpoint.py").resolve()
CODEX_MODULE_PATH = Path(__file__).with_name("lab_arena_codex.py").resolve()
WEB_BRIDGE_PATH = Path(__file__).with_name("web_egress_bridge.py").resolve()
MAX_REFUSED_FRAMES = 25  # after this many refused calls the worker answers a run's frames locally
QUOTA_SNAPSHOT_SCHEMA_VARIANTS = 2
QUOTA_SNAPSHOT_STARTUP_REQUESTS = 2
QUOTA_SNAPSHOT_CACHE_MILLISECONDS = 1000
QUOTA_SNAPSHOT_CACHE_SECONDS = QUOTA_SNAPSHOT_CACHE_MILLISECONDS / 1000.0
# A request on the worker socket is either a length-prefixed operation frame
# (first byte 0x00: the judge shim) or an HTTP request (an ASCII method).
HTTP_FIRST_BYTES = b"GPHDO"
HTTP_ERROR_STATUS = {"budget_exhausted": 402, "worker_unavailable": 503, "request_too_large": 413}
IMAGE_DIGEST_RE = __import__("re").compile(r"^(?:[a-z0-9][a-z0-9._/-]{0,200}@)?sha256:[0-9a-f]{64}$")
REQUIREMENT_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    r"(?:\[[A-Za-z0-9_,.-]{1,255}\])?"
    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]{0,127}"
    r"(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]{0,127})*)?$"
)
MAX_SOCKET_PATH_BYTES = 100
API_TIMEOUT_SECONDS = 30.0
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
_IDLE_CLAIM_STATUSES = frozenset({"no_pending", "no_open_round", "stage_closed"})

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
    if filename not in ("agent-entrypoint.py", "web-egress-bridge.py", "lab_arena_checkpoint.py", "lab_arena_codex.py"):
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

    def provider(self, run_id: str, lease_token: str, frame: Mapping[str, Any]) -> Dict[str, Any]:
        requested_timeout = frame.get("timeout_ms")
        timeout_seconds = API_TIMEOUT_SECONDS
        operation_id = frame.get("operation_id")
        operation = (
            operations.OPERATIONS.get(operation_id)
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
            timeout_seconds = max(
                API_TIMEOUT_SECONDS,
                min(
                    MAX_PROVIDER_API_TIMEOUT_SECONDS,
                    operations.BUDGET_ADMISSION_MAX_SECONDS
                    + operation_timeout
                    + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                    + PROVIDER_API_TIMEOUT_GRACE_SECONDS,
                ),
            )
        return self._post(
            "/arena/v1/runs/%s/provider" % run_id,
            frame,
            headers={"x-lab-arena-lease": lease_token},
            timeout_seconds=timeout_seconds,
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
        return self._post("/arena/v1/runs/%s/complete" % envelope["body"]["run_id"], envelope)

    def source(self, run_id: str, lease_token: str) -> bytes:
        """Download one bounded source archive under its active run lease."""

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
                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > source_bundle.MAX_SOURCE_ARCHIVE_BYTES:
                        raise RunnerError("run source exceeds the archive limit")
                    chunks.append(chunk)
        except RunnerError:
            raise
        except httpx.HTTPError as exc:
            raise RunnerError("Arena API transport failure: %s" % type(exc).__name__) from exc
        return b"".join(chunks)

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
                        % response.status_code
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
    """Install listed wheels in a size-capped temporary filesystem."""

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
                "size=%d,mode=0700,nosuid,nodev,noexec" % MAX_DEPENDENCY_BYTES,
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
        _lock_down_dependency_tree(install_target)
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


def _lock_down_dependency_tree(path: Path) -> None:
    total = 0
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
            total += int(details.st_size)
            if count > MAX_DEPENDENCY_FILES or total > MAX_DEPENDENCY_BYTES:
                raise AgentDependencyError("installed dependencies exceed the cache limit")
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
            _lock_down_dependency_tree(dependency_path)
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
    recovered_responses_retry_call_identities: set[str] = field(
        default_factory=set
    )
    action_sequence: int = 0
    refusals: int = 0  # refused calls answered by the Arena for this run
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

    def _dispatch_once(
        self,
        operation_id: str,
        parameters: Mapping[str, Any],
        timeout_ms: int,
        *,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        state = self._state
        with state.lock:
            sequence = state.action_sequence
            state.action_sequence += 1
            refused = state.refusals >= MAX_REFUSED_FRAMES
        if refused:
            # The run's quota or key keeps refusing: answer locally instead of
            # spending an Arena round trip and a ledger row on every request.
            return "budget_exhausted", None
        frame = {"operation_id": operation_id, "parameters": dict(parameters), "timeout_ms": int(timeout_ms), "action_sequence": sequence}
        while True:
            if self._cancelled(cancel_requested):
                return "worker_unavailable", None
            try:
                document = self._api.provider(
                    state.lease["run_id"], state.lease_token, frame
                )
            except RunnerError:
                # A gateway failure can occur before dispatch, after dispatch,
                # or after settlement. Never replay an unknown outcome.
                with state.lock:
                    state.calls.append(
                        {
                            "operation_id": operation_id,
                            "action_sequence": sequence,
                            "outcome": "unknown",
                            "error_code": "broker_unavailable",
                        }
                    )
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
                    state.calls.append(dict(document["call"]))
                return "worker_unavailable", None
        if not isinstance(document, Mapping) or set(document) != {
            "status",
            "headers",
            "body_b64",
            "call",
        }:
            return "worker_unavailable", None
        response_headers = document.get("headers")
        if not isinstance(response_headers, Mapping):
            return "worker_unavailable", None
        trusted_names = [
            name
            for name in response_headers
            if isinstance(name, str)
            and name.lower() == operations.TRUSTED_RESPONSE_URL_HEADER
        ]
        if len(trusted_names) > 1:
            return "worker_unavailable", None
        if trusted_names:
            if operation_id != shim.PAGE_FETCH_OPERATION:
                return "worker_unavailable", None
            try:
                response_url = operations.validate_https_url(
                    response_headers[trusted_names[0]],
                    max_length=2000,
                    field="response_url",
                )
            except operations.OperationError:
                return "worker_unavailable", None
            response_headers = dict(response_headers)
            response_headers.pop(trusted_names[0])
            response_headers[operations.TRUSTED_RESPONSE_URL_HEADER] = response_url
            document = {**document, "headers": response_headers}
        call = dict(document["call"])
        with state.lock:
            state.calls.append(call)
            if call.get("error_code") in ("budget_refused", "budget_exhausted", "miner_credentials_unavailable", "miner_provider_not_configured") or call.get("outcome") == "refused":
                state.refusals += 1
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
            )
        deadline = time.monotonic() + max(1, int(timeout_ms)) / 1000.0 + (
            MAX_PROVIDER_API_TIMEOUT_SECONDS - MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS
        )
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
                    SETTLED_MICROUSD_HEADER, CALL_IDENTITY_HEADER
                )
            )
        }
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
        billing = body.get("billing")
        if "billing" in body and not (
            billing is None or isinstance(billing, Mapping) and not billing
        ):
            return headers
        if not (
            call.get("outcome") == "settled"
            and type(call.get("actual_microusd")) is int
            and call.get("actual_microusd") >= 0
        ):
            return headers
        headers[SETTLED_MICROUSD_HEADER] = str(call["actual_microusd"])
        return headers

    def handle_frame(
        self,
        raw: bytes,
        *,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> bytes:
        """The judge shim's transport: one length-prefixed operation frame."""

        control_response = self._handle_quota_control(raw)
        if control_response is not None:
            return control_response

        try:
            operation_id, parameters, timeout_ms = shim.decode_operation_frame(raw)
        except shim.OperationFrameError as exc:
            return shim.encode_worker_error(str(exc) if str(exc) in shim.FRAME_ERROR_CODES else "invalid_frame")
        except operations.OperationError as exc:
            code = getattr(exc, "code", "invalid_request")
            return shim.encode_worker_error(code if code in shim.FRAME_ERROR_CODES else "invalid_request")
        error, document = self._dispatch(
            operation_id,
            parameters,
            timeout_ms,
            cancel_requested=cancel_requested,
        )
        if error:
            return shim.encode_worker_error(error)
        return contracts.canonical_json({
            "status": document["status"],
            "headers": document["headers"],
            "body_b64": document["body_b64"],
        }).encode("utf-8")

    def handle_http(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        *,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """The miner contract: a provider's own HTTP request, sent over the socket without a credential."""

        try:
            operation_id, parameters = operations.match_request(method, url, body, headers)
        except operations.OperationError as exc:
            code = getattr(exc, "code", "invalid_request")
            return HTTP_ERROR_STATUS.get(code, 400), {}, _http_error_body(code)
        operation_timeout_ms = operations.OPERATIONS[operation_id].timeout_seconds * 1000
        error, document = self._dispatch(
            operation_id,
            parameters,
            operation_timeout_ms,
            cancel_requested=cancel_requested,
        )
        if error:
            return HTTP_ERROR_STATUS.get(error, 400), {}, _http_error_body(error)
        response_headers = {
            str(name): str(value)
            for name, value in dict(document.get("headers") or {}).items()
            if str(name).lower()
            not in (
                operations.TRUSTED_RESPONSE_URL_HEADER,
                SETTLED_MICROUSD_HEADER,
                CALL_IDENTITY_HEADER,
            )
        }
        try:
            payload = base64.b64decode(str(document["body_b64"]), validate=True)
        except (ValueError, TypeError):
            return 503, {}, _http_error_body("worker_unavailable")
        return int(document["status"]), response_headers, payload

    def start(self) -> None:
        self._stopping.clear()
        server_self = self
        slots = threading.BoundedSemaphore(self._max_connections)

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
                return slots.acquire(blocking=False)

            def process_request(self, request: Any, client_address: Any) -> None:
                try:
                    super().process_request(request, client_address)
                except Exception:
                    slots.release()
                    raise

            def process_request_thread(self, request: Any, client_address: Any) -> None:
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    slots.release()

        class HttpBridge(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "LabArenaWorker/1"
            sys_version = ""

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                return

            def _bridge(self) -> None:
                self.close_connection = True
                if self.headers.get("Transfer-Encoding"):
                    self._answer(400, {}, _http_error_body("invalid_request"))
                    return
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    self._answer(400, {}, _http_error_body("invalid_request"))
                    return
                if length < 0 or length > shim.MAX_FRAME_BYTES:
                    self._answer(413, {}, _http_error_body("request_too_large"))
                    return
                body = self.rfile.read(length) if length else b""
                host = (self.headers.get("Host") or "").strip()
                if not host or "/" in host or any(ch.isspace() for ch in host):
                    self._answer(400, {}, _http_error_body("invalid_request"))
                    return
                headers = {name: value for name, value in self.headers.items()}
                status, response_headers, payload = server_self.handle_http(
                    self.command,
                    "https://" + host + self.path,
                    body,
                    headers,
                    cancel_requested=lambda: connection_closed(self.connection),
                )
                self._answer(status, response_headers, payload)

            def _answer(self, status: int, headers: Mapping[str, str], payload: bytes) -> None:
                self.send_response(status)
                for name, value in headers.items():
                    if name.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _bridge

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection = self.request
                connection.settimeout(server_self._read_timeout_seconds)
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
                    header = _recv_exact(connection, 4)
                    size = int.from_bytes(header, "big")
                    if size < 2 or size > shim.MAX_FRAME_BYTES:
                        payload = shim.encode_worker_error("frame_too_large")
                    else:
                        payload = server_self.handle_frame(
                            _recv_exact(connection, size),
                            cancel_requested=lambda: connection_closed(connection),
                        )
                    connection.sendall(len(payload).to_bytes(4, "big") + payload)
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


def _http_error_body(code: str) -> bytes:
    return json.dumps({"error": {"code": str(code)}}, sort_keys=True).encode("utf-8")


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
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
    completion_retry_seconds: Tuple[float, ...] = (2.0, 5.0)
    # Execution and scoring sandboxes can end while their last provider request
    # is still settling. This fixed 672-second retry-wait budget covers the
    # full 665-second provider API deadline, including host overhead. The retry
    # waits and bounded API calls still end before the 20-minute lease expires.
    accounting_open_retry_seconds: Tuple[float, ...] = (
        2.0,
        5.0,
        10.0,
        20.0,
        30.0,
        45.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
        80.0,
    )
    evaluation_date: str = ""  # fallback only; every lease names the round's evaluation date
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    socket_root: Path = Path(DEFAULT_SOCKET_ROOT)
    agent_entrypoint_path: Path = AGENT_ENTRYPOINT_PATH
    checkpoint_module_path: Path = CHECKPOINT_MODULE_PATH
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


class AssignmentExecutor:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config

    def execute(self, lease: Mapping[str, Any], lease_token: str, icp: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one leased ICP end to end and return the completion envelope."""

        config = self._config
        state = RunState(lease=dict(lease), lease_token=lease_token)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(config.work_dir)))
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        # Unix socket paths are limited to about 100 bytes, so the worker
        # socket lives in a short directory of its own, never under the run dir.
        socket_dir = Path(tempfile.mkdtemp(prefix="la", dir=str(config.socket_root)))
        socket_path = socket_dir / runtime.SANDBOX_SOCKET_NAME
        if len(str(socket_path).encode("utf-8")) > MAX_SOCKET_PATH_BYTES:
            shutil.rmtree(socket_dir, ignore_errors=True)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise RunnerError("worker socket path exceeds %d bytes; set a shorter socket_root" % MAX_SOCKET_PATH_BYTES)
        started_at = _timestamp(config.clock)
        kind = str(lease.get("kind") or "execute")
        scoring_run = kind == "score"
        checkpoint_policy = lease.get("checkpoint_deadline_policy")
        if checkpoint_policy is not None and checkpoint_policy not in contracts.CHECKPOINT_DEADLINE_PROFILES:
            raise RunnerError("lease checkpoint deadline policy is unsupported")
        if checkpoint_policy is not None:
            checkpoint_wall, checkpoint_lease = (
                contracts.CHECKPOINT_DEADLINE_PROFILES[checkpoint_policy]
            )
            if (
                lease.get("icp_wall_clock_seconds") != checkpoint_wall
                or lease.get("lease_ttl_seconds") != checkpoint_lease
            ):
                raise RunnerError("lease checkpoint deadline differs from its signed round")
        signed_wall = lease.get(
            "scoring_wall_clock_seconds" if scoring_run else "icp_wall_clock_seconds"
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
        server = WorkerSocketServer(
            socket_path,
            config.api,
            state,
            quota_snapshot_upstream_limit=_quota_snapshot_upstream_limit(
                wall_clock_seconds
            ),
        )

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
        output_document: Optional[Dict[str, Any]] = None
        result: Optional[runtime.SandboxResult] = None
        execution_diagnostic: Optional[Dict[str, Any]] = None
        checkpoint_transition: Optional[Dict[str, Any]] = None
        web_server = None
        worker = None
        evaluation_date = str(lease.get("evaluation_date") or config.evaluation_date)
        try:
            if scoring_run:
                # A scoring assignment runs the Arena judge image on one accepted
                # output; the judge's provider calls cross the same socket.
                input_document = scoring.build_scoring_input(
                    scored_run_id=str(lease["scored_run_id"]), icp=icp, companies=list((lease.get("scored_output") or {}).get("companies") or []),
                    policy=lease["scorer_policy"], evaluation_date=evaluation_date,
                    contact_source_evidence=lease.get("contact_source_evidence"),
                    company_judgment_cache=lease.get("company_judgment_cache"),
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
            (input_dir / runtime.INPUT_FILE_NAME).write_text(json.dumps(input_document, sort_keys=True), encoding="utf-8")
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
                            output_document = None
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
            # A shared host key or account failure is infrastructure when it
            # prevents an output. An agent that handles the failure and still
            # returns a valid output has completed the assignment.
            with state.lock:
                completed_calls = tuple(state.calls)
                recovered_responses_retries = frozenset(
                    state.recovered_responses_retry_call_identities
                )
            miner_credentials_failed = any(
                call.get("funding_source") == "miner_key"
                and call.get("error_code") == "miner_credentials_unavailable"
                for call in completed_calls
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
                == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
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
                output_document = None
            elif not scoring_run and trusted_quota_failure and terminal != "accepted":
                terminal = "provider_error"
                output_document = None
            elif per_icp_budget_stop_ended_attempt and terminal != "accepted":
                terminal = "budget_exhausted"
                output_document = None
            elif provider_infrastructure_failed and terminal != "accepted":
                terminal = "judge_error" if scoring_run else "provider_error"
                output_document = None
                if scoring_run and failure_diagnostic is None:
                    # A previous page-fetch failure must not replace the
                    # trusted scorer's more specific terminal diagnostic.
                    failure_diagnostic = {
                        "stage": "provider_call",
                        "error_class": "provider_unavailable",
                        "reason": "provider_error",
                    }
                    failure_detail = ""
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
        except DependencyInstallInfrastructureError as exc:
            if scoring_run:  # the trusted scorer has no submitted dependency tree
                raise
            terminal = "provider_error"
            output_document = None
            failure_diagnostic = {
                "stage": "provider_call",
                "error_class": "provider_unavailable",
                "reason": "provider_error",
            }
            _log_dependency_failure(str(lease["run_id"]), exc)
        except AgentDependencyError as exc:
            if scoring_run:  # the trusted scorer has no submitted dependency tree
                raise
            terminal = "model_error"
            output_document = None
            _log_dependency_failure(str(lease["run_id"]), exc)
        finally:
            server.stop()
            shutil.rmtree(run_dir, ignore_errors=True)
            shutil.rmtree(socket_dir, ignore_errors=True)
        finished_at = _timestamp(config.clock)
        round_id = str(lease.get("round_id") or config.round_id or "")
        run_result = {
            "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
            "resource_summary": {
                "wall_seconds": float(result.wall_seconds) if result else 0.0,
                "cpu_seconds": float(result.cpu_seconds) if result else 0.0,
                "max_rss_bytes": int(result.max_rss_bytes) if result else 0,
                "stdout_bytes": len(result.stdout) if result else 0,
                "stderr_bytes": len(result.stderr) if result else 0,
                "provider_call_count": len(state.calls),
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
        if scoring_run and terminal in ("judge_error", "judge_timeout"):
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
        body = {"run_id": lease["run_id"], "result": run_result, "output": output_document, "lease_token": lease_token}
        return contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE,
            round_id=round_id,
            hotkey=config.identity.hotkey,
            body=body,
            timestamp=int(config.clock().timestamp()),
            sign_message=config.identity.sign,
        )


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

# The round statuses in which assignments can be leased: both execution and
# scoring windows in the two-stage competition.
WORKING_STATUSES = ("stage1", "stage1_scoring", "stage2", "stage2_scoring")


class Runner:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config
        self._executor = AssignmentExecutor(config)
        self._slots = threading.BoundedSemaphore(config.max_parallel_runs)
        self._pool = ThreadPoolExecutor(max_workers=config.max_parallel_runs, thread_name_prefix="lab-arena-slot")
        self.completed: List[Dict[str, Any]] = []
        self.abandoned = 0
        self._pinned = config.round_id is not None
        self._round_ids: List[str] = [config.round_id] if config.round_id else []

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
        return self.round_id

    def claim_one(self, round_id: Optional[str] = None) -> Dict[str, Any]:
        config = self._config
        round_id = round_id or self.round_id
        if round_id is None:
            return {"status": "no_open_round"}
        capacity = self._usable_parallelism()
        if capacity < 1:
            return {"status": "no_pending"}
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
        retry_delays = iter(tuple(config.claim_retry_seconds))
        while True:
            try:
                return config.api.claim(envelope)
            except RunnerError as exc:
                try:
                    delay = next(retry_delays)
                except StopIteration:
                    raise exc from None
                time.sleep(max(0.0, float(delay)))

    def _usable_parallelism(self) -> int:
        """Return the current claim limit after permanent route quarantine."""

        config = self._config
        capacity = config.max_parallel_runs
        if config.proxy_worker_pool is not None:
            capacity = min(
                capacity,
                config.proxy_worker_pool.capacity
                - config.proxy_worker_pool.quarantined,
            )
        return max(0, capacity)

    def _run_lease(self, lease: Mapping[str, Any]) -> None:
        try:
            envelope = self._executor.execute(lease, str(lease["lease_token"]), lease["icp"])
            result = self._complete_with_retries(envelope)
            self.completed.append({"run_id": lease["run_id"], "result": result})
        except Exception as exc:  # the attempt fails closed; the service expires the lease
            self.abandoned += 1
            detail = runtime_host_diagnostic(exc) if isinstance(exc, RuntimeHostError) else type(exc).__name__
            print(
                "Lab Arena run abandoned: %s" % detail,
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
            self._slots.release()

    def _complete_with_retries(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """Deliver the signed completion; a transport or server failure is retried briefly.

        The envelope is idempotent, so a lost response or a transient object
        store failure on the Arena costs a retry, not the whole sandbox run. An
        ``accounting_open`` document is retried separately while an in-flight
        provider call settles. Other response documents are never retried.
        """

        failure_delays = iter(tuple(self._config.completion_retry_seconds))
        accounting_delays = iter(
            tuple(self._config.accounting_open_retry_seconds)
        )
        while True:
            envelope = self._completion_envelope_for_attempt(envelope)
            try:
                result = self._config.api.complete(envelope)
            except Exception as exc:
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
                    return result
                try:
                    delay = next(accounting_delays)
                except StopIteration:
                    raise RunnerError("completion remained accounting_open")
            time.sleep(max(0.0, float(delay)))

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
        if not self._pinned:
            try:
                self.refresh_round()
            except RunnerError as exc:
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
        round_index = 0
        rescan_while_active = False
        while round_index < len(round_ids):
            round_id = round_ids[round_index]
            # Oldest round first: its deadline is nearer. Each round is claimed
            # until it has nothing to lease or this call reaches its claim cap.
            while taken < max_claims:
                if stop_event is not None and stop_event.is_set():
                    break
                done = {future for future in futures if future.done()}
                futures.difference_update(done)
                for future in done:
                    future.result()
                usable_parallelism = self._usable_parallelism()
                if usable_parallelism < 1:
                    break
                if not self._slots.acquire(blocking=False):
                    if not wait_for_completion():
                        break
                    continue
                if len(futures) >= usable_parallelism:
                    self._slots.release()
                    if not wait_for_completion():
                        break
                    continue
                try:
                    response = self.claim_one(round_id)
                except RunnerError as exc:
                    self._slots.release()
                    _log_pickup_failure(
                        phase="claim",
                        reason="request_failed",
                        http_status=exc.http_status,
                        denial_code=exc.denial_code,
                    )
                    break
                if response.get("status") != "leased":
                    self._slots.release()
                    response_status = response.get("status")
                    idle = (
                        isinstance(response_status, str)
                        and response_status in _IDLE_CLAIM_STATUSES
                    )
                    if not idle:
                        _log_pickup_failure(
                            phase="claim",
                            reason="claim_denied",
                            http_status=getattr(response, "http_status", None),
                            denial_code=response.get("code"),
                        )
                    rescan_while_active = rescan_while_active or (
                        response_status == "no_pending" and bool(futures)
                    )
                    break
                taken += 1
                futures.add(self._pool.submit(self._run_lease, response))
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
        for future in futures:
            future.result()
        return taken

    def close(self) -> None:
        self._pool.shutdown(wait=True)
