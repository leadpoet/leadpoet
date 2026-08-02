"""Opt-in, fail-closed Sentry error monitoring for Leadpoet HOST processes.

Same contract style as ``gateway/observability/otel_bootstrap.py``:

- Complete no-op unless ``LEADPOET_SENTRY_ENABLED`` is truthy AND
  ``LEADPOET_SENTRY_DSN`` is set. Only namespaced ``LEADPOET_SENTRY_*``
  variables are read; ambient ``SENTRY_*`` variables are never consulted —
  every option is passed to the SDK explicitly.
- Wiring failures are swallowed (class name logged, never values): error
  monitoring can never delay or break a runtime.
- Host processes only. Never wire this inside an attested enclave: enclave
  images are measured (PCR0) and have no general egress, and
  ``tests/test_sentry_boundary_guard.py`` fails CI if an enclave surface or
  enclave requirements file references Sentry.
- Errors only. Performance tracing, profiling, session tracking, request
  bodies, stack-local variables, and PII capture are hard-off, and
  framework/client auto-instrumentation stays disabled — coverage comes
  from the process-level excepthook, threading hook, and stdlib logging
  (ERROR and above), which every runtime already uses (bittensor's
  ``bt.logging``, uvicorn's ``uvicorn.error``, and asyncio's default task
  exception handler all route through stdlib logging).
- Every outgoing event and breadcrumb passes the fail-closed scrubber in
  ``sentry_scrubbing``; a scrub failure DROPS the event so nothing
  unscrubbed can leave the process. Training/trajectory data, prompts,
  benchmarks, and contact data are protected surfaces there.

Activation is an operator decision: installing ``sentry-sdk`` and setting
the environment variables changes deployed dependency sets, so it follows
the normal release process. Without the package or the variables every
``init_sentry`` call is inert.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from . import sentry_scrubbing

_TRUTHY = {"1", "true", "yes", "on"}

ENABLED_ENV = "LEADPOET_SENTRY_ENABLED"
DSN_ENV = "LEADPOET_SENTRY_DSN"
ENVIRONMENT_ENV = "LEADPOET_SENTRY_ENVIRONMENT"
RELEASE_ENV = "LEADPOET_SENTRY_RELEASE"
EXTRA_PROTECTED_ENV = "LEADPOET_SENTRY_EXTRA_PROTECTED_MODULES"
MESSAGE_MODE_ENV = "LEADPOET_SENTRY_MESSAGE_MODE"  # "scrub" (default) | "redact-all"

_REPO_ROOT = Path(__file__).resolve().parents[1]

_BUILD_SHA_ENV_NAMES = (
    "GITHUB_SHA",
    "GITHUB_COMMIT",
    "GIT_COMMIT_HASH",
    "GIT_COMMIT",
)
_EVENT_COALESCE_SECONDS = 300.0
_EVENT_COALESCE_MAX_SIGNATURES = 1024
_DYNAMIC_SIGNATURE_RE = re.compile(
    r"sha256:[0-9a-f]{64}"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|\b[0-9a-f]{16,64}\b"
    r"|\b\d+\b",
    re.I,
)

_state = {"initialized": False, "active": False}
_state_lock = threading.RLock()


def _safe_print(line: str) -> None:
    try:
        print(line, flush=True)
    except Exception:
        pass


def sentry_enabled() -> bool:
    return (
        os.getenv(ENABLED_ENV, "").strip().lower() in _TRUTHY
        and bool(os.getenv(DSN_ENV, "").strip())
    )


def _extra_protected_prefixes() -> Tuple[str, ...]:
    """Operator-supplied ADDITIONAL protected module prefixes.

    The environment can only widen redaction, never narrow the built-in set.
    """
    raw = os.getenv(EXTRA_PROTECTED_ENV, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _redact_all() -> bool:
    return os.getenv(MESSAGE_MODE_ENV, "").strip().lower() == "redact-all"


def _read_git_release(repo_root: Path) -> Optional[str]:
    """Best-effort release identity from ``.git`` without subprocesses."""
    try:
        git_dir = repo_root / ".git"
        if git_dir.is_file():
            marker = git_dir.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir:"):
                return None
            candidate = Path(marker.split(":", 1)[1].strip())
            git_dir = candidate if candidate.is_absolute() else repo_root / candidate
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.is_file():
                candidate = ref_path.read_text(encoding="utf-8").strip()
                return candidate if _looks_like_sha(candidate) else None
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.endswith(" " + ref):
                        candidate = line.split(" ", 1)[0]
                        return candidate if _looks_like_sha(candidate) else None
            return None
        return head if _looks_like_sha(head) else None
    except Exception:
        return None


def _looks_like_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower())


def _release_identity() -> Optional[str]:
    configured = os.getenv(RELEASE_ENV, "").strip()
    if configured:
        return configured
    for name in _BUILD_SHA_ENV_NAMES:
        candidate = os.getenv(name, "").strip().lower()
        if _looks_like_sha(candidate):
            return candidate
    return _read_git_release(_REPO_ROOT)


def _core_integrations() -> list[Any]:
    """Return the only integrations allowed to observe host runtimes."""
    from sentry_sdk.integrations.atexit import AtexitIntegration
    from sentry_sdk.integrations.dedupe import DedupeIntegration
    from sentry_sdk.integrations.excepthook import ExcepthookIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.threading import ThreadingIntegration

    logging_integration = LoggingIntegration(
        level=None,
        event_level=logging.ERROR,
        sentry_logs_level=None,
    )
    if logging_integration._handler is not None:
        logging_integration._handler.addFilter(_SentryLogRecordFilter())
    return [
        AtexitIntegration(),
        DedupeIntegration(),
        ExcepthookIntegration(),
        logging_integration,
        ThreadingIntegration(),
    ]


def _normalized_signature_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _DYNAMIC_SIGNATURE_RE.sub("<dynamic>", value)[:300]


def _event_signature(event: Dict[str, Any]) -> str:
    """Build a message-safe operational signature for grouping and throttling."""
    identity: Dict[str, Any] = {
        "logger": event.get("logger"),
        "level": event.get("level"),
    }
    exception = event.get("exception")
    exception_values = exception.get("values") if isinstance(exception, dict) else None
    if isinstance(exception_values, list):
        values = []
        for value in exception_values[-4:]:
            if not isinstance(value, dict):
                continue
            frames = []
            stacktrace = value.get("stacktrace")
            raw_frames = stacktrace.get("frames") if isinstance(stacktrace, dict) else None
            if isinstance(raw_frames, list):
                for frame in raw_frames[-8:]:
                    if isinstance(frame, dict):
                        frames.append(
                            [
                                frame.get("module"),
                                frame.get("filename"),
                                frame.get("function"),
                                frame.get("lineno"),
                            ]
                        )
            values.append(
                {
                    "type": value.get("type"),
                    "module": value.get("module"),
                    "value": _normalized_signature_text(value.get("value")),
                    "frames": frames,
                }
            )
        identity["exceptions"] = values
    else:
        logentry = event.get("logentry")
        if isinstance(logentry, dict):
            identity["logentry"] = _normalized_signature_text(
                logentry.get("message") or logentry.get("formatted")
            )
        else:
            identity["message"] = _normalized_signature_text(event.get("message"))
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _EventCoalescer:
    """Bound repeated signatures without hiding the first occurrence."""

    def __init__(
        self,
        *,
        window_seconds: float = _EVENT_COALESCE_SECONDS,
        max_signatures: int = _EVENT_COALESCE_MAX_SIGNATURES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_seconds = max(1.0, float(window_seconds))
        self._max_signatures = max(1, int(max_signatures))
        self._clock = clock
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def admit(self, signature: str) -> bool:
        now = self._clock()
        with self._lock:
            previous = self._seen.get(signature)
            if previous is not None and now - previous < self._window_seconds:
                return False
            self._seen[signature] = now
            self._seen.move_to_end(signature)
            while len(self._seen) > self._max_signatures:
                self._seen.popitem(last=False)
            return True


class _SentryLogRecordFilter(logging.Filter):
    """Coalesce retry storms before Sentry builds or serializes an event."""

    def __init__(self) -> None:
        super().__init__()
        self._coalescer = _EventCoalescer()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.msg if isinstance(record.msg, str) else type(record.msg).__name__
            identity = {
                "logger": record.name,
                "level": record.levelno,
                "path": record.pathname,
                "function": record.funcName,
                "line": record.lineno,
                "exception": (
                    record.exc_info[0].__name__
                    if record.exc_info and record.exc_info[0]
                    else None
                ),
                "message": _normalized_signature_text(
                    sentry_scrubbing.scrub_text(message, 300)
                ),
            }
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            return self._coalescer.admit(signature)
        except BaseException:
            # Monitoring must never interfere with the application's own log.
            return True


def _build_before_send(
    extra_prefixes: Tuple[str, ...], redact_all: bool
) -> Callable[[Any, Any], Optional[Dict[str, Any]]]:
    coalescer = _EventCoalescer()

    def _before_send(event: Any, hint: Any) -> Optional[Dict[str, Any]]:
        try:
            scrubbed = sentry_scrubbing.scrub_event(
                event, extra_prefixes=extra_prefixes, redact_all=redact_all
            )
            if scrubbed is None:
                return None
            signature = _event_signature(scrubbed)
            if not coalescer.admit(signature):
                return None
            # A scrubbed, message-safe fingerprint groups the same failure
            # across gateway and validator worker processes in Sentry.
            scrubbed["fingerprint"] = ["leadpoet-error", signature]
            return scrubbed
        except BaseException as exc:
            # Fail closed: an event we cannot fully scrub is never sent.
            _safe_print(
                "leadpoet_sentry_scrub_failed error=%s" % type(exc).__name__
            )
            return None

    return _before_send


def _build_before_breadcrumb(
    extra_prefixes: Tuple[str, ...], redact_all: bool
) -> Callable[[Any, Any], Optional[Dict[str, Any]]]:
    def _before_breadcrumb(crumb: Any, hint: Any) -> Optional[Dict[str, Any]]:
        try:
            return sentry_scrubbing.scrub_breadcrumb(
                crumb, extra_prefixes=extra_prefixes, redact_all=redact_all
            )
        except BaseException:
            # Fail closed: a breadcrumb we cannot scrub is dropped silently
            # (breadcrumbs are high-volume; no log line per drop).
            return None

    return _before_breadcrumb


def init_sentry(component: str, tags: Optional[Dict[str, str]] = None) -> bool:
    """Initialize error monitoring for this process. Never raises.

    Returns True only when a live client was installed. The first call wins;
    later calls in the same process are no-ops that return the first result.
    Complete no-op (False) when the ``LEADPOET_SENTRY_*`` gate is not
    satisfied or ``sentry-sdk`` is not installed.
    """
    with _state_lock:
        if _state["initialized"]:
            return bool(_state["active"])
        _state["initialized"] = True
        try:
            if not sentry_enabled():
                return False
            try:
                import sentry_sdk
            except Exception as exc:
                # Enabled by env but the optional dependency is missing: say so
                # once (class only) instead of failing the runtime.
                _safe_print(
                    "leadpoet_sentry_disabled sdk_import=%s" % type(exc).__name__
                )
                return False

            dsn = os.getenv(DSN_ENV, "").strip()
            environment = sentry_scrubbing.scrub_text(
                os.getenv(ENVIRONMENT_ENV, "").strip() or "production", 100
            )
            release = sentry_scrubbing.scrub_text(
                _release_identity() or "unknown", 200
            )
            extra_prefixes = _extra_protected_prefixes()
            redact_all = _redact_all()

            sentry_sdk.init(
            # Explicit options only — ambient SENTRY_* variables never apply.
            dsn=dsn,
            environment=environment,
            release=release,
            server_name="leadpoet-host",
            # Errors only: no performance tracing, profiling, or sessions.
            traces_sample_rate=None,
            propagate_traces=False,
            enable_logs=False,
            spotlight=False,
            keep_alive=False,
            auto_session_tracking=False,
            # Payload capture is hard-off; the scrubber is the second fence.
            send_default_pii=False,
            include_local_variables=False,
            max_request_body_size="never",
            attach_stacktrace=True,
            max_breadcrumbs=0,
            max_value_length=1024,
            debug=False,
            shutdown_timeout=2,
            # Explicit error-only integrations. Framework/DB/HTTP, argv,
            # module inventory, stdlib breadcrumbs, and Sentry Logs are off.
            default_integrations=False,
            auto_enabling_integrations=False,
            integrations=_core_integrations(),
            ignore_errors=[KeyboardInterrupt, asyncio.CancelledError, GeneratorExit],
            before_send=_build_before_send(extra_prefixes, redact_all),
            before_breadcrumb=_build_before_breadcrumb(extra_prefixes, redact_all),
            )
            safe_component = sentry_scrubbing.scrub_text(str(component), 100)
            sentry_sdk.set_tag("leadpoet.component", safe_component)
            for key, value in (tags or {}).items():
                sentry_sdk.set_tag(
                    "leadpoet.%s" % key,
                    sentry_scrubbing.scrub_text(str(value), 200),
                )
            _state["active"] = True
            _safe_print(
                "leadpoet_sentry_initialized component=%s release=%s"
                % (safe_component, release)
            )
            return True
        except BaseException as exc:
            # Never let error monitoring break the runtime. Class name only —
            # the message could contain the DSN.
            _safe_print("leadpoet_sentry_init_skipped error=%s" % type(exc).__name__)
            _state["active"] = False
            return False


def set_sentry_tag(key: str, value: Any) -> None:
    """Attach a low-cardinality tag to future events. Safe no-op when inactive."""
    if not _state["active"]:
        return
    try:
        import sentry_sdk

        sentry_sdk.set_tag(
            "leadpoet.%s" % key, sentry_scrubbing.scrub_text(str(value), 200)
        )
    except BaseException:
        pass


def _reset_for_tests() -> None:
    with _state_lock:
        _state["initialized"] = False
        _state["active"] = False
