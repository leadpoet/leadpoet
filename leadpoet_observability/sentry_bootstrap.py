"""Opt-in, fail-open Sentry observability for Leadpoet HOST processes.

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
- Terminal errors are captured at 100 percent. A small, bounded sample of
  explicitly named operational spans records stage latency; profiling,
  session tracking, request bodies, stack-local variables, PII capture, and
  framework/client auto-instrumentation remain hard-off. Error coverage also
  comes from the process-level excepthook, threading hook, and stdlib logging
  (ERROR and above).
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
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
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
TRACES_SAMPLE_RATE_ENV = "LEADPOET_SENTRY_TRACES_SAMPLE_RATE"

_DEFAULT_TRACES_SAMPLE_RATE = 0.01
_MAX_TRACES_SAMPLE_RATE = 0.10
_MAX_BREADCRUMBS = 50

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
        print(line, file=sys.stderr, flush=True)
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


def _trace_sample_rate() -> float:
    """Return a bounded success-trace rate; malformed values fail to default."""

    raw = os.getenv(TRACES_SAMPLE_RATE_ENV, "").strip()
    try:
        value = _DEFAULT_TRACES_SAMPLE_RATE if not raw else float(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_TRACES_SAMPLE_RATE
    return min(_MAX_TRACES_SAMPLE_RATE, max(0.0, value))


def scrub_sentry_value(value: Any, max_length: int = 200) -> Any:
    """Expose the bootstrap's fail-closed scalar scrubber to stdlib helpers."""

    try:
        return sentry_scrubbing.scrub_text(value, max_length)
    except BaseException:
        return sentry_scrubbing.REDACTED


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


def _event_signature(
    event: Dict[str, Any], *, include_semantic: bool = True
) -> str:
    """Build a message-safe operational signature for grouping and throttling."""
    identity: Dict[str, Any] = {}
    tags = event.get("tags")
    if (
        include_semantic
        and isinstance(tags, dict)
        and tags.get("leadpoet.failure_code")
    ):
        identity["semantic"] = {
            "component": tags.get("leadpoet.component"),
            "stage": tags.get("leadpoet.stage"),
            "failure_code": tags.get("leadpoet.failure_code"),
            "correlation_id": tags.get("leadpoet.correlation_id"),
            "restart_invocation_id": tags.get(
                "leadpoet.restart_invocation_id"
            ),
            "weight_correlation_id": tags.get(
                "leadpoet.weight_correlation_id"
            ),
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
        identity["logger"] = event.get("logger")
        identity["level"] = event.get("level")
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
    semantic_exception_coalescer = _EventCoalescer()

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
            tags = scrubbed.get("tags")
            failure_code = (
                tags.get("leadpoet.failure_code")
                if isinstance(tags, dict)
                else None
            )
            component = (
                tags.get("leadpoet.component")
                if isinstance(tags, dict)
                else None
            )
            stage = (
                tags.get("leadpoet.stage")
                if isinstance(tags, dict)
                else None
            )
            if failure_code and component and stage:
                # A semantic capture is commonly followed by logger.exception
                # for the same application error. Mark its message-safe
                # exception identity so that later generic integration event
                # is dropped while the structured event survives.
                semantic_exception_coalescer.admit(
                    _event_signature(scrubbed, include_semantic=False)
                )
                scrubbed["fingerprint"] = [
                    "leadpoet-semantic",
                    str(component),
                    str(stage),
                    str(failure_code),
                ]
            else:
                exception = scrubbed.get("exception")
                if isinstance(exception, dict) and not semantic_exception_coalescer.admit(
                    _event_signature(scrubbed, include_semantic=False)
                ):
                    return None
                # A scrubbed, message-safe fingerprint groups the same generic
                # failure across gateway and validator worker processes.
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


def _build_before_send_transaction(
    extra_prefixes: Tuple[str, ...], redact_all: bool
) -> Callable[[Any, Any], Optional[Dict[str, Any]]]:
    def _before_send_transaction(event: Any, hint: Any) -> Optional[Dict[str, Any]]:
        try:
            return sentry_scrubbing.scrub_transaction_event(
                event,
                extra_prefixes=extra_prefixes,
                redact_all=redact_all,
            )
        except BaseException as exc:
            _safe_print(
                "leadpoet_sentry_transaction_scrub_failed error=%s"
                % type(exc).__name__
            )
            return None

    return _before_send_transaction


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
                # Explicit options only; ambient SENTRY_* variables never apply.
                dsn=dsn,
                environment=environment,
                release=release,
                server_name="leadpoet-host",
                # Only explicit manual Leadpoet spans are sampled. Framework,
                # HTTP, DB, and chain auto-instrumentation remain disabled.
                traces_sample_rate=_trace_sample_rate(),
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
                max_breadcrumbs=_MAX_BREADCRUMBS,
                max_value_length=1024,
                debug=False,
                shutdown_timeout=1,
                # Explicit error-only integrations. Framework/DB/HTTP, argv,
                # module inventory, stdlib breadcrumbs, and Sentry Logs are off.
                default_integrations=False,
                auto_enabling_integrations=False,
                integrations=_core_integrations(),
                ignore_errors=[
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    GeneratorExit,
                ],
                before_send=_build_before_send(extra_prefixes, redact_all),
                before_breadcrumb=_build_before_breadcrumb(
                    extra_prefixes, redact_all
                ),
                before_send_transaction=_build_before_send_transaction(
                    extra_prefixes, redact_all
                ),
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


def add_sentry_breadcrumb(
    *,
    category: str,
    message: str,
    level: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Add one scrubbed operational breadcrumb. Never raises."""

    if not _state["active"]:
        return
    try:
        import sentry_sdk

        sentry_sdk.add_breadcrumb(
            category=scrub_sentry_value(str(category), 100),
            message=scrub_sentry_value(str(message), 200),
            level=str(level)[:20],
            data=dict(data or {}),
        )
    except BaseException:
        pass


def capture_sentry_failure(
    *,
    failure_code: str,
    component: str,
    stage: str,
    exception: Optional[BaseException],
    tags: Dict[str, Any],
    context: Dict[str, Any],
) -> bool:
    """Capture one semantic terminal event using a temporary SDK scope."""

    if not _state["active"]:
        return False
    try:
        import sentry_sdk

        scope_manager = getattr(sentry_sdk, "new_scope", None)
        if not callable(scope_manager):
            scope_manager = getattr(sentry_sdk, "push_scope")
        with scope_manager() as scope:
            scope.set_tag("leadpoet.component", scrub_sentry_value(component, 100))
            scope.set_tag(
                "leadpoet.failure_code", scrub_sentry_value(failure_code, 120)
            )
            scope.set_tag("leadpoet.stage", scrub_sentry_value(stage, 120))
            for key, value in tags.items():
                scope.set_tag(
                    "leadpoet.%s" % scrub_sentry_value(str(key), 100),
                    scrub_sentry_value(str(value), 200),
                )
            scope.set_context("leadpoet.operation", dict(context))
            if exception is not None:
                sentry_sdk.capture_exception(exception)
            else:
                sentry_sdk.capture_event(
                    {
                        "level": "error",
                        "message": "Leadpoet terminal failure: %s" % failure_code,
                    }
                )
        return True
    except BaseException:
        return False


def record_sentry_distribution(
    name: str,
    value: float,
    *,
    unit: str,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort metric emission across Sentry SDK minor API changes."""

    if not _state["active"]:
        return
    try:
        import sentry_sdk

        metrics = getattr(sentry_sdk, "metrics", None)
        distribution = getattr(metrics, "distribution", None)
        if callable(distribution):
            attributes = {
                str(key)[:100]: scrub_sentry_value(str(item), 100)
                for key, item in (tags or {}).items()
            }
            try:
                distribution(
                    str(name)[:120],
                    float(value),
                    unit=str(unit)[:40],
                    attributes=attributes,
                )
            except TypeError:
                # Compatibility with the older SDK metrics spelling.
                distribution(
                    str(name)[:120],
                    float(value),
                    unit=str(unit)[:40],
                    tags=attributes,
                )
    except BaseException:
        pass


@contextmanager
def start_sentry_span(
    *,
    operation: str,
    description: str,
    data: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
):
    """Start one manual span; SDK absence or failure leaves behavior unchanged."""

    if not _state["active"]:
        yield None
        return
    try:
        import sentry_sdk

        # Auto-instrumentation is deliberately disabled, so there is no
        # ambient transaction to own a standalone span. Each bounded stage is
        # therefore a sampled manual transaction with a static operation/name.
        span_kwargs = {
            "op": scrub_sentry_value(operation, 100),
            "name": scrub_sentry_value(description, 160),
        }
        if isinstance(trace_id, str) and re.fullmatch(r"[0-9a-f]{32}", trace_id):
            span_kwargs["trace_id"] = trace_id
        manager = sentry_sdk.start_transaction(
            **span_kwargs
        )
    except BaseException:
        yield None
        return
    try:
        span = manager.__enter__()
    except BaseException:
        yield None
        return
    try:
        set_data = getattr(span, "set_data", None)
        if callable(set_data):
            for key, value in dict(data or {}).items():
                set_data(str(key)[:100], value)
    except BaseException:
        # Metadata is optional; a collector/SDK failure cannot affect the
        # application stage or prevent the transaction from closing.
        pass
    try:
        yield span
    except BaseException as application_error:
        try:
            manager.__exit__(
                type(application_error), application_error, application_error.__traceback__
            )
        except BaseException:
            pass
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except BaseException:
            pass


def flush_sentry(timeout: float = 1.0) -> None:
    if not _state["active"]:
        return
    try:
        import sentry_sdk

        sentry_sdk.flush(timeout=max(0.0, min(float(timeout), 1.0)))
    except BaseException:
        pass


def _reset_for_tests() -> None:
    with _state_lock:
        _state["initialized"] = False
        _state["active"] = False
