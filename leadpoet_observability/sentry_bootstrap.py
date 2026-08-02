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
import os
from pathlib import Path
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

_state = {"initialized": False, "active": False}


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
        head = (repo_root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            ref_path = repo_root / ".git" / ref
            if ref_path.is_file():
                candidate = ref_path.read_text(encoding="utf-8").strip()
                return candidate if _looks_like_sha(candidate) else None
            packed = repo_root / ".git" / "packed-refs"
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


def _build_before_send(
    extra_prefixes: Tuple[str, ...], redact_all: bool
) -> Callable[[Any, Any], Optional[Dict[str, Any]]]:
    def _before_send(event: Any, hint: Any) -> Optional[Dict[str, Any]]:
        try:
            return sentry_scrubbing.scrub_event(
                event, extra_prefixes=extra_prefixes, redact_all=redact_all
            )
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
        environment = os.getenv(ENVIRONMENT_ENV, "").strip() or "production"
        release = os.getenv(RELEASE_ENV, "").strip() or _read_git_release(_REPO_ROOT)
        extra_prefixes = _extra_protected_prefixes()
        redact_all = _redact_all()

        sentry_sdk.init(
            # Explicit options only — ambient SENTRY_* variables never apply.
            dsn=dsn,
            environment=environment,
            release=release,
            # Errors only: no performance tracing, profiling, or sessions.
            traces_sample_rate=None,
            auto_session_tracking=False,
            # Payload capture is hard-off; the scrubber is the second fence.
            send_default_pii=False,
            include_local_variables=False,
            max_request_body_size="never",
            attach_stacktrace=True,
            max_breadcrumbs=50,
            max_value_length=1024,
            debug=False,
            shutdown_timeout=2,
            # Core integrations only (excepthook, threading, stdlib logging,
            # dedupe, atexit, modules). Framework/DB/HTTP auto-instrumentation
            # must never attach request or query payloads to events.
            auto_enabling_integrations=False,
            ignore_errors=[KeyboardInterrupt, asyncio.CancelledError, GeneratorExit],
            before_send=_build_before_send(extra_prefixes, redact_all),
            before_breadcrumb=_build_before_breadcrumb(extra_prefixes, redact_all),
        )
        sentry_sdk.set_tag("leadpoet.component", str(component))
        for key, value in (tags or {}).items():
            sentry_sdk.set_tag(
                "leadpoet.%s" % key,
                sentry_scrubbing.scrub_text(str(value), 200),
            )
        _state["active"] = True
        _safe_print(
            "leadpoet_sentry_initialized component=%s release=%s"
            % (component, release or "unknown")
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
    _state["initialized"] = False
    _state["active"] = False
