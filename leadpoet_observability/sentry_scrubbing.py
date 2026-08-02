"""Fail-closed scrubbing for Sentry error events.

Pure stdlib on purpose: importable (and testable) without sentry-sdk
installed, and auditable as the single place that decides what error
telemetry may contain. The rules mirror the repository's existing
protection boundaries (``research_lab/observability/redaction.py``,
``TELEMETRY.md``):

- Stack LOCALS never leave the process (``include_local_variables=False``
  at init; any residual ``vars`` are deleted here as a second fence).
- Source context lines are stripped from EVERY frame. An error raised
  inside LLM-generated candidate code, a private model artifact, or a
  prompt builder must never export source text; file/function/line survive
  for debugging.
- Events that touch a protected surface (Research Lab, trajectory/training
  capture, model internals, fulfillment/qualification lead content, LLM
  provider clients) keep exception TYPE, stack, and logger name — but their
  messages are replaced with a redaction token. Regex scrubbing cannot be
  trusted to recognize prompts, ICPs, trajectories, benchmarks, or lead
  payloads, so those surfaces are redacted wholesale.
- Every other string is regex-scrubbed (emails, formatted phone numbers,
  secret-shaped values, URL query strings) and length-capped. Join keys
  (UUIDs, sha256 refs) survive verbatim so events stay debuggable.
- The request envelope, user envelope, cookies, and ``sys.argv`` are
  dropped entirely.

Helpers here never raise on well-formed input, but the bootstrap wraps
every call anyway: a scrub failure DROPS the event (fail closed) rather
than letting anything unscrubbed leave the process.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple

REDACTED = "[leadpoet-redacted]"
REDACTED_PROTECTED = "[leadpoet-redacted:protected-surface]"
TRUNCATION_SUFFIX = "…[leadpoet-truncated]"

MAX_MESSAGE_LENGTH = 500
MAX_STRING_LENGTH = 500
MAX_BREADCRUMB_MESSAGE_LENGTH = 200
MAX_WALK_DEPTH = 8
MAX_DICT_ITEMS = 100
MAX_LIST_ITEMS = 50

# Module prefixes whose events keep type/stack/logger but lose message
# content entirely. A prefix matches itself and any dotted submodule.
# These are the surfaces where messages can embed trajectory/training IP or
# unredacted contact data: the Research Lab engine and capture pipeline,
# model internals, lead fulfillment/qualification content paths, and the
# LLM provider clients whose exceptions echo request/response fragments.
PROTECTED_MODULE_PREFIXES: Tuple[str, ...] = (
    "research_lab",
    "gateway.research_lab",
    "gateway.fulfillment",
    "gateway.qualification",
    "qualification",
    "leadpoet_verifier",
    "miner_models",
    "validator_models",
    "Leadpoet.base.utils.pool",
    "Leadpoet.base.utils.queue",
    "langfuse",
    "openai",
    "anthropic",
    "openrouter",
    "firecrawl",
)

# Path fragments that mark a stack frame as protected when the module name
# is unavailable (scripts running as __main__, site-packages clients).
PROTECTED_PATH_FRAGMENTS: Tuple[str, ...] = (
    "/research_lab/",
    "/fulfillment/",
    "/qualification/",
    "/leadpoet_verifier/",
    "/miner_models/",
    "/validator_models/",
    "/langfuse/",
    "/openai/",
    "/anthropic/",
    "/firecrawl/",
    "/Leadpoet/base/utils/pool",
    "/Leadpoet/base/utils/queue",
)

# Substrings that mark a whole string as unshippable (mirrors the marker
# vocabulary in research_lab/observability/redaction.py). Redaction here
# replaces the string instead of raising: an error event must still ship,
# just without the material.
_SECRET_MARKERS: Tuple[str, ...] = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
    "authorization: bearer",
    "judge_prompt",
    "hidden_benchmark",
    "hidden_icp",
    "icp_plaintext",
    "private_repo",
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)

# Deliberately stricter than a generic digit-run matcher: epoch ranges,
# block numbers, and byte counts ("6553000-6553360") must survive. Only
# strings with explicit phone formatting cues are treated as phone numbers.
_PHONE_RE = re.compile(
    r"(?:\+\d[\d\-\s().]{7,}\d)"
    r"|(?:\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4})"
    r"|(?:\b\d{3}[-.]\d{3}[-.]\d{4}\b)"
)

# UUIDs, sha256 digests, and prefixed refs are join keys and must survive
# scrubbing verbatim (same guard as redaction.py REF_LIKE_RE).
_REF_LIKE_RE = re.compile(
    r"^(?:[a-z0-9_.-]+:)*(?:sha256:[0-9a-f]{64}"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|[0-9a-f]{16,64})$",
    re.I,
)

_SECRET_VALUE_RE = re.compile(
    r"(?:sk-[A-Za-z0-9\-_]{16,})"
    r"|(?:AKIA[0-9A-Z]{16})"
    r"|(?:eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,})"
    r"|(?:(?i:bearer)\s+[A-Za-z0-9\-._~+/]{8,}=*)"
)

_URL_QUERY_RE = re.compile(r"\?[^\s'\"]+")

_SECRET_KEY_RE = re.compile(
    r"api[_-]?key|apikey|secret|token|credential|authorization|password"
    r"|passwd|cookie|session|dsn|private[_-]?key|service[_-]?role",
    re.I,
)

# Keys whose values are content, not operational metadata. Over-redaction
# in extra/contexts is acceptable; under-redaction is not — but join keys
# must survive ("leadpoet.component", "candidate_sha", "status_code",
# "content_length" stay; "lead_email", "page_content", "candidate_code",
# "request_body" are redacted), hence the word-boundary anchoring on the
# ambiguous stems.
_CONTENT_KEY_RE = re.compile(
    r"prompt|completion|\bllm\b|llm_|_llm|trajector|\bicp\b|icp_|_icp"
    r"|benchmark|sealed|page_content|\blead(s)?\b|lead_|_lead"
    r"|\bcontact(s)?\b|contact_|_contact|email|phone|linkedin|payload"
    r"|body\b|content\b|candidate_code|source_code|\bdiff\b|diff_|_diff"
    r"|patch\b|messages\b|provider_response|response_text",
    re.I,
)


def _is_protected_module(name: Any, extra_prefixes: Iterable[str] = ()) -> bool:
    if not isinstance(name, str) or not name:
        return False
    for prefix in tuple(PROTECTED_MODULE_PREFIXES) + tuple(extra_prefixes):
        if not prefix:
            continue
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


def _is_protected_filename(filename: Any) -> bool:
    if not isinstance(filename, str) or not filename:
        return False
    normalized = "/" + filename.replace("\\", "/").lstrip("/")
    return any(fragment in normalized for fragment in PROTECTED_PATH_FRAGMENTS)


def scrub_text(value: Any, max_length: int = MAX_STRING_LENGTH) -> Any:
    """Regex-scrub one string; non-strings pass through unchanged."""
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    for marker in _SECRET_MARKERS:
        if marker in lowered:
            return REDACTED
    text = _SECRET_VALUE_RE.sub(REDACTED, value)
    text = _EMAIL_RE.sub("[leadpoet-redacted:email]", text)
    if not _REF_LIKE_RE.match(text.strip()):
        text = _PHONE_RE.sub("[leadpoet-redacted:phone]", text)
    text = _URL_QUERY_RE.sub("?[leadpoet-redacted:query]", text)
    if len(text) > max_length:
        text = text[:max_length] + TRUNCATION_SUFFIX
    return text


def _scrub_object(value: Any, depth: int = 0) -> Any:
    """Key-aware recursive scrub for extra/contexts/tags/breadcrumb data."""
    if depth > MAX_WALK_DEPTH:
        return REDACTED
    if isinstance(value, str):
        return scrub_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= MAX_DICT_ITEMS:
                out[REDACTED] = "[leadpoet-truncated:dict]"
                break
            key = str(raw_key)
            if _SECRET_KEY_RE.search(key) or _CONTENT_KEY_RE.search(key):
                out[key] = REDACTED
            else:
                out[key] = _scrub_object(item, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        out_list = [_scrub_object(item, depth + 1) for item in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            out_list.append("[leadpoet-truncated:list]")
        return out_list
    # Arbitrary objects: their repr could contain anything. Never serialize.
    return REDACTED


def _scrub_stacktrace(stacktrace: Any) -> None:
    """Strip source context and residual locals from every frame, in place."""
    if not isinstance(stacktrace, dict):
        return
    frames = stacktrace.get("frames")
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame.pop("context_line", None)
        frame.pop("pre_context", None)
        frame.pop("post_context", None)
        frame.pop("vars", None)


def _stacktrace_is_protected(stacktrace: Any, extra_prefixes: Iterable[str]) -> bool:
    if not isinstance(stacktrace, dict):
        return False
    frames = stacktrace.get("frames")
    if not isinstance(frames, list):
        return False
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if _is_protected_module(frame.get("module"), extra_prefixes):
            return True
        if _is_protected_filename(frame.get("filename")) or _is_protected_filename(
            frame.get("abs_path")
        ):
            return True
    return False


def _iter_exception_values(event: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    exception = event.get("exception")
    if isinstance(exception, dict):
        values = exception.get("values")
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    yield value


def _iter_thread_values(event: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    threads = event.get("threads")
    if isinstance(threads, dict):
        values = threads.get("values")
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    yield value


def event_touches_protected_surface(
    event: Dict[str, Any], extra_prefixes: Iterable[str] = ()
) -> bool:
    """True when any part of the event originates on a protected surface.

    Sensitivity is event-level on purpose: a gateway wrapper exception whose
    message embeds ``str(inner_research_lab_error)`` must be redacted along
    with the inner value, so one protected link redacts the whole chain.
    """
    if _is_protected_module(event.get("logger"), extra_prefixes):
        return True
    for value in _iter_exception_values(event):
        if _is_protected_module(value.get("module"), extra_prefixes):
            return True
        if _stacktrace_is_protected(value.get("stacktrace"), extra_prefixes):
            return True
    for thread in _iter_thread_values(event):
        if _stacktrace_is_protected(thread.get("stacktrace"), extra_prefixes):
            return True
    return False


def scrub_breadcrumb(
    crumb: Any,
    extra_prefixes: Iterable[str] = (),
    redact_all: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return a shippable breadcrumb, or None to drop it entirely.

    Breadcrumbs from protected loggers are dropped rather than redacted: a
    redacted crumb carries no operational signal, and the buffer must never
    hold protected content. ``redact_all`` drops every breadcrumb.
    """
    if not isinstance(crumb, dict):
        return None
    if redact_all:
        return None
    category = crumb.get("category")
    if _is_protected_module(category, extra_prefixes):
        return None
    if "message" in crumb:
        crumb["message"] = scrub_text(
            crumb.get("message"), MAX_BREADCRUMB_MESSAGE_LENGTH
        )
    data = crumb.get("data")
    if isinstance(data, dict):
        data.pop("http.query", None)
        data.pop("http.fragment", None)
        crumb["data"] = _scrub_object(data)
    return crumb


def scrub_event(
    event: Any,
    extra_prefixes: Iterable[str] = (),
    redact_all: bool = False,
) -> Optional[Dict[str, Any]]:
    """Scrub one Sentry event in place and return it, or None to drop it."""
    if not isinstance(event, dict):
        return None
    protected = bool(redact_all) or event_touches_protected_surface(
        event, extra_prefixes
    )

    # Envelopes that are payload-shaped by construction are dropped outright.
    event.pop("request", None)
    event.pop("user", None)
    extra = event.get("extra")
    if isinstance(extra, dict):
        extra.pop("sys.argv", None)

    for value in _iter_exception_values(event):
        _scrub_stacktrace(value.get("stacktrace"))
        if protected:
            if "value" in value:
                value["value"] = REDACTED_PROTECTED
        elif "value" in value:
            value["value"] = scrub_text(value.get("value"), MAX_MESSAGE_LENGTH)
    for thread in _iter_thread_values(event):
        _scrub_stacktrace(thread.get("stacktrace"))

    logentry = event.get("logentry")
    if isinstance(logentry, dict):
        if protected:
            for key in ("message", "formatted"):
                if key in logentry:
                    logentry[key] = REDACTED_PROTECTED
            logentry.pop("params", None)
        else:
            for key in ("message", "formatted"):
                if key in logentry:
                    logentry[key] = scrub_text(logentry[key], MAX_MESSAGE_LENGTH)
            if "params" in logentry:
                logentry["params"] = _scrub_object(logentry.get("params"))
    if "message" in event:
        event["message"] = (
            REDACTED_PROTECTED
            if protected
            else scrub_text(event.get("message"), MAX_MESSAGE_LENGTH)
        )

    breadcrumbs = event.get("breadcrumbs")
    if isinstance(breadcrumbs, dict):
        values = breadcrumbs.get("values")
        if isinstance(values, list):
            kept = []
            for crumb in values:
                scrubbed = scrub_breadcrumb(
                    crumb, extra_prefixes=extra_prefixes, redact_all=redact_all
                )
                if scrubbed is not None:
                    kept.append(scrubbed)
            breadcrumbs["values"] = kept

    for envelope in ("extra", "contexts", "tags"):
        if isinstance(event.get(envelope), dict):
            event[envelope] = _scrub_object(event[envelope])

    if isinstance(event.get("transaction"), str):
        event["transaction"] = scrub_text(event["transaction"], 200)

    return event
