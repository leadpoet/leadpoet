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

_ALLOWED_EVENT_KEYS = frozenset(
    {
        "breadcrumbs",
        "contexts",
        "culprit",
        "dist",
        "environment",
        "event_id",
        "exception",
        "extra",
        "level",
        "logentry",
        "logger",
        "message",
        "platform",
        "release",
        "server_name",
        "stacktrace",
        "tags",
        "threads",
        "timestamp",
    }
)

_ALLOWED_TRANSACTION_KEYS = frozenset(
    {
        "contexts",
        "dist",
        "environment",
        "event_id",
        "measurements",
        "platform",
        "release",
        "server_name",
        "spans",
        "start_timestamp",
        "tags",
        "timestamp",
        "transaction",
        "transaction_info",
        "type",
    }
)

_ALLOWED_SPAN_KEYS = frozenset(
    {
        "data",
        "description",
        "exclusive_time",
        "op",
        "origin",
        "parent_span_id",
        "same_process_as_parent",
        "span_id",
        "start_timestamp",
        "status",
        "tags",
        "timestamp",
        "trace_id",
    }
)

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
    "Leadpoet.base.miner",
    "Leadpoet.base.validator",
    "Leadpoet.protocol",
    "Leadpoet.validator",
    "Leadpoet.base.utils.pool",
    "Leadpoet.base.utils.queue",
    "gateway.tee.autoresearch_executor_v2",
    "gateway.tee.model_sandbox_v2",
    "gateway.tee.provider_broker_v2",
    "gateway.tee.provider_client_v2",
    "gateway.tee.provider_evidence_v2",
    "gateway.tee.provider_semantics_v2",
    "gateway.tee.scoring_executor",
    "gateway.tee.scoring_executor_v2",
    "gateway.tee.source_add_runtime_v2",
    "gateway.tee.source_bundle_v2",
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
    "/Leadpoet/base/miner",
    "/Leadpoet/base/validator",
    "/Leadpoet/protocol",
    "/Leadpoet/validator/",
    "/langfuse/",
    "/openai/",
    "/anthropic/",
    "/firecrawl/",
    "/Leadpoet/base/utils/pool",
    "/Leadpoet/base/utils/queue",
    "/gateway/tee/autoresearch_executor_v2",
    "/gateway/tee/model_sandbox_v2",
    "/gateway/tee/provider_broker_v2",
    "/gateway/tee/provider_client_v2",
    "/gateway/tee/provider_evidence_v2",
    "/gateway/tee/provider_semantics_v2",
    "/gateway/tee/scoring_executor",
    "/gateway/tee/source_add_runtime_v2",
    "/gateway/tee/source_bundle_v2",
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
    "private key",
    "seed phrase",
    "mnemonic",
    "coldkey",
    "begin private key",
    "begin rsa private key",
    "begin openssh private key",
    "raw_attestation",
    "attestation_document",
    "receipt_graph",
    "credential_envelope",
    "raw_envelope",
    "complete_signature",
    "private_nonce",
    "plaintext_secret",
    "ciphertext_payload",
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
    r"|(?:(?:AKIA|ASIA)[0-9A-Z]{16})"
    r"|(?:sb_secret_[A-Za-z0-9_\-]{8,})"
    r"|(?:ghp_[A-Za-z0-9]{20,})"
    r"|(?:github_pat_[A-Za-z0-9_]{20,})"
    r"|(?:eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,})"
    r"|(?:(?i:bearer)\s+[A-Za-z0-9\-._~+/]{8,}=*)"
)

_URL_USERINFO_RE = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"
)

_SS58_RE = re.compile(r"\b5[1-9A-HJ-NP-Za-km-z]{47}\b")
_WALLET_FIELD_RE = re.compile(
    r"(?i)\b(wallet(?:[_\s-]?(?:name|hotkey))?|hotkey|coldkey|mnemonic"
    r"|seed[_\s-]?phrase|private[_\s-]?key)\s*[:=]\s*([^\s,;]+)"
)
_SAFE_MAPPING_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")

_URL_QUERY_RE = re.compile(r"\?[^\s'\"]+")

_SECRET_KEY_RE = re.compile(
    r"api[_-]?key|apikey|secret|token|credential|authorization|password"
    r"|passwd|cookie|session|dsn|private[_-]?key|service[_-]?role"
    r"|wallet|hotkey|coldkey|mnemonic|seed[_-]?phrase",
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
    r"|patch\b|messages\b|provider_response|response_text"
    r"|raw_.*(?:attestation|signature|nonce|envelope)"
    r"|attestation_(?:document|payload)|receipt_graph"
    r"|(?:complete|full)_signature|credential_envelope"
    r"|ciphertext|plaintext_secret|private_nonce",
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
    text = _URL_USERINFO_RE.sub(r"\1[leadpoet-redacted:credentials]@", text)
    text = _SS58_RE.sub("[leadpoet-redacted:wallet]", text)
    text = _WALLET_FIELD_RE.sub(
        lambda match: "%s=%s" % (match.group(1), "[leadpoet-redacted:wallet]"),
        text,
    )
    text = _EMAIL_RE.sub("[leadpoet-redacted:email]", text)
    if not _REF_LIKE_RE.match(text.strip()):
        text = _PHONE_RE.sub("[leadpoet-redacted:phone]", text)
    text = _URL_QUERY_RE.sub("?[leadpoet-redacted:query]", text)
    if len(text) > max_length:
        text = text[:max_length] + TRUNCATION_SUFFIX
    return text


def _scrub_mapping_key(raw_key: Any, index: int) -> str:
    """Keep ordinary field names while dropping content-shaped dynamic keys."""
    key = str(raw_key)
    if (
        not _SAFE_MAPPING_KEY_RE.fullmatch(key)
        or _SECRET_VALUE_RE.search(key)
        or _EMAIL_RE.search(key)
        or _PHONE_RE.search(key)
        or _SS58_RE.search(key)
        or _URL_USERINFO_RE.search(key)
        or _URL_QUERY_RE.search(key)
        or "sk-or-" in key.lower()
    ):
        return "[leadpoet-redacted:key:%d]" % index
    return key


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
            key = _scrub_mapping_key(raw_key, index)
            while key in out:
                key = "[leadpoet-redacted:key:%d:%d]" % (index, len(out))
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
        for key in ("module", "filename", "abs_path", "function", "package"):
            if isinstance(frame.get(key), str):
                frame[key] = scrub_text(frame[key], 300)


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
    if isinstance(category, str):
        crumb["category"] = scrub_text(category, 200)
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
    # Caller-controlled fingerprints can contain arbitrary payload strings.
    # The bootstrap installs a scrubbed operational fingerprint afterward.
    event.pop("fingerprint", None)
    for key in tuple(event):
        if key not in _ALLOWED_EVENT_KEYS:
            event.pop(key, None)
    extra = event.get("extra")
    if isinstance(extra, dict):
        extra.pop("sys.argv", None)

    for value in _iter_exception_values(event):
        _scrub_stacktrace(value.get("stacktrace"))
        for key in ("type", "module"):
            if isinstance(value.get(key), str):
                value[key] = scrub_text(value[key], 200)
        if isinstance(value.get("mechanism"), dict):
            value["mechanism"] = _scrub_object(value["mechanism"])
        if protected:
            if "value" in value:
                value["value"] = REDACTED_PROTECTED
        elif "value" in value:
            value["value"] = scrub_text(value.get("value"), MAX_MESSAGE_LENGTH)
    for thread in _iter_thread_values(event):
        _scrub_stacktrace(thread.get("stacktrace"))
        if isinstance(thread.get("name"), str):
            thread["name"] = scrub_text(thread["name"], 200)
    _scrub_stacktrace(event.get("stacktrace"))

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

    for key in (
        "culprit",
        "dist",
        "environment",
        "event_id",
        "level",
        "logger",
        "platform",
        "release",
        "server_name",
        "timestamp",
    ):
        if isinstance(event.get(key), str):
            event[key] = scrub_text(event[key], 200)

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

    # Error events never retain caller-controlled transaction names. Manual
    # operational transactions use the separate strict transaction allowlist.
    event.pop("transaction", None)

    return event


def scrub_transaction_event(
    event: Any,
    extra_prefixes: Iterable[str] = (),
    redact_all: bool = False,
) -> Optional[Dict[str, Any]]:
    """Scrub manually-created operational transactions and spans.

    Auto-instrumentation is disabled in the bootstrap. This allowlist exists
    solely for static Leadpoet operation/stage names and bounded metadata.
    """

    if not isinstance(event, dict):
        return None
    event.pop("request", None)
    event.pop("user", None)
    event.pop("breadcrumbs", None)
    for key in tuple(event):
        if key not in _ALLOWED_TRANSACTION_KEYS:
            event.pop(key, None)

    transaction = event.get("transaction")
    if isinstance(transaction, str):
        event["transaction"] = (
            REDACTED_PROTECTED
            if redact_all
            else scrub_text(transaction, 160)
        )
    transaction_info = event.get("transaction_info")
    if isinstance(transaction_info, dict):
        event["transaction_info"] = _scrub_object(transaction_info)

    spans = event.get("spans")
    if isinstance(spans, list):
        kept = []
        for raw_span in spans[:MAX_LIST_ITEMS]:
            if not isinstance(raw_span, dict):
                continue
            span = {
                key: value
                for key, value in raw_span.items()
                if key in _ALLOWED_SPAN_KEYS
            }
            for key in ("description", "op", "origin", "status"):
                if isinstance(span.get(key), str):
                    span[key] = (
                        REDACTED_PROTECTED
                        if redact_all and key == "description"
                        else scrub_text(span[key], 160)
                    )
            for envelope in ("data", "tags"):
                if isinstance(span.get(envelope), dict):
                    span[envelope] = _scrub_object(span[envelope])
            kept.append(span)
        event["spans"] = kept

    for envelope in ("contexts", "measurements", "tags"):
        if isinstance(event.get(envelope), dict):
            event[envelope] = _scrub_object(event[envelope])
    for key in (
        "dist",
        "environment",
        "event_id",
        "platform",
        "release",
        "server_name",
        "type",
    ):
        if isinstance(event.get(key), str):
            event[key] = scrub_text(event[key], 200)
    return event
