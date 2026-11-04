"""Private, bounded request evidence. Responses remain in the provider ledger.

The object describes a planned request; only its ledger receipt proves dispatch.
Redaction is explicit and makes the object unsuitable for exact replay.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urlsplit, urlunsplit

from lab_arena import trajectory


SCHEMA_VERSION = "leadpoet.lab_arena.training_request.v1"
OBJECT_PREFIX = "private/arena-training/requests/"
MAX_OBJECT_BYTES = 8 * 1024 * 1024
MAX_COMPRESSED_BYTES = MAX_OBJECT_BYTES + 64 * 1024
MAX_VALUE_NODES = 250_000
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# Keep readable text returned by a provider. Opaque state and signatures remain
# redacted, with the existing reason code for stored-object compatibility.
_OPAQUE_KEYS = frozenset({"hidden_reasoning", "encrypted_content"})
_READABLE_FIELDS = {
    "reasoning.summary": "summary", "reasoning.text": "text", "thinking": "thinking",
    "summary_text": "text", "reasoning_text": "text",
}
_OPAQUE_TYPES = frozenset({"reasoning.encrypted", "redacted_thinking"})
_REASONING_TYPES = frozenset({"reasoning", *_READABLE_FIELDS, *_OPAQUE_TYPES})
_REASONING_ITEM_FIELDS = frozenset({
    "type", "id", "status", "index", "format", "content", "summary", "text", "thinking",
})
_CONFIG_KEYS = frozenset({
    "effort", "summary", "context", "max_tokens", "exclude", "enabled",
})
_PARAMETER_KEYS = frozenset({
    "admitted_parameters", "normalized_parameters", "effective_parameters", "hashed_parameters",
})
# Escaped JSON quotes (and their adjacent separators) delimit embedded URLs.
# Including the comma/backslash in a query value lets URL normalization remove
# them, so the later credential scan consumes another JSON field on each pass.
# Check an escape run only at its start, so long backslash runs stay linear.
_URL_RE = re.compile(r"https?://(?:(?![,;]?(?<!\\)\\+[\"'])[^\s<>\"'])+", re.IGNORECASE)
_URL_SECRET_KEYS = frozenset({
    "key", "token", "sig", "signature", "auth", "credential", "credentials",
    "x_amz_credential", "x_amz_signature", "x_amz_security_token", "session_token",
})
_EXTRA_SECRET_KEYS = frozenset({
    "credential", "credentials", "private_key", "secret_key", "session_token",
    "x_amz_credential", "x_amz_signature", "x_amz_security_token",
})
# The 2026-10-28 privacy audit recovered this exact public source by restoring
# two URL fields and matching the original execution document hash. No other
# URL is exempt from the bare-key matcher without equivalent evidence.
_VERIFIED_PUBLIC_SOURCE_URL = (
    "https://www.prnewswire.com/news-releases/"
    "biohaven-and-sk-biopharmaceuticals-enter-into-strategic-global-licensing-"
    "agreement-for-novel-kv7-ion-channel-platform-and-opakalim-lead-candidate-"
    "for-epilepsy-302860299.html"
)


class CaptureError(ValueError):
    """Request evidence was not complete or did not pass integrity checks."""


def retryable_storage_error(error: Exception) -> bool:
    """Recognize transport availability failures, never missing/denied objects."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    try:
        from botocore.exceptions import (
            ConnectTimeoutError, ConnectionClosedError, EndpointConnectionError,
            ReadTimeoutError, ResponseStreamingError,
        )
    except ImportError:
        # Local embedded stores do not require the optional S3 client SDK.
        transport_errors = ()
    else:
        transport_errors = (ConnectTimeoutError, ConnectionClosedError,
                            EndpointConnectionError, ReadTimeoutError, ResponseStreamingError)
    if isinstance(error, transport_errors):
        return True
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return False
    detail = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = detail.get("Code") if isinstance(detail, Mapping) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    if (type(status) is int and status in {401, 403, 404}) or (
            isinstance(code, str) and code in {"NoSuchKey", "NotFound", "AccessDenied"}):
        return False
    return ((isinstance(code, str) and code in {
        "RequestTimeout", "RequestTimeoutException", "SlowDown", "InternalError", "ServiceUnavailable",
    }) or (type(status) is int and status in {429, 500, 502, 503, 504}))


@dataclass(frozen=True)
class PreparedRequest:
    """In-memory evidence; a reference is not proof that a write succeeded."""

    pointer: dict[str, Any]
    compressed: bytes
    objects: Any

    def persist(self) -> bool:
        ref = self.pointer.get("ref")
        if not isinstance(ref, str) or not ref:
            return False
        for attempt in range(2):
            try:
                self.objects.put(ref, self.compressed)
                return True
            except Exception as exc:
                if attempt == 0 and retryable_storage_error(exc):
                    # The immutable ref and prepared bytes are identical even
                    # if the first write committed before its response timed out.
                    time.sleep(0.05)
                    continue
                return False
        return False


def unavailable(reason: str = "capture_failed") -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": reason}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def empty_value(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and len(value) == 0)


def _readable_text_payload(value: Any, part_type: str | None = None) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return (isinstance(value, list) and bool(value)
            and all(isinstance(part, Mapping) and part.get("type") == part_type
                    and isinstance(part.get("text"), str) and bool(part["text"].strip())
                    for part in value))


def reasoning_kind(item: Mapping[str, Any]) -> str | None:
    """Classify a typed provider reasoning item by its readable payload."""
    kind = item.get("type")
    if not isinstance(kind, str) or kind not in _REASONING_TYPES:
        return None
    if kind in _OPAQUE_TYPES:
        return "opaque"
    if kind == "reasoning":
        return "readable" if any((
            _readable_text_payload(item.get("summary"), "summary_text"),
            _readable_text_payload(item.get("content"), "reasoning_text"),
            _readable_text_payload(item.get("text")),
        )) else "opaque"
    field_name = _READABLE_FIELDS.get(kind)
    if field_name is not None:
        if not _readable_text_payload(item.get(field_name)):
            return "opaque"
    return "readable"


def readable_reasoning_item(item: Any) -> bool:
    return isinstance(item, Mapping) and reasoning_kind(item) == "readable"


def _provider_call_path(path: tuple[str, ...], root: Any = None
                        ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Split only the two canonical export call roots from their relative path."""
    if isinstance(root, Mapping) and (isinstance(root.get("choices"), list)
            or root.get("object") in ("response", "chat.completion", "chat.completion.chunk")):
        return None  # Recognizable raw provider replies cannot establish export roots.
    for collection in (("provider_calls",), ("judgment", "signal_source_provider_calls")):
        index = len(collection)
        if len(path) > index and path[:index] == collection and path[index].isdigit():
            if root is not None:
                calls = root
                for key in collection:
                    calls = calls.get(key) if isinstance(calls, Mapping) else None
                if not isinstance(calls, list) or int(path[index]) >= len(calls):
                    return None
            return path[:index + 1], path[index + 1:]
    return None


def _rejected_model_config_path(path: tuple[str, ...], root: Any) -> bool:
    """Bind rejected frame/HTTP controls to canonical host request containers.

    Parse errors have no admitted operation identity. The same decoded control
    values must remain a fixed point in saved and exported evidence.
    """
    http = path[-2:] == ("admitted_parameters", "body_text")
    if not http and path[-3:] != ("admitted_parameters", "raw_frame", "parameters"):
        return False
    if not isinstance(root, Mapping) or (isinstance(root.get("choices"), list)
            or root.get("object") in ("response", "chat.completion", "chat.completion.chunk")):
        return False

    def at(parts):
        item = root
        for part in parts:
            if isinstance(item, Mapping):
                item = item.get(part)
            elif isinstance(item, list) and part.isdigit() and int(part) < len(item):
                item = item[int(part)]
            else:
                return None
        return item

    request_path = path[:-2] if http else path[:-3]
    if request_path:
        # No suffix matching: provider bodies and arbitrary wrappers cannot
        # establish a host observation by copying canonical field names.
        if request_path == ("request",):
            observation = root
        elif (len(request_path) == 3 and request_path[0] == "observations"
              and request_path[1].isdigit() and request_path[2] == "request"):
            observation = at(request_path[:-1])
        elif (len(request_path) == 4 and request_path[:2] == ("payload", "observations")
              and request_path[2].isdigit() and request_path[3] == "request"
              and root.get("schema_version") in (
                  "leadpoet.lab_arena.training_observations.v1",
                  "leadpoet.lab_arena.training_observations.v2")):
            observation = at(request_path[:-1])
        elif (len(request_path) == 6 and request_path[0] in {
                "worker_observation_artifacts", "partial_worker_observation_artifacts",
                "unverified_worker_observation_artifacts"}
              and request_path[2:4] == ("payload", "observations")
              and request_path[4].isdigit() and request_path[5] == "request"):
            observation = at(request_path[:-1])
        else:
            return False
        if (request_path != ("request",)
                and not isinstance(at(request_path[:-2]), list)):
            return False
        if (len(request_path) == 6
                and not isinstance(root.get(request_path[0]), Mapping)):
            return False
        if (not isinstance(observation, Mapping)
                or set(observation) != {"ordinal", "action_sequence", "operation_id", "kind",
                                       "request", "outcome", "call_identity", "privacy",
                                       "contributes_to_provider_call_count"}
                or observation.get("kind") != "local_parse_error"
                or observation.get("operation_id") != "unknown"
                or observation.get("action_sequence") is not None
                or observation.get("call_identity") is not None
                or observation.get("contributes_to_provider_call_count") is not False
                or not isinstance(observation.get("outcome"), Mapping)
                or observation["outcome"].get("protocol") != ("http" if http else "frame")):
            return False
    request = at(request_path)
    if (not isinstance(request, Mapping) or set(request) != {"admitted_parameters", "timeout_ms"}
            or type(request.get("timeout_ms")) is not int or request["timeout_ms"] != 0
            or not isinstance(request.get("admitted_parameters"), Mapping)):
        return False
    admitted = request["admitted_parameters"]
    if http:
        body = admitted.get("body_text")
        return (set(admitted) == {"method", "url", "headers", "body_text"}
                and admitted.get("method") == "POST"
                and admitted.get("url") in (
                    "https://openrouter.ai/api/v1/chat/completions",
                    "https://openrouter.ai/api/v1/responses")
                and isinstance(admitted.get("headers"), Mapping)
                and isinstance(body, Mapping)
                and isinstance(body.get("model"), str) and bool(body["model"].strip())
                and not isinstance(body.get("choices"), list)
                and body.get("object") not in ("response", "chat.completion", "chat.completion.chunk"))
    if set(admitted) != {"raw_frame"}:
        return False
    frame = admitted["raw_frame"]
    return (isinstance(frame, Mapping)
            and {"operation_id", "parameters"} <= set(frame)
            and set(frame) <= {"schema_version", "operation_id", "parameters", "timeout_ms"}
            and isinstance(frame["operation_id"], str)
            and frame["operation_id"] in {"openrouter.chat", "openrouter.responses"}
            and isinstance(frame["parameters"], Mapping)
            and isinstance(frame["parameters"].get("model"), str)
            and bool(frame["parameters"]["model"].strip()))


def _rejected_model_controls(value: Mapping[str, Any]) -> bool:
    """Allow only control values, not arbitrary strings in control-shaped data."""
    choices = {
        "effort": {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        "summary": {"auto", "concise", "detailed"},
        "context": {"auto", "all_turns", "current_turn"},
    }
    for key, child in value.items():
        if child == "[REDACTED]":
            continue  # Keep credential-filtered controls a fixed point.
        if key in choices:
            valid = isinstance(child, str) and child in choices[key]
        elif key in {"exclude", "enabled"}:
            valid = type(child) is bool
        else:
            valid = key == "max_tokens" and type(child) is int and child > 0
        if not valid:
            return False
    return True


def readable_reasoning_content_path(path: tuple[str, ...], root: Any = None) -> bool:
    """Allow the plaintext chat alias only in a provider response message."""
    if (len(path) < 3 or path[-3] != "choices" or not path[-2].isdigit()
            or path[-1] != "message"):
        return False
    if len(path) == 3:
        return True  # A raw provider response, before it enters an example.
    call_path = _provider_call_path(path[:-3], root)
    if call_path is None:
        return False
    relative = call_path[1]
    return (relative == ("response",)
            or len(relative) == 3 and relative[0] in {"response_receipts", "upstream_response_receipts"}
            and relative[1].isdigit() and relative[2] == "response")


def approved_history_reasoning_content(root: Any, path: tuple[str, ...], item: Any) -> bool:
    """Keep plain assistant replay text only in a known Chat request message."""
    if not isinstance(item, Mapping) or item.get("role") != "assistant" or not isinstance(root, Mapping):
        return False
    if (isinstance(root.get("choices"), list)
            or root.get("object") in ("response", "chat.completion", "chat.completion.chunk")):
        return False
    if len(path) >= 3 and path[-2] == "messages" and path[-1].isdigit():
        prefix = path[:-2]
    else:
        return False
    def projection(parts: tuple[str, ...]) -> bool:
        return parts == ("outbound", "body") or len(parts) == 1 and parts[0] in _PARAMETER_KEYS

    if not projection(prefix):
        call_path = _provider_call_path(prefix, root)
        if call_path is None:
            return False
        location, relative = call_path
        if not relative or relative[0] != "request" or not projection(relative[1:]):
            return False
        calls = root
        for key in location[:-1]:
            calls = calls.get(key) if isinstance(calls, Mapping) else None
        if not isinstance(calls, list) or int(location[-1]) >= len(calls):
            return False
        call = calls[int(location[-1])]
        operation = call.get("operation_id") if isinstance(call, Mapping) else None
        request = call.get("request") if isinstance(call, Mapping) else None
        if not isinstance(request, Mapping) or request.get("operation_id") != operation:
            return False
    else:
        operation = root.get("operation_id")
    return operation == "openrouter.chat"


def opaque_reasoning_field(name: str, child: Any, item_kind: str | None, safe_config: bool,
                           path: tuple[str, ...] = (), root: Any = None) -> bool:
    """Report an opaque reasoning field without dropping returned plain text."""
    if name == "reasoning_content":
        return not (isinstance(child, str) and readable_reasoning_content_path(path, root))
    if name in _OPAQUE_KEYS:
        return True
    if item_kind is not None and name not in _REASONING_ITEM_FIELDS:
        return True
    if item_kind == "opaque" and name not in {"type", "id", "status", "index", "format"}:
        return True
    if item_kind == "readable" and name == "summary" and not _readable_text_payload(child, "summary_text"):
        return True
    if item_kind == "readable" and name == "content" and not _readable_text_payload(child, "reasoning_text"):
        return True
    if item_kind == "readable" and name in {"text", "thinking"} and not _readable_text_payload(child):
        return True
    if name == "reasoning_details":
        return not isinstance(child, list)
    if name == "reasoning":
        return not (child is None or isinstance(child, str) or safe_config)
    return False


def sanitize_training_value(value: Any, secrets: Sequence[str] = (), *,
                            max_nodes: int = MAX_VALUE_NODES,
                            aggregate_parts: bool = False) -> tuple[Any, list[str]]:
    """Redact private material without truncation; report every redaction class.

    Final joined examples may budget each canonical call and input snapshot
    separately. All other context shares the envelope limit; paths stay intact.
    """
    reasons: set[str] = set()
    envelope_budget = [0]
    variants = set()
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= 4:
            for variant in (secret, quote(secret, safe=""), quote_plus(secret)):
                variants.update((variant, re.sub(r"%[0-9A-F]{2}", lambda m: m[0].lower(), variant)))

    # Final examples repeat large request and catalog strings. Keep only bounded
    # long-text results within this walk; every node, path, and key is still checked.
    text_cache: OrderedDict[str, str] = OrderedDict()
    # Standalone captures and aggregates repeat short schema strings and keys;
    # aggregates also repeat complete long responses. These modes share a bounded
    # plain-string lane without evicting the medium/custom lane.
    # Secret subclasses may carry behavior, so they disable this lane.
    large_text_cache: OrderedDict[str, tuple[str, int]] = OrderedDict()
    large_text_cache_enabled = all(type(variant) is str for variant in variants)
    large_text_bytes = 0
    large_text_byte_limit = 8 * 1024 * 1024

    def clean_text(text: str) -> str:
        nonlocal large_text_bytes
        if (large_text_cache_enabled and type(text) is str
                and (len(text) < 1_024 or (aggregate_parts and len(text) > 16_384))):
            cached = large_text_cache.get(text)
            if cached is not None:
                large_text_cache.move_to_end(text)
                return cached[0]
            cleaned = scan_text(text)
            if type(cleaned) is str:
                # Count both retained strings, even if they are the same object.
                # Built-in Unicode storage includes wide characters and avoids
                # encoding allocations or custom __sizeof__/encode methods.
                charge = str.__sizeof__(text) + str.__sizeof__(cleaned)
                if charge <= large_text_byte_limit:
                    while large_text_cache and (len(large_text_cache) >= 512
                                                or large_text_bytes + charge > large_text_byte_limit):
                        _, (_, removed_charge) = large_text_cache.popitem(last=False)
                        large_text_bytes -= removed_charge
                    large_text_cache[text] = (cleaned, charge)
                    large_text_bytes += charge
            return cleaned
        if 1_024 <= len(text) <= 16_384:
            cached = text_cache.get(text)
            if cached is not None:
                text_cache.move_to_end(text)
                return cached
        cleaned = scan_text(text)
        if 1_024 <= len(text) <= 16_384 and len(cleaned) <= 16_384:
            text_cache[text] = cleaned
            if len(text_cache) > 512:
                text_cache.popitem(last=False)
        return cleaned

    def scan_text(text: str) -> str:
        # A known secret can span URL component or URL/text boundaries.
        for secret in sorted(variants, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")

        # Protect the audited URL's false-positive "sk-" path span from the
        # generic bare-key matcher only. Exact secrets above and every other
        # credential context still use ordinary checks.
        slug_spans = []
        for url_match in _URL_RE.finditer(text) if "sk-" in text else ():
            try:
                parts = urlsplit(url_match[0])
            except ValueError:
                continue
            if urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) != _VERIFIED_PUBLIC_SOURCE_URL:
                continue
            path_offset = url_match[0].find(parts.path, url_match[0].find("://") + 3)
            if path_offset < 0:
                continue
            for candidate in trajectory._BARE_API_KEY_RE.finditer(parts.path):
                start = url_match.start() + path_offset + candidate.start()
                slug_spans.append((start, start + len(candidate[0])))
        slug_prefix = "arena-public-slug-" + uuid.uuid4().hex + "-" if slug_spans else ""
        while slug_prefix and slug_prefix in text:
            slug_prefix = "arena-public-slug-" + uuid.uuid4().hex + "-"
        slugs = []
        for start, end in reversed(slug_spans):
            slugs.append(text[start:end])
            text = text[:start] + slug_prefix + str(len(slugs) - 1) + "-end" + text[end:]

        def restore_slugs(value: str) -> str:
            if not slugs:
                return value
            return re.sub(re.escape(slug_prefix) + r"([0-9]+)-end",
                          lambda match: slugs[int(match[1])], value)

        def redact(value):
            return trajectory.redact_text(value, tuple(variants), escape_nul=False)

        def clean_url(match):
            url = match[0]
            try:
                parts = urlsplit(url)
                netloc = parts.netloc
                if parts.username is not None or parts.password is not None:
                    netloc = netloc.rsplit("@", 1)[-1]
                netloc, fragment = map(redact, (netloc, parts.fragment))
                # Exact secrets in this un-decoded path were checked before
                # masking. Do not mistake an internal marker for input data;
                # decoded query values below still need the exact-secret scan.
                path = (trajectory.redact_text(parts.path, escape_nul=False)
                        if slug_prefix and slug_prefix in parts.path
                        else redact(parts.path))
                pairs = []
                query_changed = False
                for key, child in parse_qsl(parts.query, keep_blank_values=True):
                    safe_key = redact(key)
                    normalized = safe_key.lower().replace("-", "_")
                    # A blank query value contains no credential bytes. Keep
                    # its exact spelling, as with empty structured fields.
                    if child and (normalized in _URL_SECRET_KEYS or trajectory._SENSITIVE_KEY_RE.fullmatch(normalized)):
                        safe_child = "[REDACTED]"
                    else:
                        safe_child = redact(child)
                    query_changed |= safe_key != key or safe_child != child
                    pairs.append((safe_key, safe_child))
                changed = query_changed or (netloc, path, fragment) != (parts.netloc, parts.path, parts.fragment)
                if changed:
                    reasons.add("credential")
                    return urlunsplit((parts.scheme, netloc, path,
                                       urlencode(pairs) if query_changed else parts.query, fragment))
            except ValueError:
                # The credential patterns and exact-value scan still apply.
                return redact(url)
            return url

        # Clean URL secrets before a credential match can remove its scheme.
        # These existing token matchers must see text across URL boundaries.
        text = _URL_RE.sub(clean_url, text)
        text = trajectory.redact_auth_schemes(text)
        text = trajectory._BARE_API_KEY_RE.sub("[REDACTED]", text)
        if not _URL_RE.search(text):
            # A credential prefix can remove the URL scheme after its path
            # was masked. The same pre-mask exact-secret check still applies.
            cleaned = (trajectory.redact_text(text, escape_nul=False)
                       if slug_prefix and slug_prefix in text else redact(text))
            return restore_slugs(cleaned)
        # Keep URL delimiters out of the generic credential scan, while a URL
        # used as a credential value still gets removed with its full context.
        # '[' cannot turn Basic/Bearer plus a public URL into a credential.
        prefix = "[arena-url-" + uuid.uuid4().hex + "-"
        while prefix in text:
            prefix = "[arena-url-" + uuid.uuid4().hex + "-"
        urls = []

        def mask_url(match):
            # Removing a credential prefix can expose a nested URL span.
            urls.append(clean_url(match))
            return prefix + str(len(urls) - 1) + "]"

        masked = _URL_RE.sub(mask_url, text)
        # JSON escapes NUL losslessly. Diagnostic stream escaping would change
        # private evidence and falsely report a credential redaction.
        # Exact secrets were removed before masking; never match them against
        # temporary tokens. Restore all surviving URLs in one scan.
        cleaned = trajectory.redact_text(masked, escape_nul=False)
        return restore_slugs(re.sub(re.escape(prefix) + r"([0-9]+)\]",
                                    lambda match: urls[int(match[1])], cleaned))

    def request_config_path(path: tuple[str, ...]) -> bool:
        # The same request is sanitized first as a standalone blob and later
        # inside its exported example. Match the complete trusted container
        # path, so provider response objects cannot impersonate request config.
        if len(path) == 1 and path[0] in _PARAMETER_KEYS or path == ("outbound", "body"):
            return True
        call_path = _provider_call_path(path, value)
        if call_path is None:
            return False
        relative = call_path[1]
        return (len(relative) == 2 and relative[0] == "request" and relative[1] in _PARAMETER_KEYS
                or relative == ("request", "outbound", "body"))

    def clean(item: Any, path: tuple[str, ...] = (), budget=None) -> Any:
        if budget is None:
            budget = envelope_budget
        call_path = _provider_call_path(path, value) if aggregate_parts else None
        if (aggregate_parts and isinstance(value, Mapping)
                and (call_path is not None and not call_path[1]
                     or len(path) == 2 and path[0] == "input_snapshots"
                     and isinstance(value.get(path[0]), Mapping))):
            # Joined evidence has independently bounded calls and snapshots.
            # Keep their original paths and scan every leaf. Their roots still
            # count against the envelope, whose other context shares one cap.
            envelope_budget[0] += 1
            if envelope_budget[0] > max_nodes:
                raise CaptureError("request_structure_limit")
            budget = [0]
        budget[0] += 1
        if budget[0] > max_nodes or len(path) > 64:
            raise CaptureError("request_structure_limit")
        if isinstance(item, str):
            text = clean_text(item)
            if text != item:
                reasons.add("credential")
            return text
        if isinstance(item, Mapping):
            result = {}
            item_kind = reasoning_kind(item)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise CaptureError("request_key_invalid")
                safe_key = clean_text(key)
                if safe_key != key:
                    reasons.add("credential")
                    key = safe_key
                if key in result:
                    raise CaptureError("request_redaction_key_collision")
                normalized = key.lower().replace("-", "_")
                safe_config = (
                    normalized == "reasoning"
                    and isinstance(child, Mapping)
                    and set(child) <= _CONFIG_KEYS
                    and all(isinstance(v, (str, bool, int)) or v is None for v in child.values())
                    and (request_config_path(path)
                         or item.get("object") == "response" and isinstance(item.get("output"), list)
                         or _rejected_model_config_path(path, value) and _rejected_model_controls(child))
                )
                # False and zero retain the existing sensitive-value handling.
                empty_credential = child is None or (
                    isinstance(child, (str, dict, list)) and len(child) == 0
                )
                if (
                    trajectory._SENSITIVE_KEY_RE.fullmatch(normalized)
                    or normalized in _EXTRA_SECRET_KEYS
                ) and not empty_credential:
                    result[key] = "[REDACTED]"
                    reasons.add("credential")
                elif (normalized == "reasoning_content" and isinstance(child, str)
                      and approved_history_reasoning_content(value, path, item)):
                    result[key] = clean(child, (*path, key), budget)
                elif opaque_reasoning_field(normalized, child, item_kind, safe_config, path, value):
                    if empty_value(child):
                        result[key] = child
                    else:
                        result[key] = "[REDACTED]"
                        reasons.add("hidden_reasoning")
                elif normalized == "reasoning_details":
                    entries = []
                    for index, entry in enumerate(child):
                        if readable_reasoning_item(entry) or empty_value(entry):
                            entries.append(clean(entry, (*path, key, str(index)), budget))
                        else:
                            entries.append("[REDACTED]")
                            reasons.add("hidden_reasoning")
                    result[key] = entries
                else:
                    result[key] = clean(child, (*path, key), budget)
            return result
        if isinstance(item, (list, tuple)):
            return [clean(child, (*path, str(index)), budget) for index, child in enumerate(item)]
        if item is None or isinstance(item, (bool, int, float)):
            return item
        raise CaptureError("request_value_invalid")

    return clean(value), sorted(reasons)


class RequestCapture:
    """Write a deterministic object using the existing private object store."""

    def __init__(self, objects: Any) -> None:
        self.objects = objects

    def __call__(self, document: Mapping[str, Any], *, secrets: Sequence[str] = ()) -> dict[str, Any]:
        prepared = self.prepare(document, secrets=secrets)
        if not isinstance(prepared, PreparedRequest):
            return prepared
        if not prepared.persist():
            return unavailable()
        return dict(prepared.pointer, status="stored")

    def prepare(self, document: Mapping[str, Any], *, secrets: Sequence[str] = ()) -> PreparedRequest | dict[str, Any]:
        """Prepare bytes with no I/O; the broker persists only after admission."""
        try:
            # Bound serialization before making redacted copies or compressed bytes.
            if len(_canonical(document)) > MAX_OBJECT_BYTES:
                return unavailable("request_size_limit")
            cleaned, redactions = sanitize_training_value(document, secrets)
            cleaned["schema_version"] = SCHEMA_VERSION
            cleaned["privacy"] = {"redacted": bool(redactions), "redactions": redactions}
            raw = _canonical(cleaned)
            if len(raw) > MAX_OBJECT_BYTES:
                return unavailable("request_size_limit")
            digest = hashlib.sha256(raw).hexdigest()
            ref = OBJECT_PREFIX + digest + ".json.gz"
            compressed = gzip.compress(raw, compresslevel=6, mtime=0)
            if len(compressed) > MAX_COMPRESSED_BYTES:
                return unavailable("request_size_limit")
            pointer = {
                "schema_version": SCHEMA_VERSION, "status": "referenced", "ref": ref,
                "sha256": "sha256:" + digest, "encoding": "gzip",
                "size_bytes": len(raw), "compressed_bytes": len(compressed),
                "redacted": bool(redactions), "redactions": redactions,
            }
            return PreparedRequest(pointer, compressed, self.objects)
        except Exception:
            # No raw exception, request, endpoint, or secret is emitted.
            return unavailable()


def read_request(objects: Any, pointer: Mapping[str, Any]) -> dict[str, Any]:
    """Verify and inflate one receipt pointer; missing or altered data fails closed."""
    if (
        not isinstance(pointer, Mapping) or pointer.get("schema_version") != SCHEMA_VERSION
        or pointer.get("status") not in ("stored", "referenced") or pointer.get("encoding") != "gzip"
        or not isinstance(pointer.get("sha256"), str)
        or not _HASH_RE.fullmatch(pointer["sha256"])
        or pointer.get("ref") != OBJECT_PREFIX + pointer["sha256"][7:] + ".json.gz"
        or type(pointer.get("size_bytes")) is not int
        or not 0 < pointer["size_bytes"] <= MAX_OBJECT_BYTES
        or type(pointer.get("compressed_bytes")) is not int
        or not 0 < pointer["compressed_bytes"] <= MAX_COMPRESSED_BYTES
    ):
        raise CaptureError("request_pointer_invalid")
    try:
        compressed = objects.get_bounded(pointer["ref"], MAX_COMPRESSED_BYTES)
        if len(compressed) != pointer["compressed_bytes"]:
            raise CaptureError("request_compressed_size_mismatch")
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(MAX_OBJECT_BYTES + 1)
        if len(raw) != pointer["size_bytes"] or len(raw) > MAX_OBJECT_BYTES:
            raise CaptureError("request_size_mismatch")
        if "sha256:" + hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
            raise CaptureError("request_hash_mismatch")
        document = json.loads(raw)
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
            raise CaptureError("request_schema_invalid")
        if document.get("privacy") != {
            "redacted": pointer.get("redacted"), "redactions": pointer.get("redactions"),
        } or type(pointer.get("redacted")) is not bool or not isinstance(pointer.get("redactions"), list) or (
            pointer["redacted"] != bool(pointer["redactions"])
            or any(reason not in {"credential", "hidden_reasoning"} for reason in pointer["redactions"])
        ):
            raise CaptureError("request_privacy_mismatch")
        return document
    except CaptureError:
        raise
    except Exception:
        raise CaptureError("request_unavailable") from None
