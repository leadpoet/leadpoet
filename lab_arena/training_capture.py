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
import uuid
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


class CaptureError(ValueError):
    """Request evidence was not complete or did not pass integrity checks."""


@dataclass(frozen=True)
class PreparedRequest:
    """In-memory evidence; a reference is not proof that a write succeeded."""

    pointer: dict[str, Any]
    compressed: bytes
    objects: Any

    def persist(self) -> bool:
        try:
            self.objects.put(self.pointer["ref"], self.compressed)
            return True
        except Exception:
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


def readable_reasoning_content_path(path: tuple[str, ...]) -> bool:
    """Allow the plaintext chat alias only in a provider response message."""
    if (len(path) < 3 or path[-3] != "choices" or not path[-2].isdigit()
            or path[-1] != "message"):
        return False
    if len(path) == 3:
        return True  # A raw provider response, before it enters an example.
    return (len(path) >= 6 and path[0] == "provider_calls" and path[1].isdigit()
            and "response" in path[2:-3] and "request" not in path[2:-3])


def opaque_reasoning_field(name: str, child: Any, item_kind: str | None, safe_config: bool,
                           path: tuple[str, ...] = ()) -> bool:
    """Report an opaque reasoning field without dropping returned plain text."""
    if name == "reasoning_content":
        return not (isinstance(child, str) and readable_reasoning_content_path(path))
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
                            max_nodes: int = MAX_VALUE_NODES) -> tuple[Any, list[str]]:
    """Redact private material without truncation; report every redaction class."""
    reasons: set[str] = set()
    count = 0
    variants = set()
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= 4:
            for variant in (secret, quote(secret, safe=""), quote_plus(secret)):
                variants.update((variant, re.sub(r"%[0-9A-F]{2}", lambda m: m[0].lower(), variant)))

    def clean_text(text: str) -> str:
        # A known secret can span URL component or URL/text boundaries.
        for secret in sorted(variants, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")

        def redact(value):
            return trajectory.redact_text(value, tuple(variants), escape_nul=False)

        def clean_url(match):
            url = match[0]
            try:
                parts = urlsplit(url)
                netloc = parts.netloc
                if parts.username is not None or parts.password is not None:
                    netloc = netloc.rsplit("@", 1)[-1]
                netloc, path, fragment = map(redact, (netloc, parts.path, parts.fragment))
                pairs = []
                query_changed = False
                for key, child in parse_qsl(parts.query, keep_blank_values=True):
                    safe_key = redact(key)
                    normalized = safe_key.lower().replace("-", "_")
                    if normalized in _URL_SECRET_KEYS or trajectory._SENSITIVE_KEY_RE.fullmatch(normalized):
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
            return redact(text)
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
        return re.sub(re.escape(prefix) + r"([0-9]+)\]",
                      lambda match: urls[int(match[1])], cleaned)

    def request_config_path(path: tuple[str, ...]) -> bool:
        # The same request is sanitized first as a standalone blob and later
        # inside its exported example. Match the complete trusted container
        # path, so provider response objects cannot impersonate request config.
        if len(path) == 1 and path[0] in _PARAMETER_KEYS or path == ("outbound", "body"):
            return True
        if len(path) not in (4, 5) or path[0] != "provider_calls" or not path[1].isdigit() or path[2] != "request":
            return False
        return (len(path) == 4 and path[3] in _PARAMETER_KEYS
                or path[3:] == ("outbound", "body"))

    def clean(item: Any, path: tuple[str, ...] = ()) -> Any:
        nonlocal count
        count += 1
        if count > max_nodes or len(path) > 64:
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
                    and request_config_path(path)
                    and isinstance(child, Mapping)
                    and set(child) <= _CONFIG_KEYS
                    and all(isinstance(v, (str, bool, int)) or v is None for v in child.values())
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
                elif opaque_reasoning_field(normalized, child, item_kind, safe_config, path):
                    if empty_value(child):
                        result[key] = child
                    else:
                        result[key] = "[REDACTED]"
                        reasons.add("hidden_reasoning")
                elif normalized == "reasoning_details":
                    entries = []
                    for index, entry in enumerate(child):
                        if readable_reasoning_item(entry) or empty_value(entry):
                            entries.append(clean(entry, (*path, key, str(index))))
                        else:
                            entries.append("[REDACTED]")
                            reasons.add("hidden_reasoning")
                    result[key] = entries
                else:
                    result[key] = clean(child, (*path, key))
            return result
        if isinstance(item, (list, tuple)):
            return [clean(child, (*path, str(index))) for index, child in enumerate(item)]
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
