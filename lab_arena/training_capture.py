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
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urlsplit, urlunsplit

from lab_arena import trajectory


SCHEMA_VERSION = "leadpoet.lab_arena.training_request.v1"
OBJECT_PREFIX = "private/arena-training/requests/"
MAX_OBJECT_BYTES = 8 * 1024 * 1024
MAX_COMPRESSED_BYTES = MAX_OBJECT_BYTES + 64 * 1024
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HIDDEN_KEYS = frozenset({
    "reasoning_content", "hidden_reasoning", "encrypted_content", "reasoning_details",
})
_CONFIG_KEYS = frozenset({
    "effort", "summary", "context", "max_tokens", "exclude", "enabled",
})
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
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


def sanitize_training_value(value: Any, secrets: Sequence[str] = ()) -> tuple[Any, list[str]]:
    """Redact private material without truncation; report every redaction class."""
    reasons: set[str] = set()
    count = 0
    variants = set()
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= 4:
            for variant in (secret, quote(secret, safe=""), quote_plus(secret)):
                variants.update((variant, re.sub(r"%[0-9A-F]{2}", lambda m: m[0].lower(), variant)))

    def clean_text(text: str) -> str:
        def clean_url(match):
            url = match[0]
            try:
                parts = urlsplit(url)
                netloc = parts.netloc
                if parts.username is not None or parts.password is not None:
                    netloc = netloc.rsplit("@", 1)[-1]
                changed = netloc != parts.netloc
                pairs = []
                for key, child in parse_qsl(parts.query, keep_blank_values=True):
                    normalized = key.lower().replace("-", "_")
                    if normalized in _URL_SECRET_KEYS or trajectory._SENSITIVE_KEY_RE.fullmatch(normalized):
                        child = "[REDACTED]"
                        changed = True
                    pairs.append((key, child))
                if changed:
                    reasons.add("credential")
                    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(pairs), parts.fragment))
            except ValueError:
                # The credential patterns and exact-value scan still apply.
                pass
            return url

        text = _URL_RE.sub(clean_url, text)
        return trajectory.redact_text(text, tuple(variants))

    def clean(item: Any, path: tuple[str, ...] = ()) -> Any:
        nonlocal count
        count += 1
        if count > 250_000 or len(path) > 64:
            raise CaptureError("request_structure_limit")
        if isinstance(item, str):
            text = clean_text(item)
            if text != item:
                reasons.add("credential")
            return text
        if isinstance(item, Mapping):
            result = {}
            typed_reasoning = item.get("type") == "reasoning"
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
                    and path in {("admitted_parameters",), ("normalized_parameters",),
                                 ("effective_parameters",), ("hashed_parameters",),
                                 ("outbound", "body")}
                    and isinstance(child, Mapping)
                    and set(child) <= _CONFIG_KEYS
                    and all(isinstance(v, (str, bool, int)) or v is None for v in child.values())
                )
                if trajectory._SENSITIVE_KEY_RE.fullmatch(normalized) or normalized in _EXTRA_SECRET_KEYS:
                    result[key] = "[REDACTED]"
                    reasons.add("credential")
                elif normalized in _HIDDEN_KEYS or (normalized == "reasoning" and not safe_config) or (
                    typed_reasoning and normalized == "content"
                ):
                    result[key] = "[REDACTED]"
                    reasons.add("hidden_reasoning")
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
