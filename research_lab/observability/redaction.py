"""Fail-closed redaction boundary for Langfuse and Engine payloads."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

SECRET_MARKERS = (
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

BLOCKED_KEYS = {
    "prompt",
    "llm_response",
    "page_content",
    "sealed_items",
    "raw_icp",
    "private_prompt",
    "judge_prompt",
    "hidden_benchmark",
    "hidden_icp",
    "icp_plaintext",
}

HASH_VALUE_KEYS = {
    "email",
    "phone",
    "linkedin",
    "authorization",
    "api_key",
    "token",
    "credential",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
}

SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|raw[_-]?secret|raw[_-]?openrouter|token|credential|authorization)", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")


class RedactionBlocked(ValueError):
    """Raised when a payload contains material that must not leave Leadpoet."""


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _blocked_marker(text: str) -> str | None:
    lowered = text.lower()
    for marker in SECRET_MARKERS:
        if marker in lowered:
            return marker
    return None


def _safe_scalar(value: Any, *, key: str, mode: str) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    marker = _blocked_marker(text)
    if marker:
        raise RedactionBlocked(f"blocked secret/protected marker: {marker}")
    if key.lower() in HASH_VALUE_KEYS or SECRET_KEY_RE.search(key):
        return f"[REDACTED:{key}:sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}]"
    if EMAIL_RE.search(text):
        if mode == "prod":
            return f"[REDACTED:email:{sha256_text(text)[:23]}]"
        return EMAIL_RE.sub("[REDACTED:email]", text)
    if PHONE_RE.search(text) and ("phone" in key.lower() or mode == "prod"):
        return f"[REDACTED:phone:{sha256_text(text)[:23]}]"
    if mode == "prod" and len(text) > 2000:
        return {"redacted_summary_hash": sha256_text(text), "redacted_length": len(text)}
    return text


def redact_for_langfuse(value: Any, *, mode: str = "prod") -> Any:
    """Return a Langfuse-safe object or raise ``RedactionBlocked``.

    Production payloads must contain refs, hashes, counts, labels, and short
    redacted summaries only. Known raw private-material keys are rejected
    instead of silently scrubbed so unsafe trace calls are visible locally.
    """

    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.lower()
            if normalized_key in BLOCKED_KEYS:
                raise RedactionBlocked(f"blocked protected key: {key}")
            if SECRET_KEY_RE.search(key):
                safe[key] = _safe_scalar(item, key=key, mode=mode)
            else:
                safe[key] = redact_for_langfuse(item, mode=mode)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [redact_for_langfuse(item, mode=mode) for item in value]
    return _safe_scalar(value, key="", mode=mode)


def assert_langfuse_safe(value: Any) -> None:
    redact_for_langfuse(value, mode="prod")


def safe_doc_hash(value: Any) -> str:
    return sha256_text(_canonical_json(value))
