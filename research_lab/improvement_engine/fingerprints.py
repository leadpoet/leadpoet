"""Stable issue fingerprinting."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .models import EngineTraceEvent


UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}", re.I)


def normalize_reason(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = UUID_RE.sub("<uuid>", text)
    text = SHA_RE.sub("sha256:<hash>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:240] or "unknown"


def issue_fingerprint(event: EngineTraceEvent) -> str:
    fields = [
        event.failure_category,
        event.runtime_stage,
        normalize_reason(event.normalized_failure_reason),
        event.component,
        str(event.metadata.get("schema_error_path") or ""),
        str(event.metadata.get("scoring_version") or ""),
        str(event.metadata.get("evaluator_version") or ""),
    ]
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
