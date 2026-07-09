"""Small formatting helpers for Research Lab worker logs.

Also home to the shared event-doc error sanitizers (bug #36 write-side
discipline): raw ``str(exc)`` from providers/infra can carry secret-shaped
markers (ECR registry hosts, Supabase roles, credentialed URLs) that either
poison the audit-bundle secret scan or trip the scripts/34 event_doc CHECK —
which silently drops the event under the contained-write pattern. Every event
doc that records an error should go through ``safe_event_error_text`` /
``event_error_diagnostics`` instead of ``str(exc)``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


def safe_event_error_text(exc: BaseException) -> str:
    """Marker-redacted, length-capped error text safe for event docs."""
    text = f"{exc.__class__.__name__}: {str(exc)}"
    for marker in ("sk-or-", "sb_secret", "service_role", "openrouter_api_key", "raw_secret"):
        text = re.sub(re.escape(marker), "[redacted]", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://[^/\s:]+:[^@\s]+@", "https://[redacted]@", text)
    return text[:500]


def runtime_error_diagnostics(error_text: str) -> dict[str, Any]:
    """Return DB-safe runtime diagnostics without provider URLs or request text."""

    lowered = error_text.lower()
    provider = "unknown"
    if "scrapingdog" in lowered:
        provider = "scrapingdog"
    elif "exa" in lowered:
        provider = "exa"
    elif "openrouter" in lowered:
        provider = "openrouter"
    elif "deepline" in lowered:
        provider = "deepline"

    status = 0
    for code in (400, 401, 402, 403, 404, 408, 409, 410, 429, 500, 502, 503, 504):
        if f"http error {code}" in lowered or f"status={code}" in lowered or f'"status":{code}' in lowered:
            status = code
            break

    if status >= 500:
        category = "provider_http_5xx"
    elif status >= 400:
        category = "provider_http_4xx"
    else:
        category = "runtime_provider_error"

    return {
        "error_class": "PrivateModelRuntimeError" if "privatemodelruntimeerror" in lowered else "RuntimeError",
        "provider": provider,
        "status": status,
        "category": category,
    }


def event_error_diagnostics(exc: BaseException) -> dict[str, Any]:
    """DB-safe structured error doc for audit-visible event docs."""
    diagnostics = runtime_error_diagnostics(f"{exc.__class__.__name__}: {exc}")
    diagnostics["error_class"] = exc.__class__.__name__[:120]
    return diagnostics


def compact_ref(value: Any, *, keep: int = 12) -> str:
    text = str(value or "")
    if not text:
        return "-"
    if len(text) <= keep + 3:
        return text
    return f"{text[:keep]}..."


def format_worker_block(title: str, rows: Mapping[str, Any] | Iterable[tuple[str, Any]], *, width: int = 80) -> str:
    if isinstance(rows, Mapping):
        items = rows.items()
    else:
        items = rows
    border = "=" * width
    lines = ["", border, title, border]
    for label, value in items:
        display = "-" if value is None or value == "" else str(value)
        lines.append(f"   {label:<24}: {display}")
    lines.append(border)
    return "\n".join(lines)


def format_worker_line(prefix: str, **fields: Any) -> str:
    parts = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    suffix = " ".join(parts)
    return f"{prefix}: {suffix}" if suffix else prefix
