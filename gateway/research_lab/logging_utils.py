"""Small formatting helpers for Research Lab worker logs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


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
