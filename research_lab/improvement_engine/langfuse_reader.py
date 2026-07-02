"""Placeholder Langfuse reader.

The v1 scanner reads canonical Leadpoet stores first. This module exists so
future Langfuse-assisted scans can be added without changing scanner callers.
"""

from __future__ import annotations

from typing import Any


async def read_langfuse_trace_summaries(*_: Any, **__: Any) -> list[dict[str, Any]]:
    return []
