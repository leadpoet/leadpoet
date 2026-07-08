"""Runtime object registry for lightweight gateway metrics."""

from __future__ import annotations

from typing import Any


priority_middleware: Any = None


def set_priority_middleware(instance: Any) -> None:
    global priority_middleware
    priority_middleware = instance
