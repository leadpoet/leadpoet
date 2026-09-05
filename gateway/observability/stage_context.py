"""Carry the name of the internal stage a request last entered.

A gateway request that fails deep inside a multi-stage handler answers with one
generic status code, so the exported span says a request failed but never where.
The stage names are a fixed, low-cardinality vocabulary owned by this codebase —
they contain no identifiers, hotkeys, epochs, or caller-supplied text — so the
last stage entered is safe to attach to the span for a failed request.

The value is per-request and never read on success.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

_current_stage: ContextVar[Optional[str]] = ContextVar(
    "gateway_current_stage", default=None
)


def enter_stage(stage: str) -> None:
    """Record the stage this request has just entered."""

    try:
        _current_stage.set(str(stage))
    except Exception:
        # Diagnostics must never affect request behavior.
        pass


def reset_stage() -> None:
    """Clear the recorded stage at the start of a request."""

    try:
        _current_stage.set(None)
    except Exception:
        pass


def current_stage() -> Optional[str]:
    """Return the stage this request last entered, if any."""

    try:
        return _current_stage.get()
    except Exception:
        return None
