"""The one tool-market surface the public gateway imports.

Keeps the gateway's reviewed ``lab_arena`` import contract to a single
binding (tests/lab_arena/test_lab_arena_boundary.py). Production wiring,
which pulls in the chain and the vault client, is imported only when the
mode is ``live``.
"""

from __future__ import annotations

from typing import Any, Tuple

from lab_arena.chain import verify_hotkey_signature
from lab_arena.lead_orders.auth import AuthError, verify_buyer_request
from lab_arena.store import ArenaStoreError
from lab_arena.tool_market.config import ToolMarketConfigError, load_config
from lab_arena.tool_market.market import ToolMarketError


def build_state() -> Tuple[Any, Any]:
    """``(config, market)``; the market is ``None`` when off."""

    config = load_config()
    if not config.enabled:
        return config, None
    from lab_arena.tool_market.wiring import build_from_environment

    return build_from_environment(config)


__all__ = [
    "ArenaStoreError",
    "AuthError",
    "ToolMarketConfigError",
    "ToolMarketError",
    "build_state",
    "verify_buyer_request",
    "verify_hotkey_signature",
]
