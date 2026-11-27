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
from lab_arena.tool_market.catalog import operation
from lab_arena.tool_market.config import ToolMarketConfigError, load_config
from lab_arena.tool_market.market import ToolMarketError


def build_state() -> Tuple[Any, Any]:
    """``(config, market)``; the market is ``None`` when off."""

    config = load_config()
    if not config.enabled:
        return config, None
    from lab_arena.tool_market.wiring import build_from_environment

    return config, build_from_environment(config)


def lead_search_validation(market: Any, token: str, request: dict) -> dict:
    """Adapt Lead Search's raw-v2 ZeroBounce contract at the buyer boundary."""
    if set(request) - {"payload", "forceToolRefresh"} or not isinstance(request.get("payload"), dict) or (
        type(request.get("forceToolRefresh", False)) is not bool
    ):
        raise ToolMarketError(400, "request_invalid")
    # This endpoint serves platform customer data. Every call requires private supply.
    result = market.call(token, {"tool": "zerobounce", "operation": "validate",
                                 "params": request["payload"], "private": True})
    status, body = result["status"], result["body"]
    op = operation("zerobounce", "validate")
    answered = 200 <= op.status(status, body) < 300 and op.units(status, body) is not None
    return {"call_id": result["call_id"], "status": "completed" if answered else "failed",
            "toolResponse": {"raw": body} if answered else {},
            "billing": {"cost_usd": result["charge_microusd"] / 1_000_000}}


__all__ = [
    "lead_search_validation",
    "ArenaStoreError",
    "AuthError",
    "ToolMarketConfigError",
    "ToolMarketError",
    "build_state",
    "verify_buyer_request",
    "verify_hotkey_signature",
]
