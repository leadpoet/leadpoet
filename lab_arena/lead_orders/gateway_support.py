"""The one lead-order surface the public gateway imports.

Keeps the gateway's reviewed ``lab_arena`` import contract to a single
binding (tests/lab_arena/test_lab_arena_boundary.py). Production wiring,
which pulls in KMS, chain and provider clients, is imported only when the
mode is not ``off``.
"""

from __future__ import annotations

from typing import Any, Tuple

from lab_arena.chain import verify_hotkey_signature
from lab_arena.lead_orders.auth import AuthError, verify_buyer_request, verify_runner
from lab_arena.lead_orders.broker import BrokerRequest, LeadOrderBrokerError
from lab_arena.lead_orders.config import LeadOrdersConfigError, load_config
from lab_arena.lead_orders.service import LeadOrderServiceError

REQUEST_ERRORS = (AuthError, LeadOrderServiceError, LeadOrderBrokerError)


def build_state() -> Tuple[Any, Any, Any]:
    """``(config, service, broker)``; service and broker are ``None`` when off."""

    config = load_config()
    if not config.enabled:
        return config, None, None
    from lab_arena.lead_orders.wiring import build_from_environment

    return build_from_environment()


__all__ = [
    "AuthError",
    "BrokerRequest",
    "LeadOrdersConfigError",
    "REQUEST_ERRORS",
    "build_state",
    "verify_buyer_request",
    "verify_hotkey_signature",
    "verify_runner",
]
