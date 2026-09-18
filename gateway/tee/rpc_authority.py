"""Fail-closed RPC authority for each physical gateway enclave role."""

from __future__ import annotations

import os
from typing import FrozenSet


COORDINATOR_ROLE = "gateway_coordinator"

COMMON_METHODS = frozenset(
    {
        "role_health",
        "v2_configure_runtime",
        "v2_get_boot_identity",
    }
)
COORDINATOR_METHODS = frozenset(
    {
        "get_event_signing_identity",
    }
)


class RPCAuthorityError(ValueError):
    """An unknown role or cross-role RPC method was requested."""


def active_enclave_role() -> str:
    role = str(os.getenv("LEADPOET_ENCLAVE_ROLE", "") or "").strip()
    if role != COORDINATOR_ROLE:
        raise RPCAuthorityError("gateway enclave role is missing or unknown")
    return role


def allowed_exact_methods(role: str) -> FrozenSet[str]:
    if role == COORDINATOR_ROLE:
        return COMMON_METHODS | COORDINATOR_METHODS
    raise RPCAuthorityError("unknown gateway enclave role")


def rpc_method_allowed(role: str, method: str) -> bool:
    normalized_method = str(method or "")
    return normalized_method in allowed_exact_methods(role)
