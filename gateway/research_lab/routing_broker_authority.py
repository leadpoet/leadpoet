"""Process-local bridge from the protected routing grant to ProviderBrokerV2.

The broker is created by the coordinator service, while the signed grant
authority is constructed by the routing worker.  This registry connects the
two without allowing the broker to accept caller-selected provider fields.
Until an attested authority registers, validation fails closed.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Mapping


class RoutingBrokerAuthorityUnavailable(RuntimeError):
    """No signed routing grant authority is registered for this boot."""


_LOCK = RLock()
_VALIDATOR: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None


def register_routing_broker_authority_v2(
    validator: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
) -> None:
    if not callable(validator):
        raise TypeError("routing broker authority validator must be callable")
    global _VALIDATOR
    with _LOCK:
        _VALIDATOR = validator


def clear_routing_broker_authority_v2() -> None:
    global _VALIDATOR
    with _LOCK:
        _VALIDATOR = None


def validate_routing_broker_authorization_v2(
    proof: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    with _LOCK:
        validator = _VALIDATOR
    if validator is None:
        raise RoutingBrokerAuthorityUnavailable(
            "routing broker signed grant authority is unavailable"
        )
    validator(proof, request)


__all__ = [
    "RoutingBrokerAuthorityUnavailable",
    "register_routing_broker_authority_v2",
    "clear_routing_broker_authority_v2",
    "validate_routing_broker_authorization_v2",
]
