"""Bounded readiness state for the reviewed routing product registration."""

from __future__ import annotations

import os
from typing import Any


_ROUTING_FEATURE_ENV_NAMES = (
    "RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED",
    "RESEARCH_LAB_ROUTING_EXPERIMENT_LIVE_ENABLED",
    "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED",
)
_ROUTING_PRODUCT_COMPOSITION_ENV = "RESEARCH_LAB_ROUTING_PRODUCT_COMPOSITION"
_ROUTING_PRODUCT_COMPOSITION_VALUE = "reviewed_v2"
_ROUTING_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def routing_registration_health(application: Any) -> dict[str, object]:
    """Return bounded routing registration state for authority readiness.

    The gateway cannot observe a separate consumer process implicitly. A
    reviewed deployment must publish all three consumer state flags after it
    has registered the exact factory, started supervision, and received the
    child's ready signal. Missing or non-boolean state is not ready.
    """

    state = getattr(application, "state", None)
    composition = getattr(state, "reviewed_routing_product_composition", None)
    api_service = getattr(state, "routing_experiment_api_service", None)
    api_ready = (
        composition is not None
        and api_service is not None
        and api_service is getattr(composition, "api_service", None)
        and getattr(getattr(composition, "run_factory", None), "name", None)
        == "attested_provider_broker_v2"
    )
    configured = composition is not None or any(
        str(os.environ.get(name, "")).strip().lower() in _ROUTING_TRUE_VALUES
        for name in _ROUTING_FEATURE_ENV_NAMES
    ) or (
        str(os.environ.get(_ROUTING_PRODUCT_COMPOSITION_ENV, "")).strip().lower()
        == _ROUTING_PRODUCT_COMPOSITION_VALUE
    )
    supervisor = getattr(state, "reviewed_routing_consumer_supervisor", None)
    live_health = getattr(supervisor, "health", None)
    if callable(live_health):
        try:
            snapshot = live_health()
        except Exception:
            snapshot = {}
        consumer = {
            "supervised": snapshot.get("supervised") is True,
            "registered": snapshot.get("registered") is True,
            "ready": snapshot.get("ready") is True,
        }
    else:
        # Readiness is a live process fact.  Startup booleans can become stale
        # after a child exits, so they must never substitute for an installed
        # supervisor health authority.
        consumer = {
            "supervised": False,
            "registered": False,
            "ready": False,
        }
    consumer_ready = all(consumer.values())
    status = (
        "disabled"
        if not configured
        else "ready"
        if api_ready and consumer_ready
        else "unavailable"
    )
    return {
        "status": status,
        "enabled": configured,
        "api_composition": {"status": "ready" if api_ready else "unavailable"},
        "consumer": {
            "status": "ready" if consumer_ready else "unavailable",
            **consumer,
        },
    }


__all__ = ["routing_registration_health"]
