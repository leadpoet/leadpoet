"""Feature-specific routing readiness does not alter gateway liveness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gateway import main
from gateway.research_lab.routing_experiment_api import RoutingExperimentApiService


@pytest.mark.asyncio
async def test_disabled_routing_readiness_is_neutral(monkeypatch) -> None:
    for name in (
        "RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED",
        "RESEARCH_LAB_ROUTING_EXPERIMENT_LIVE_ENABLED",
        "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED",
        "RESEARCH_LAB_ROUTING_PRODUCT_COMPOSITION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main.app, "state", SimpleNamespace())

    result = await main.routing_experiment_readiness()

    assert result["status"] == "disabled"
    assert result["routing"]["enabled"] is False


@pytest.mark.asyncio
async def test_enabled_unregistered_routing_readiness_is_503(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", "true")
    monkeypatch.setattr(main.app, "state", SimpleNamespace())

    with pytest.raises(HTTPException) as raised:
        await main.routing_experiment_readiness()

    assert raised.value.status_code == 503
    assert raised.value.detail["routing"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_fully_registered_routing_readiness_is_ready(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", "true")
    service = RoutingExperimentApiService()
    composition = SimpleNamespace(
        api_service=service,
        run_factory=SimpleNamespace(name="attested_provider_broker_v2"),
    )

    class _Supervisor:
        def health(self):
            return {
                "supervised": True,
                "registered": True,
                "ready": True,
            }

    monkeypatch.setattr(
        main.app,
        "state",
        SimpleNamespace(
            reviewed_routing_product_composition=composition,
            routing_experiment_api_service=service,
            reviewed_routing_consumer_supervisor=_Supervisor(),
        ),
    )

    result = await main.routing_experiment_readiness()

    assert result["status"] == "ready"
    assert result["routing"]["consumer"]["ready"] is True
