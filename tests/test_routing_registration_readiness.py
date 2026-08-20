"""Routing registration state must remain bounded and fail closed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.research_lab.routing_experiment_api import RoutingExperimentApiService
from gateway.research_lab.routing_registration import routing_registration_health


def _application() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


def _install_api_composition(application: SimpleNamespace) -> None:
    service = RoutingExperimentApiService()
    composition = SimpleNamespace(
        api_service=service,
        run_factory=SimpleNamespace(name="exact_model_runner_v3"),
    )
    application.state.reviewed_routing_product_composition = composition
    application.state.routing_experiment_api_service = service


def _install_ready_supervisor(application: SimpleNamespace) -> None:
    class _Supervisor:
        def health(self):
            return {
                "supervised": True,
                "registered": True,
                "ready": True,
            }

    application.state.reviewed_routing_consumer_supervisor = _Supervisor()


def test_no_composition_is_unavailable_when_routing_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", "true")
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED", raising=False)

    health = routing_registration_health(_application())

    assert health["enabled"] is True
    assert health["status"] == "unavailable"
    assert health["api_composition"] == {"status": "unavailable"}
    assert health["consumer"]["status"] == "unavailable"


def test_composition_without_consumer_is_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED", raising=False)
    application = _application()
    _install_api_composition(application)

    health = routing_registration_health(application)

    assert health["enabled"] is True
    assert health["status"] == "unavailable"
    assert health["api_composition"] == {"status": "ready"}
    assert health["consumer"]["status"] == "unavailable"


def test_substituted_api_service_is_not_ready(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", "true")
    application = _application()
    _install_api_composition(application)
    application.state.routing_experiment_api_service = RoutingExperimentApiService()
    application.state.reviewed_routing_consumer_supervised = True
    application.state.reviewed_routing_consumer_registered = True
    application.state.reviewed_routing_consumer_ready = True

    health = routing_registration_health(application)

    assert health["status"] == "unavailable"
    assert health["api_composition"] == {"status": "unavailable"}


def test_disabled_routing_is_neutral(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED", raising=False)

    health = routing_registration_health(_application())

    assert health == {
        "status": "disabled",
        "enabled": False,
        "api_composition": {"status": "unavailable"},
        "consumer": {
            "status": "unavailable",
            "supervised": False,
            "registered": False,
            "ready": False,
        },
    }


def test_full_registered_state_reports_ready(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED", raising=False)
    application = _application()
    _install_api_composition(application)
    _install_ready_supervisor(application)

    health = routing_registration_health(application)

    assert health["status"] == "ready"
    assert health["api_composition"] == {"status": "ready"}
    assert health["consumer"] == {
        "status": "ready",
        "supervised": True,
        "registered": True,
        "ready": True,
    }


def test_stale_startup_flags_without_live_supervisor_are_not_ready(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", raising=False)
    monkeypatch.delenv(
        "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED", raising=False
    )
    application = _application()
    _install_api_composition(application)
    application.state.reviewed_routing_consumer_supervised = True
    application.state.reviewed_routing_consumer_registered = True
    application.state.reviewed_routing_consumer_ready = True

    health = routing_registration_health(application)

    assert health["status"] == "unavailable"
    assert health["consumer"] == {
        "status": "unavailable",
        "supervised": False,
        "registered": False,
        "ready": False,
    }


def test_live_supervisor_health_overrides_stale_startup_flags(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED", raising=False)
    application = _application()
    _install_api_composition(application)
    application.state.reviewed_routing_consumer_supervised = True
    application.state.reviewed_routing_consumer_registered = True
    application.state.reviewed_routing_consumer_ready = True

    class _Supervisor:
        def health(self):
            return {
                "supervised": True,
                "registered": False,
                "ready": False,
            }

    application.state.reviewed_routing_consumer_supervisor = _Supervisor()
    health = routing_registration_health(application)

    assert health["status"] == "unavailable"
    assert health["consumer"] == {
        "status": "unavailable",
        "supervised": True,
        "registered": False,
        "ready": False,
    }


@pytest.mark.parametrize(
    "environment_name",
    (
        "RESEARCH_LAB_ROUTING_EXPERIMENT_LIVE_ENABLED",
        "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED",
        "RESEARCH_LAB_ROUTING_PRODUCT_COMPOSITION",
    ),
)
def test_each_activation_intent_requires_registration(
    monkeypatch, environment_name: str
) -> None:
    for name in (
        "RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED",
        "RESEARCH_LAB_ROUTING_EXPERIMENT_LIVE_ENABLED",
        "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED",
        "RESEARCH_LAB_ROUTING_PRODUCT_COMPOSITION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        environment_name,
        "reviewed_v2" if environment_name.endswith("COMPOSITION") else "true",
    )

    health = routing_registration_health(_application())

    assert health["enabled"] is True
    assert health["status"] == "unavailable"
