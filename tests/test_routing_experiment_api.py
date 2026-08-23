from __future__ import annotations

from dataclasses import dataclass

import pytest

from gateway.research_lab.routing_experiment_api import RoutingExperimentApiService
from gateway.research_lab.routing_experiment_store import RoutingExperimentStoreError
from tests.test_intent_routing_experiments_v2 import _spec


@dataclass
class _Store:
    spec: object | None = None
    submitted: list[object] | None = None

    def __post_init__(self):
        self.submitted = []

    def submit(self, spec, *, execution_envelope=None):
        assert execution_envelope is None
        self.submitted.append(spec)
        return {"idempotent": False}

    def load_spec(self, experiment_hash):
        if self.spec is not None and self.spec.experiment_hash() == experiment_hash:
            return self.spec
        return None

    def execution_request(self, _experiment_hash):
        return None


class _Admission:
    def admit(self, _spec):
        return None


def test_api_service_persists_canonical_spec_without_runtime_or_provider_path():
    spec, _adapters, _labels, _tool, _source_tool = _spec()
    store = _Store()
    service = RoutingExperimentApiService(
        store_factory=lambda: store,
        admission_authority=_Admission(),
    )

    response = service.submit(spec.to_dict())

    assert response["experiment_hash"] == spec.experiment_hash()
    assert response["execution_started"] is False
    assert response["provider_execution"] == "not_requested_by_api"
    assert "spec" not in response
    assert "unit_refs" not in str(response)
    assert len(store.submitted) == 1


def test_api_service_status_is_immutable_and_missing_specs_are_not_found():
    spec, _adapters, _labels, _tool, _source_tool = _spec()
    store = _Store(spec=spec)
    service = RoutingExperimentApiService(
        store_factory=lambda: store,
        admission_authority=_Admission(),
    )

    status = service.status(spec.experiment_hash())

    assert status["status"] == "submitted"
    assert "spec" not in status
    with pytest.raises(KeyError):
        service.status("sha256:" + "f" * 64)


def test_api_service_does_not_hide_durable_authority_errors():
    class BrokenStore:
        def submit(self, _spec, *, execution_envelope=None):
            del execution_envelope
            raise RoutingExperimentStoreError("authority unavailable")

    spec, _adapters, _labels, _tool, _source_tool = _spec()
    service = RoutingExperimentApiService(
        store_factory=BrokenStore,
        admission_authority=_Admission(),
    )
    with pytest.raises(RoutingExperimentStoreError):
        service.submit(spec.to_dict())
