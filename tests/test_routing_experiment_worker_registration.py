from __future__ import annotations

import pytest

from gateway.research_lab.routing_experiment_runtime import (
    RoutingExperimentRuntimeConfig,
)
from gateway.research_lab.routing_experiment_worker import (
    ROUTING_ATTESTATION_AUTHORITY_ENV,
    ROUTING_CLAIM_AUTHORITY_ENV,
    RoutingExperimentWorkerError,
    assert_reviewed_routing_runtime_registered,
    build_reviewed_routing_experiment_worker,
)


def _config() -> RoutingExperimentRuntimeConfig:
    return RoutingExperimentRuntimeConfig(
        enabled=True,
        attested_authority_mode="attested",
    )


def _environment() -> dict[str, str]:
    return {
        ROUTING_CLAIM_AUTHORITY_ENV: "supabase_v3",
        ROUTING_ATTESTATION_AUTHORITY_ENV: "tee_v2",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "service-role-test-only",
    }


def test_reviewed_worker_registration_fails_closed_without_durable_claim():
    with pytest.raises(
        RoutingExperimentWorkerError,
        match="durable claim authority is unavailable",
    ):
        assert_reviewed_routing_runtime_registered(
            _config(),
            environment={
                ROUTING_ATTESTATION_AUTHORITY_ENV: "tee_v2",
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-test-only",
            },
        )


def test_reviewed_worker_factory_is_named_and_requires_all_durable_config():
    worker = build_reviewed_routing_experiment_worker(
        worker_ref="routing-worker-test",
        config_factory=_config,
        store_factory=lambda: object(),
        environment=_environment(),
    )
    assert worker.worker_ref == "routing-worker-test"
    assert worker.service.config.enabled is True
    assert worker.service.config.attested_authority_mode == "attested"
