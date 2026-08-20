from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.research_lab.routing_experiment_runtime import (
    RoutingExperimentRuntimeConfig,
)
from gateway.research_lab.routing_experiment_worker import (
    ROUTING_ATTESTATION_AUTHORITY_ENV,
    ROUTING_CLAIM_AUTHORITY_ENV,
    RoutingExperimentWorkerError,
    _validate_exact_unit_and_label_identity,
    assert_reviewed_routing_runtime_registered,
    build_reviewed_routing_experiment_worker,
)
from research_lab.canonical import sha256_json


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


def _exact_identity_fixture():
    labels = {"unit.cal": True, "unit.hold": False}
    unit_hash = "sha256:" + "a" * 64
    spec = SimpleNamespace(
        input=SimpleNamespace(
            calibration_unit_refs=("unit.cal",),
            holdout_unit_refs=("unit.hold",),
            unit_input_set_hash=unit_hash,
            gold_label_set_hash=sha256_json(
                {"labels": sorted(labels.items())}
            ),
        )
    )
    dataset = SimpleNamespace(
        units={"unit.cal": {}, "unit.hold": {}},
        unit_set_hash=unit_hash,
    )
    return spec, dataset, labels


def test_exact_runner_binds_signed_units_and_labels_before_sql():
    spec, dataset, labels = _exact_identity_fixture()

    assert _validate_exact_unit_and_label_identity(
        spec=spec,
        unit_dataset=dataset,
        gold_labels=labels,
    ) == labels


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("unit_hash", "unit dataset hash"),
        ("missing_unit_hash", "unit dataset hash"),
        ("label_hash", "label hash"),
        ("label_type", "labels differ"),
    ),
)
def test_exact_runner_rejects_unit_or_label_identity_before_sql(
    mutation,
    message,
):
    spec, dataset, labels = _exact_identity_fixture()
    if mutation == "unit_hash":
        dataset.unit_set_hash = "sha256:" + "b" * 64
    elif mutation == "missing_unit_hash":
        spec.input.unit_input_set_hash = ""
    elif mutation == "label_hash":
        spec.input.gold_label_set_hash = "sha256:" + "c" * 64
    else:
        labels["unit.cal"] = 1

    with pytest.raises(RoutingExperimentWorkerError, match=message):
        _validate_exact_unit_and_label_identity(
            spec=spec,
            unit_dataset=dataset,
            gold_labels=labels,
        )
