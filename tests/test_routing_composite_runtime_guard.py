from __future__ import annotations

import pytest

from gateway.research_lab.routing_experiment_runtime import (
    RoutingExperimentRuntimeError,
    _require_reviewed_direct_prepared_call,
)
from gateway.research_lab.routing_provider_bindings import (
    PreparedRoutingProviderCall,
    PreparedRoutingProviderWorkflow,
)
from research_lab.routing_experiments import ProviderBindingIdentity


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _binding() -> ProviderBindingIdentity:
    return ProviderBindingIdentity(
        binding_id="predictleads-composite",
        provider_id="deepline",
        tool_id="intent.source_add.predictleads_connections",
        source_lineage_id="deepline.predictleads.connections",
        adapter_version="v1",
        manifest_hash=_hash("1"),
        capability_hash=_hash("2"),
        execution_contract_hash=_hash("3"),
        cost_model_hash=_hash("4"),
    )


def _workflow() -> PreparedRoutingProviderWorkflow:
    return PreparedRoutingProviderWorkflow(
        binding=_binding(),
        binding_manifest_hash=_hash("1"),
        binding_catalog_manifest_hash=_hash("5"),
        binding_catalog_version="catalog-v1",
        unit_ref="company-1",
        unit_input_hash=_hash("6"),
        unit_dataset_manifest_hash=_hash("7"),
        unit_set_hash=_hash("8"),
        model_binding_requirements_hash=_hash("9"),
        workflow_id="intent.source_add.predictleads_connections",
        workflow_manifest_hash=_hash("a"),
        workflow_input={
            "company_domain": "acme.example",
            "minimum_date": "2026-08-01",
            "maximum_date": "2026-08-18",
        },
        workflow_input_hash=_hash("b"),
        ordered_actions=(
            "predictleads_company",
            "predictleads_company_connections",
            "predictleads_company",
        ),
        branch_optional_actions=(),
        max_calls=3,
        timeout_ms=30_000,
        credit_ceiling_microunits=1_680_000,
        max_results=1,
        retry_policy_hash=_hash("c"),
        evidence_contract_hash=_hash("d"),
        output_contract_hash=_hash("e"),
    )


def test_composite_workflow_stops_before_any_reservation_or_tee_dispatch():
    with pytest.raises(
        RoutingExperimentRuntimeError,
        match="protected aggregate receipt schema is not released",
    ):
        _require_reviewed_direct_prepared_call(_workflow())


def test_malformed_prepared_value_fails_closed_instead_of_falling_through():
    with pytest.raises(
        RoutingExperimentRuntimeError,
        match="not a reviewed direct action",
    ):
        _require_reviewed_direct_prepared_call({"action_id": "predictleads_company"})


def test_direct_prepared_action_keeps_the_existing_dispatch_type():
    prepared = PreparedRoutingProviderCall.__new__(PreparedRoutingProviderCall)
    assert _require_reviewed_direct_prepared_call(prepared) is prepared
