from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from gateway.research_lab.routing_model_binding_observation import (
    ROUTING_MODEL_BINDING_REQUIREMENTS_SCHEMA_V2,
    RoutingModelBindingObservationError,
    VerifiedRoutingModelBindingRequirements,
    observe_routing_model_bindings_v2,
    routing_model_binding_identity_hash,
    routing_model_binding_requirements_hash,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import ProviderBindingIdentity
from tests.routing_experiment_authority_fixture import _signed_receipt


def _raw_manifest_digest(manifest: dict) -> str:
    rendered = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _binding_and_row(*, provider_id: str = "bloomberry_jobs"):
    manifest = {
        "schema_version": "leadpoet.intent-source-binding-manifest:v1",
        "tool_id": "intent.source_add." + provider_id,
        "provider_id": provider_id,
        "stage": "intent_evidence",
        "execution_mode": "invoke",
        "cost_class": "metered",
        "unit_cost": 0.09,
        "max_calls": 1,
        "max_results": 1,
        "timeout_seconds": 30.0,
        "capabilities": ["intent.hiring"],
        "intent_categories": ["HIRING"],
        "evidence_types": ["job_posting"],
        "category_contracts": [
            {
                "category": "HIRING",
                "capabilities": ["intent.hiring"],
                "evidence_types": ["job_posting"],
                "requirements": ["domain_scoped_request"],
            }
        ],
        "binding_requirements": [
            "domain_scoped_request",
            "bloomberry_search_job_postings",
        ],
    }
    raw_digest = _raw_manifest_digest(manifest)
    binding = ProviderBindingIdentity(
        binding_id="deepline-" + provider_id + "-v1",
        provider_id=provider_id,
        tool_id=manifest["tool_id"],
        source_lineage_id="deepline." + provider_id,
        adapter_version="v1",
        manifest_hash="sha256:" + raw_digest,
        capability_hash="sha256:" + "1" * 64,
        execution_contract_hash="sha256:" + "2" * 64,
        cost_model_hash="sha256:" + "3" * 64,
    )
    row = {
        "tool_id": manifest["tool_id"],
        "revision": "source-add-" + raw_digest[:12],
        "manifest_sha256": raw_digest,
        "manifest": manifest,
    }
    return binding, row


def _metadata(*, row: dict, **changes) -> dict:
    value = {
        "compiler_version": "routing-compiler-v2",
        "catalog_sha256": sha256_json({"catalog": 1}),
        "policy_sha256": sha256_json({"policy": 1}),
        "source_add_manifest_attestations": [
            {
                "tool_id": row["tool_id"],
                "revision": row["revision"],
                "manifest_sha256": row["manifest_sha256"],
            }
        ],
        "source_add_binding_manifests": [row],
        "private_bindings_exposed": False,
    }
    value.update(changes)
    return value


def test_model_binding_probe_matches_exact_manifest_and_derives_requirements():
    binding, row = _binding_and_row()
    result = observe_routing_model_bindings_v2(
        runtime_metadata=_metadata(row=row),
        provider_bindings=(binding,),
        artifact_lineage_hash=sha256_json({"artifact": 1}),
    )

    identity = routing_model_binding_identity_hash(binding)
    assert result["requirements"] == [
        {
            "binding_identity_hash": identity,
            "requirements_hash": routing_model_binding_requirements_hash(
                row["manifest"]
            ),
        }
    ]
    assert result["requirements"][0]["requirements_hash"] == sha256_json(
        {
            "schema_version": ROUTING_MODEL_BINDING_REQUIREMENTS_SCHEMA_V2,
            "binding_requirements": sorted(row["manifest"]["binding_requirements"]),
        }
    )


def test_model_binding_requirements_reject_legacy_compatibility_receipt():
    binding, row = _binding_and_row()
    result = observe_routing_model_bindings_v2(
        runtime_metadata=_metadata(row=row),
        provider_bindings=(binding,),
        artifact_lineage_hash=sha256_json({"artifact": 1}),
    )
    receipt = _signed_receipt(
        purpose="research_lab.model_compatibility.v2",
        input_root=result["request_root"],
        output_root=sha256_json(result),
        index=90,
    )
    with pytest.raises(RoutingModelBindingObservationError, match="receipt"):
        VerifiedRoutingModelBindingRequirements.from_attested(result, receipt)


@pytest.mark.parametrize(
    "changes",
    [
        {"source_add_binding_manifests": None},
        {"private_bindings_exposed": True},
        {"catalog_sha256": "not-a-hash"},
    ],
)
def test_model_binding_probe_rejects_invalid_top_level_metadata(changes):
    binding, row = _binding_and_row()
    with pytest.raises(RoutingModelBindingObservationError):
        observe_routing_model_bindings_v2(
            runtime_metadata=_metadata(row=row, **changes),
            provider_bindings=(binding,),
            artifact_lineage_hash=sha256_json({"artifact": 1}),
        )


def test_model_binding_probe_rejects_duplicate_or_extra_manifest_rows():
    binding, row = _binding_and_row()
    duplicate_metadata = _metadata(row=row)
    duplicate_metadata["source_add_binding_manifests"] = [row, row]
    duplicate_metadata["source_add_manifest_attestations"] = [
        duplicate_metadata["source_add_manifest_attestations"][0]
    ] * 2
    with pytest.raises(RoutingModelBindingObservationError):
        observe_routing_model_bindings_v2(
            runtime_metadata=duplicate_metadata,
            provider_bindings=(binding,),
            artifact_lineage_hash=sha256_json({"artifact": 1}),
        )

    extra_row = dict(row)
    extra_row["unexpected"] = True
    extra_metadata = _metadata(row=row)
    extra_metadata["source_add_binding_manifests"] = [extra_row]
    with pytest.raises(RoutingModelBindingObservationError):
        observe_routing_model_bindings_v2(
            runtime_metadata=extra_metadata,
            provider_bindings=(binding,),
            artifact_lineage_hash=sha256_json({"artifact": 1}),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row["manifest"].update({"provider_id": "other_provider"}),
        lambda row: row["manifest"].update({"stage": "candidate_acquisition"}),
        lambda row: row["manifest"].update({"max_calls": True}),
        lambda row: row["manifest"].update({"binding_requirements": [True]}),
    ],
)
def test_model_binding_probe_rejects_substituted_or_malformed_manifest(mutate):
    binding, row = _binding_and_row()
    mutate(row)
    metadata = _metadata(row=row)
    with pytest.raises(RoutingModelBindingObservationError):
        observe_routing_model_bindings_v2(
            runtime_metadata=metadata,
            provider_bindings=(binding,),
            artifact_lineage_hash=sha256_json({"artifact": 1}),
        )


def test_model_binding_probe_rejects_undeclared_binding_and_manifest_hash_drift():
    binding, row = _binding_and_row()
    undeclared = replace(binding, tool_id="intent.source_add.not_declared")
    with pytest.raises(RoutingModelBindingObservationError):
        observe_routing_model_bindings_v2(
            runtime_metadata=_metadata(row=row),
            provider_bindings=(undeclared,),
            artifact_lineage_hash=sha256_json({"artifact": 1}),
        )

    drifted = replace(binding, manifest_hash="sha256:" + "f" * 64)
    with pytest.raises(RoutingModelBindingObservationError):
        observe_routing_model_bindings_v2(
            runtime_metadata=_metadata(row=row),
            provider_bindings=(drifted,),
            artifact_lineage_hash=sha256_json({"artifact": 1}),
        )
