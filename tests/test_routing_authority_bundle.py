from __future__ import annotations

import base64
from dataclasses import asdict

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from gateway.research_lab.routing_authority_bundle import (
    RoutingAuthorityBundleError,
    load_verified_routing_authority_bundle,
)
from gateway.research_lab.routing_provider_bindings import (
    ROUTING_BINDING_CATALOG_SCHEMA,
    ROUTING_UNIT_DATASET_SCHEMA,
)
from research_lab.canonical import sha256_json
from tests.test_routing_experiment_artifacts import _documents
from tests.test_routing_provider_bindings import _authorities


def _signed(document: dict, key: ec.EllipticCurvePrivateKey, signature_ref: str) -> dict:
    payload = {**document, "signature_ref": signature_ref}
    payload["manifest_hash"] = sha256_json({k: v for k, v in payload.items() if k != "manifest_hash"})
    return payload


def _key_material() -> tuple[dict[str, ec.EllipticCurvePrivateKey], dict[str, str]]:
    keys = {role: ec.generate_private_key(ec.SECP256R1()) for role in ("artifact", "lineage", "binding_catalog", "unit_dataset")}
    public = {
        role: key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        for role, key in keys.items()
    }
    return keys, public


def _bundle() -> tuple[dict, dict[str, str]]:
    pointer_uri, immutable_uri, lineage_uri, private, lineage = _documents()
    keys, public = _key_material()
    artifact_pointer = dict(private)
    artifact_manifest = dict(private)
    artifact_lineage = dict(lineage)
    binding, catalog, dataset = _authorities()
    binding_manifest = next(iter(catalog.bindings.values()))
    catalog_uri = catalog.manifest_uri
    catalog_doc = _signed(
        {
            "schema_version": ROUTING_BINDING_CATALOG_SCHEMA,
            "manifest_uri": catalog_uri,
            "catalog_version": catalog.catalog_version,
            "bindings": [
                {
                    "binding": binding_manifest.binding.to_dict(),
                    "compiler_family": binding_manifest.compiler_family,
                    "transport_id": binding_manifest.transport_id,
                    "execution_kind": binding_manifest.execution_kind,
                    "action_id": binding_manifest.action_id,
                    "workflow_id": binding_manifest.workflow_id,
                    "workflow_manifest_hash": binding_manifest.workflow_manifest_hash,
                    "input_projection": dict(binding_manifest.input_projection),
                    "input_constants": dict(binding_manifest.input_constants),
                    "model_binding_requirements_hash": binding_manifest.model_binding_requirements_hash,
                    "output_contract_hash": binding_manifest.output_contract_hash,
                    "evidence_contract_hash": binding_manifest.evidence_contract_hash,
                    "retry_policy_hash": binding_manifest.retry_policy_hash,
                    "max_results": binding_manifest.max_results,
                    "timeout_ms": binding_manifest.timeout_ms,
                    "credit_ceiling_microunits": binding_manifest.credit_ceiling_microunits,
                }
            ],
        },
        keys["binding_catalog"],
        catalog.signature_ref,
    )
    dataset_doc = _signed(
        {
            "schema_version": ROUTING_UNIT_DATASET_SCHEMA,
            "manifest_uri": dataset.manifest_uri,
            "units": {key: dict(value) for key, value in dataset.units.items()},
            "unit_set_hash": dataset.unit_set_hash,
            "provenance_hash": dataset.provenance_hash,
        },
        keys["unit_dataset"],
        dataset.signature_ref,
    )
    # Private artifact documents are signed by the model key.  The test uses
    # fresh local signatures so it never depends on KMS or a provider.
    artifact_pointer = _signed({k: v for k, v in artifact_pointer.items() if k not in {"manifest_hash", "signature_ref"}}, keys["artifact"], private["signature_ref"])
    artifact_manifest = dict(artifact_pointer)
    artifact_lineage = _signed({k: v for k, v in artifact_lineage.items() if k not in {"manifest_hash", "signature_ref"}}, keys["lineage"], lineage["signature_ref"])
    documents = {
        "artifact_pointer": artifact_pointer,
        "artifact_manifest": artifact_manifest,
        "artifact_lineage": artifact_lineage,
        "binding_catalog": catalog_doc,
        "unit_dataset": dataset_doc,
    }
    role_by_document = {
        "artifact_pointer": "artifact",
        "artifact_manifest": "artifact",
        "artifact_lineage": "lineage",
        "binding_catalog": "binding_catalog",
        "unit_dataset": "unit_dataset",
    }
    signatures = {}
    for name, document in documents.items():
        role = role_by_document[name]
        signatures[name] = {
            "key_id": role + "-key",
            "signature": base64.b64encode(
                keys[role].sign(document["manifest_hash"].encode(), ec.ECDSA(hashes.SHA256()))
            ).decode(),
        }
    bundle = {
        "schema_version": "leadpoet.routing_authority_bundle.v1",
        "pointer_uri": pointer_uri,
        "key_ids": {
            "artifact": "artifact-key",
            "lineage": "lineage-key",
            "binding_catalog": "binding_catalog-key",
            "unit_dataset": "unit_dataset-key",
        },
        "verification_keys": {
            "artifact-key": public["artifact"],
            "lineage-key": public["lineage"],
            "binding_catalog-key": public["binding_catalog"],
            "unit_dataset-key": public["unit_dataset"],
        },
        "documents": documents,
        "signatures": signatures,
    }
    pins = {
        "artifact": public["artifact"],
        "lineage": public["lineage"],
        "binding_catalog": public["binding_catalog"],
        "unit_dataset": public["unit_dataset"],
    }
    return bundle, pins


def test_bundle_reconstructs_typed_authorities_without_uri_fetching():
    bundle, pins = _bundle()
    result = load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)
    assert result.artifact_lineage.commit_sha == "a" * 40
    assert result.binding_catalog.catalog_version == "catalog-001"
    assert tuple(result.unit_dataset.units) == ("company-1",)
    assert result.bundle_hash == sha256_json(bundle)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["documents"]["artifact_pointer"].update({"unexpected": True}),
        lambda value: value["documents"]["binding_catalog"].update({"unexpected": True}),
        lambda value: value["signatures"].pop("unit_dataset"),
        lambda value: value["verification_keys"].update({"unknown": "not-a-key"}),
        lambda value: value.update({"network_endpoint": "https://example.invalid"}),
    ],
)
def test_bundle_rejects_unknown_or_incomplete_authority_material(mutate):
    bundle, pins = _bundle()
    mutate(bundle)
    with pytest.raises(RoutingAuthorityBundleError):
        load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)


def test_bundle_rejects_unpinned_key_and_tampered_signature():
    bundle, pins = _bundle()
    bundle["verification_keys"]["artifact-key"] = pins["lineage"]
    with pytest.raises(RoutingAuthorityBundleError, match="does not match pin"):
        load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)

    bundle, pins = _bundle()
    bundle["signatures"]["artifact_lineage"]["signature"] = base64.b64encode(b"bad").decode()
    with pytest.raises(RoutingAuthorityBundleError):
        load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)


def test_bundle_rejects_missing_pins_before_any_document_resolution():
    bundle, _pins = _bundle()
    with pytest.raises(RoutingAuthorityBundleError, match="pins are incomplete"):
        load_verified_routing_authority_bundle(bundle, pinned_public_keys={})


def test_bundle_rejects_a_valid_key_from_the_wrong_authority_role():
    bundle, pins = _bundle()
    bundle["signatures"]["artifact_pointer"]["key_id"] = "lineage-key"
    with pytest.raises(RoutingAuthorityBundleError, match="wrong role"):
        load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)


def test_bundle_rejects_duplicate_document_uris_after_valid_signature_checks():
    bundle, pins = _bundle()
    unit_key = ec.generate_private_key(ec.SECP256R1())
    unit_public = unit_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    bundle["verification_keys"]["unit_dataset-key"] = unit_public
    pins["unit_dataset"] = unit_public
    document = bundle["documents"]["unit_dataset"]
    document["manifest_uri"] = bundle["documents"]["binding_catalog"][
        "manifest_uri"
    ]
    document["manifest_hash"] = sha256_json(
        {key: value for key, value in document.items() if key != "manifest_hash"}
    )
    bundle["signatures"]["unit_dataset"]["signature"] = base64.b64encode(
        unit_key.sign(
            document["manifest_hash"].encode(),
            ec.ECDSA(hashes.SHA256()),
        )
    ).decode()
    with pytest.raises(RoutingAuthorityBundleError, match="URI is duplicated"):
        load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)
