from __future__ import annotations

import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from gateway.research_lab.routing_authority_bundle import (
    RoutingAuthorityBundleError,
    load_verified_routing_authority_bundle,
)
from research_lab.canonical import sha256_json
from tests.test_routing_authority_bundle import _bundle, _signed


def _new_key() -> tuple[ec.EllipticCurvePrivateKey, str]:
    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return key, public


def _artifact_documents(
    source: dict,
    *,
    bucket: str,
    commit: str,
    model_hash: str,
    key: ec.EllipticCurvePrivateKey,
    branch: str,
    artifact_signature_ref: str,
    lineage_signature_ref: str,
) -> dict:
    pointer_uri = f"s3://{bucket}/branches/{branch}/current.json"
    private = deepcopy(source["documents"]["artifact_pointer"])
    private.update(
        {
            "git_commit_sha": commit,
            "model_artifact_hash": model_hash,
            "manifest_uri": f"s3://{bucket}/releases/{commit}/manifest.json",
            "signature_ref": artifact_signature_ref,
            "build_id": f"build-{commit[:8]}",
        }
    )
    private["model_release_identity"] = {
        "schema_version": "model-release-identity:v1",
        "source_commit": commit,
        "model_artifact_digest": model_hash.removeprefix("sha256:"),
        "dependency_lock_sha256": "3" * 64,
        "runtime_base_image_digest": "model-base@sha256:" + "4" * 64,
        "consumer_contract_sha256": source["documents"]["artifact_lineage"][
            "routing_contract_hash"
        ].removeprefix("sha256:"),
        "catalog_sha256": source["documents"]["artifact_lineage"][
            "routing_catalog_hash"
        ].removeprefix("sha256:"),
        "policy_sha256": source["documents"]["artifact_lineage"][
            "routing_policy_hash"
        ].removeprefix("sha256:"),
        "candidate_profiles_sha256": "8" * 64,
        "intent_profiles_sha256": "9" * 64,
        "feature_schema_sha256": source["documents"]["artifact_lineage"][
            "feature_schema_hash"
        ].removeprefix("sha256:"),
        "candidate_waterfall_contract_sha256": "b" * 64,
        "verifier_artifact_digest": "verifier@sha256:" + "c" * 64,
        "tool_binding_manifest_sha256": "d" * 64,
        "llm_configuration_sha256": "e" * 64,
        "release_identity_sha256": "f" * 64,
    }
    private = _signed(
        {name: value for name, value in private.items() if name not in {"manifest_hash", "signature_ref"}},
        key,
        artifact_signature_ref,
    )
    lineage = deepcopy(source["documents"]["artifact_lineage"])
    lineage.update(
        {
            "manifest_uri": f"s3://{bucket}/releases/{commit}/routing-lineage.json",
            "pointer_uri": pointer_uri,
            "pointer_document_hash": sha256_json(private),
            "private_manifest_hash": private["manifest_hash"],
            "model_artifact_hash": model_hash,
            "commit_sha": commit,
            "branch": branch,
            "image_digest": "111111111111.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "2" * 64,
            "build_id": private["build_id"],
            "signature_ref": lineage_signature_ref,
        }
    )
    lineage = _signed(
        {name: value for name, value in lineage.items() if name not in {"manifest_hash", "signature_ref"}},
        key,
        lineage_signature_ref,
    )
    return {
        "pointer_uri": pointer_uri,
        "lineage_manifest_uri": lineage["manifest_uri"],
        "documents": {
            "artifact_pointer": private,
            "artifact_manifest": dict(private),
            "artifact_lineage": lineage,
        },
        "signatures": {
            "artifact_pointer": {
                "key_id": "unused",
                "signature": "unused",
            },
            "artifact_manifest": {
                "key_id": "unused",
                "signature": "unused",
            },
            "artifact_lineage": {
                "key_id": "unused",
                "signature": "unused",
            },
        },
    }


def _dual_bundle() -> tuple[dict, dict[str, str]]:
    source, source_pins = _bundle()
    baseline_key, baseline_public = _new_key()
    challenger_key, challenger_public = _new_key()
    baseline = _artifact_documents(
        source,
        bucket="private-model-baseline",
        commit="b" * 40,
        model_hash="sha256:" + "b" * 64,
        key=baseline_key,
        branch="main",
        artifact_signature_ref="s3://private-model-baseline/signatures/model.sig",
        lineage_signature_ref="s3://private-model-baseline/signatures/lineage.sig",
    )
    challenger = _artifact_documents(
        source,
        bucket="private-model-challenger",
        commit="c" * 40,
        model_hash="sha256:" + "c" * 64,
        key=challenger_key,
        branch="leadpoet-lab",
        artifact_signature_ref="s3://private-model-challenger/signatures/model.sig",
        lineage_signature_ref="s3://private-model-challenger/signatures/lineage.sig",
    )
    registrations = {
        "baseline": baseline,
        "challenger": challenger,
    }
    for variant, key_id in (("baseline", "baseline-artifact-key"), ("challenger", "challenger-artifact-key")):
        lineage_key_id = variant + "-lineage-key"
        registrations[variant]["key_ids"] = {
            "artifact": key_id,
            "lineage": lineage_key_id,
        }
        private_key = baseline_key if variant == "baseline" else challenger_key
        registration = registrations[variant]
        for name in registration["documents"]:
            registration["signatures"][name] = {
                "key_id": key_id if name != "artifact_lineage" else lineage_key_id,
                "signature": base64.b64encode(
                    private_key.sign(
                        registration["documents"][name]["manifest_hash"].encode(),
                        ec.ECDSA(hashes.SHA256()),
                    )
                ).decode(),
            }

    common_key_ids = {
        "binding_catalog": source["key_ids"]["binding_catalog"],
        "unit_dataset": source["key_ids"]["unit_dataset"],
    }
    bundle = {
        "schema_version": "leadpoet.routing_authority_bundle.v2",
        "artifact_registrations": registrations,
        "key_ids": common_key_ids,
        "verification_keys": {
            common_key_ids["binding_catalog"]: source["verification_keys"][common_key_ids["binding_catalog"]],
            common_key_ids["unit_dataset"]: source["verification_keys"][common_key_ids["unit_dataset"]],
            "baseline-artifact-key": baseline_public,
            "baseline-lineage-key": baseline_public,
            "challenger-artifact-key": challenger_public,
            "challenger-lineage-key": challenger_public,
        },
        "documents": {
            "binding_catalog": source["documents"]["binding_catalog"],
            "unit_dataset": source["documents"]["unit_dataset"],
        },
        "signatures": {
            "binding_catalog": source["signatures"]["binding_catalog"],
            "unit_dataset": source["signatures"]["unit_dataset"],
        },
    }
    pins = {
        "binding_catalog": source_pins["binding_catalog"],
        "unit_dataset": source_pins["unit_dataset"],
        "baseline_artifact": baseline_public,
        "baseline_lineage": baseline_public,
        "challenger_artifact": challenger_public,
        "challenger_lineage": challenger_public,
    }
    return bundle, pins


def test_dual_bundle_reconstructs_main_and_lab_lineages():
    bundle, pins = _dual_bundle()
    result = load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)
    assert [item.branch for item in result.artifact_lineages] == ["main", "leadpoet-lab"]
    assert len({item.identity_hash() for item in result.artifact_lineages}) == 2


def test_dual_bundle_rejects_wrong_challenger_branch_and_duplicate_identity():
    bundle, pins = _dual_bundle()
    bundle["artifact_registrations"]["challenger"]["documents"]["artifact_lineage"]["branch"] = "main"
    lineage = bundle["artifact_registrations"]["challenger"]["documents"]["artifact_lineage"]
    lineage["manifest_hash"] = sha256_json({key: value for key, value in lineage.items() if key != "manifest_hash"})
    with pytest.raises(RoutingAuthorityBundleError):
        load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)
