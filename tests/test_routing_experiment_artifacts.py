from __future__ import annotations

import pytest

from gateway.research_lab.routing_experiment_artifacts import (
    ROUTING_ARTIFACT_LINEAGE_SCHEMA_VERSION,
    RoutingArtifactAuthorityError,
    SignedRoutingArtifactAuthority,
    SignedRoutingGoldLabelLoader,
)
from research_lab.canonical import sha256_json


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _documents(*, lineage_override=None):
    pointer_uri = "s3://private-model/releases/branches/leadpoet-lab/current.json"
    immutable_uri = "s3://private-model/releases/commits/abc/model-manifest.json"
    lineage_uri = "s3://private-model/releases/commits/abc/routing-lineage.json"
    private_payload = {
        "model_artifact_hash": _hash("1"),
        "git_commit_sha": "a" * 40,
        "image_digest": "111111111111.dkr.ecr.us-east-1.amazonaws.com/model@" + _hash("2"),
        "config_hash": _hash("3"),
        "component_registry_version": "registry-1",
        "scoring_adapter_version": "adapter-1",
        "manifest_uri": immutable_uri,
        "signature_ref": "s3://private-model/signatures/model.sig",
        "build_id": "build-1",
        "compatibility_contract": {
            "contract_id": "leadpoet-sourcing-model-consumer:v52",
            "path": "sourcing_model/consumer_contract.json",
            "sha256": _hash("4"),
        },
        "consumer_parity_fixtures": {
            "path": "sourcing_model/consumer_parity_fixtures.json",
            "sha256": _hash("5"),
        },
    }
    private_document = {
        **private_payload,
        "manifest_hash": sha256_json(private_payload),
    }
    lineage_payload = {
        "schema_version": ROUTING_ARTIFACT_LINEAGE_SCHEMA_VERSION,
        "manifest_uri": lineage_uri,
        "repository": "leadpoet/Sourcing_model",
        "branch": "leadpoet-lab",
        "pointer_uri": pointer_uri,
        "pointer_document_hash": sha256_json(private_document),
        "private_manifest_hash": private_document["manifest_hash"],
        "model_artifact_hash": private_document["model_artifact_hash"],
        "commit_sha": private_document["git_commit_sha"],
        "image_digest": private_document["image_digest"],
        "build_id": private_document["build_id"],
        "routing_contract_hash": _hash("6"),
        "routing_catalog_hash": _hash("7"),
        "routing_policy_hash": _hash("8"),
        "feature_schema_hash": _hash("9"),
        "verifier_contract_hash": _hash("a"),
        "signature_ref": "s3://private-model/signatures/routing-lineage.sig",
    }
    if lineage_override:
        lineage_payload.update(lineage_override)
    lineage_document = {
        **lineage_payload,
        "manifest_hash": sha256_json(lineage_payload),
    }
    return pointer_uri, immutable_uri, lineage_uri, private_document, lineage_document


def _private_verifier(manifest, *, key_id):
    return {
        "verified": True,
        "manifest_hash": manifest.manifest_hash,
        "signature_ref": manifest.signature_ref,
        "key_id": key_id,
        "signing_algorithm": "ECDSA_SHA_256",
        "consumer_contract_binding_mode": "semantic_v1_required",
    }


def _lineage_verifier(document, key_id):
    return {
        "verified": True,
        "manifest_hash": document["manifest_hash"],
        "signature_ref": document["signature_ref"],
        "key_id": key_id,
        "signing_algorithm": "ECDSA_SHA_256",
    }


def test_signed_current_pointer_and_additional_routing_lineage_are_both_required():
    pointer_uri, immutable_uri, lineage_uri, private, lineage = _documents()
    documents = {pointer_uri: private, immutable_uri: private, lineage_uri: lineage}
    authority = SignedRoutingArtifactAuthority(
        pointer_uri=pointer_uri,
        lineage_manifest_uri=lineage_uri,
        loader=documents.__getitem__,
        verifier=_private_verifier,
        key_id="kms-private-model",
        lineage_verifier=_lineage_verifier,
        lineage_key_id="kms-routing-lineage",
    )
    resolved = authority.resolve()
    assert resolved.pointer_document_hash == sha256_json(private)
    assert resolved.routing_lineage_manifest_hash == lineage["manifest_hash"]
    assert resolved.routing_catalog_hash == _hash("7")
    assert resolved.image_digest == private["image_digest"]
    verified = authority.verify(
        artifact=resolved.sourcing_model_identity(),
        manifest=private,
    )
    assert verified["artifact_lineage_hash"] == resolved.identity_hash()


@pytest.mark.parametrize(
    "override",
    [
        {"pointer_document_hash": "sha256:" + "0" * 64},
        {"model_artifact_hash": "sha256:" + "0" * 64},
        {"commit_sha": "0" * 40},
        {"routing_policy_hash": "not-a-hash"},
    ],
)
def test_routing_lineage_identity_tampering_fails_closed(override):
    pointer_uri, immutable_uri, lineage_uri, private, lineage = _documents(
        lineage_override=override
    )
    authority = SignedRoutingArtifactAuthority(
        pointer_uri=pointer_uri,
        lineage_manifest_uri=lineage_uri,
        loader={pointer_uri: private, immutable_uri: private, lineage_uri: lineage}.__getitem__,
        verifier=_private_verifier,
        key_id="kms-private-model",
        lineage_verifier=_lineage_verifier,
        lineage_key_id="kms-routing-lineage",
    )
    with pytest.raises(RoutingArtifactAuthorityError):
        authority.resolve()


def test_caller_manifest_cannot_replace_the_signed_current_pointer():
    pointer_uri, immutable_uri, lineage_uri, private, lineage = _documents()
    authority = SignedRoutingArtifactAuthority(
        pointer_uri=pointer_uri,
        lineage_manifest_uri=lineage_uri,
        loader={pointer_uri: private, immutable_uri: private, lineage_uri: lineage}.__getitem__,
        verifier=_private_verifier,
        key_id="kms-private-model",
        lineage_verifier=_lineage_verifier,
        lineage_key_id="kms-routing-lineage",
    )
    resolved = authority.resolve()
    with pytest.raises(RoutingArtifactAuthorityError, match="differs"):
        authority.verify(
            artifact=resolved.sourcing_model_identity(),
            manifest={**private, "build_id": "caller-build"},
        )


def test_verify_uses_the_exact_verified_pointer_snapshot_without_unsigned_reload():
    pointer_uri, immutable_uri, lineage_uri, private, lineage = _documents()
    documents = {pointer_uri: private, immutable_uri: private, lineage_uri: lineage}
    calls = []

    def loader(uri):
        calls.append(uri)
        return documents[uri]

    authority = SignedRoutingArtifactAuthority(
        pointer_uri=pointer_uri,
        lineage_manifest_uri=lineage_uri,
        loader=loader,
        verifier=_private_verifier,
        key_id="kms-private-model",
        lineage_verifier=_lineage_verifier,
        lineage_key_id="kms-routing-lineage",
    )
    resolved = authority.resolve()
    documents[pointer_uri] = {**private, "build_id": "mutable-pointer"}
    assert authority.verify(
        artifact=resolved.sourcing_model_identity(),
        manifest=private,
    )["artifact_lineage_hash"] == resolved.identity_hash()
    assert calls.count(pointer_uri) == 1


@pytest.mark.parametrize(
    "uri",
    (
        "s3://lab-routing/labels/current.json",
        "s3://lab-routing/labels/branches/leadpoet-lab/labels.json",
    ),
)
def test_gold_label_authority_rejects_mutable_manifest_uri(uri):
    with pytest.raises(RoutingArtifactAuthorityError, match="immutable"):
        SignedRoutingGoldLabelLoader(
            manifest_uri=uri,
            loader=lambda _uri: {},
            verifier=lambda _document, _key: {},
            key_id="kms-labels",
        )
