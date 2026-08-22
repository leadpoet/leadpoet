from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path

import pytest

from gateway.research_lab.routing_release_builder import (
    RoutingReleaseDependencyError,
    ReviewedRoutingReleaseAuthoritySources,
    build_reviewed_routing_release_dependencies,
    render_generated_release_module,
)
from gateway.tee.protected_workflows import load_manifest
from research_lab.canonical import sha256_json
from tests.test_routing_authority_bundle import _bundle
from tests.test_routing_product_composition import (
    _ModelRunnerRegistry,
    _ModelVerifier,
    _EvaluationAdapter,
    _ArtifactAuthority,
    _Rpc,
    _PROTECTED_RELEASE_RECEIPT,
    _env,
)
from tests.routing_experiment_authority_fixture import authority_fixture


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_MANIFEST = ROOT / "gateway" / "tee" / "protected_workflows.json"


class _ReadyRunnerFactory:
    def validate_readiness(self):
        return None

    def __call__(self, _spec):
        raise AssertionError("the release builder must not run a provider runner")


def _gold_document(unit_ref: str) -> dict:
    labels = {unit_ref: True}
    payload = {
        "schema_version": "leadpoet.routing_gold_labels.v1",
        "labels": labels,
        "label_set_hash": sha256_json({"labels": [[unit_ref, True]]}),
        "provenance_hash": "sha256:" + "9" * 64,
        "manifest_uri": "s3://routing-labels/releases/labels-1.json",
        "signature_ref": "s3://routing-labels/signatures/labels-1.sig",
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def _sources_and_environment(*, tamper_bundle: bool = False):
    bundle, pins = _bundle()
    from gateway.research_lab.routing_authority_bundle import (
        load_verified_routing_authority_bundle,
    )

    verified = load_verified_routing_authority_bundle(bundle, pinned_public_keys=pins)
    if tamper_bundle:
        bundle["signatures"]["unit_dataset"]["signature"] = base64.b64encode(
            hashlib.sha256(b"tampered").digest()
        ).decode()
    unit_ref = tuple(verified.unit_dataset.units)[0]
    gold = _gold_document(unit_ref)
    protected_manifest_hash = load_manifest(PROTECTED_MANIFEST)["manifest_hash"]
    fixture = authority_fixture()
    source = ReviewedRoutingReleaseAuthoritySources(
        authority_bundle_document=bundle,
        authority_bundle_pinned_public_keys=pins,
        gold_label_document=gold,
        gold_label_key_id="gold-label-key",
        gold_label_verifier=lambda document, key_id: {
            "verified": True,
            "manifest_hash": document["manifest_hash"],
            "signature_ref": document["signature_ref"],
            "key_id": key_id,
            "signing_algorithm": "ECDSA_SHA_256",
        },
        expected_label_set_hash=gold["label_set_hash"],
        expected_unit_refs=(unit_ref,),
        model_binding_observation=fixture["model_binding_observation"],
        protected_release_receipt=dict(_PROTECTED_RELEASE_RECEIPT),
        artifact_authority=_ArtifactAuthority(),
        model_runner_registry=_ModelRunnerRegistry(),
        model_verifier=_ModelVerifier(),
        evaluation_adapter=_EvaluationAdapter(),
        scoring_job_rpc=_Rpc(),
        call_authorization_job_rpc=_Rpc(),
        dispatch_job_rpc=_Rpc(),
        reviewed_runner_factory=_ReadyRunnerFactory(),
        billing_rollup_factory=lambda _spec: lambda _store: {},
        execution_envelope_factory=lambda _spec: None,
        store_factory=lambda: object(),
        protected_workflow_manifest_hash=protected_manifest_hash,
    )
    environment = dict(_env())
    environment.update(
        {
            "RESEARCH_LAB_ROUTING_MODEL_COMMIT_SHA": verified.artifact_lineage.commit_sha,
            "RESEARCH_LAB_ROUTING_MODEL_CATALOG_HASH": verified.artifact_lineage.routing_catalog_hash,
            "RESEARCH_LAB_ROUTING_BINDING_CATALOG_MANIFEST_HASH": verified.binding_catalog.manifest_hash,
            "RESEARCH_LAB_ROUTING_CONTRACT_HASH": verified.artifact_lineage.routing_contract_hash,
            "RESEARCH_LAB_ROUTING_AUTHORITY_BUNDLE_HASH": verified.bundle_hash,
        }
    )
    return source, environment, protected_manifest_hash


def test_release_builder_constructs_typed_dependencies_from_signed_inputs(monkeypatch):
    sources, environment, protected_hash = _sources_and_environment()
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    dependencies = build_reviewed_routing_release_dependencies(
        sources,
        environment=environment,
        expected_protected_workflow_manifest_hash=protected_hash,
    )
    assert dependencies.inputs.authority_bundle.bundle_hash == environment[
        "RESEARCH_LAB_ROUTING_AUTHORITY_BUNDLE_HASH"
    ]
    assert dependencies.inputs.gold_labels.labels == {"company-1": True}
    assert dependencies.inputs.dispatch_job_rpc.__class__.__name__ == (
        "RoutingProviderDispatchTeeRpc"
    )


def test_generated_release_module_loads_without_sys_modules_injection(
    tmp_path: Path, monkeypatch
):
    sources, environment, protected_hash = _sources_and_environment()
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    provider_dir = tmp_path / "release_provider"
    provider_dir.mkdir()
    (provider_dir / "attested_routing_release_authorities.py").write_text(
        "import os\n"
        "from tests.test_routing_release_builder import _sources_and_environment\n"
        "def load_reviewed_routing_release_authority_sources():\n"
        "    sources, environment, _ = _sources_and_environment()\n"
        "    os.environ.update(environment)\n"
        "    return sources\n",
        encoding="utf-8",
    )
    import gateway.research_lab as research_lab_package

    original_path = list(research_lab_package.__path__)
    research_lab_package.__path__.insert(0, str(provider_dir))
    generated = tmp_path / "routing_release_dependencies.py"
    generated.write_text(
        render_generated_release_module(
            protected_workflow_manifest_hash=protected_hash
        ),
        encoding="utf-8",
    )
    try:
        specification = importlib.util.spec_from_file_location(
            "generated_routing_release_dependencies", generated
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        dependencies = module.load_reviewed_routing_release_dependencies()
    finally:
        research_lab_package.__path__[:] = original_path
    assert dependencies.inputs.gold_labels.labels == {"company-1": True}


def test_release_builder_rejects_tampered_signed_bundle(monkeypatch):
    sources, environment, protected_hash = _sources_and_environment(
        tamper_bundle=True
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RoutingReleaseDependencyError, match="signed authority"):
        build_reviewed_routing_release_dependencies(
            sources,
            environment=environment,
            expected_protected_workflow_manifest_hash=protected_hash,
        )


def test_generated_release_module_rejects_protected_manifest_drift(
    tmp_path: Path, monkeypatch
):
    sources, environment, protected_hash = _sources_and_environment()
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    generated = tmp_path / "routing_release_dependencies.py"
    generated.write_text(
        render_generated_release_module(
            protected_workflow_manifest_hash="sha256:" + "0" * 64
        ),
        encoding="utf-8",
    )
    specification = importlib.util.spec_from_file_location(
        "generated_routing_release_dependencies_drift", generated
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with pytest.raises(RoutingReleaseDependencyError, match="manifest"):
        build_reviewed_routing_release_dependencies(
            sources,
            environment=environment,
            expected_protected_workflow_manifest_hash=(
                module.EXPECTED_PROTECTED_WORKFLOW_MANIFEST_HASH
            ),
        )
