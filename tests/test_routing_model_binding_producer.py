from __future__ import annotations

import base64
from dataclasses import replace
import json

import pytest

from gateway.research_lab.routing_experiment_artifacts import (
    RoutingArtifactAuthorityError,
    VerifiedRoutingArtifactLineage,
)
from gateway.research_lab.routing_model_binding_observation import (
    ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
    VerifiedRoutingModelBindingRequirements,
)
from gateway.research_lab.routing_model_binding_producer import (
    MeasuredModelMetadataExecutionV2,
    RoutingModelBindingObservationProducerV2,
    RoutingModelBindingProducerError,
    resolve_verified_routing_artifact_lineage_v2,
)
from gateway.tee.scoring_executor_v2 import (
    OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2,
    ScoringExecutorV2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
from leadpoet_canonical.attested_v2 import (
    build_receipt_graph,
    sha256_bytes,
    validate_receipt_graph,
)
from research_lab.canonical import sha256_json
from research_lab.eval import PrivateModelArtifactManifest

from tests.test_routing_model_binding_observation import _binding_and_row, _metadata
from tests.test_execution_job_manager_v2 import _manager, _manifest, _run
from tests.v2_epoch_test_utils import epoch_test_environment


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _artifact_and_lineage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("# exact model\n", encoding="utf-8")
    source_bundle = build_source_bundle_v2(source)
    image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/sourcing-model@sha256:" + "1" * 64
    draft = PrivateModelArtifactManifest(
        model_artifact_hash=source_bundle["source_tree_hash"],
        git_commit_sha="a" * 40,
        image_digest=image,
        config_hash=_hash("2"),
        component_registry_version="component-registry-v2",
        scoring_adapter_version="scoring-adapter-v2",
        manifest_uri="s3://models/immutable/model.json",
        manifest_hash="",
        signature_ref="kms://models/model",
        build_id="build-1",
    )
    artifact = replace(draft, manifest_hash=sha256_json(draft.hash_payload()))
    lineage = VerifiedRoutingArtifactLineage(
        repository="leadpoet/Sourcing_model",
        branch="leadpoet-lab",
        commit_sha=artifact.git_commit_sha,
        pointer_uri="s3://models/branches/leadpoet-lab/current.json",
        pointer_document_hash=_hash("3"),
        immutable_manifest_uri=artifact.manifest_uri,
        routing_lineage_manifest_uri="s3://models/lineage/lineage-1.json",
        routing_lineage_manifest_hash=_hash("4"),
        manifest_hash=artifact.manifest_hash,
        signature_ref=artifact.signature_ref,
        signature_key_id="kms-model-key",
        signature_algorithm="ECDSA_SHA_256",
        model_artifact_hash=artifact.model_artifact_hash,
        image_digest=artifact.image_digest,
        config_hash=artifact.config_hash,
        build_id=artifact.build_id,
        component_registry_version=artifact.component_registry_version,
        scoring_adapter_version=artifact.scoring_adapter_version,
        routing_contract_hash=_hash("5"),
        routing_catalog_hash=_hash("6"),
        routing_policy_hash=_hash("7"),
        feature_schema_hash=_hash("8"),
        verifier_contract_hash=_hash("9"),
    )
    return artifact, lineage, source_bundle


class _FakeSignedArtifactAuthority:
    """Small signed-authority double; the returned object is the trust root."""

    def __init__(self, lineage, artifact_document):
        self.lineage = lineage
        self.artifact_document = dict(artifact_document)

    def resolve(self):
        return self.lineage

    def verify(self, *, artifact, manifest):
        if artifact != self.lineage.sourcing_model_identity():
            raise RoutingArtifactAuthorityError("artifact identity differs")
        if dict(manifest) != self.artifact_document:
            raise RoutingArtifactAuthorityError("immutable manifest differs")
        return {
            "verified": True,
            "artifact_lineage_hash": self.lineage.identity_hash(),
        }


def test_lineage_resolver_returns_only_signed_authority_identity(tmp_path):
    artifact, lineage, _source_bundle = _artifact_and_lineage(tmp_path)
    resolved = resolve_verified_routing_artifact_lineage_v2(
        lineage_document=lineage.to_dict(),
        artifact_document=artifact.to_dict(),
        authority=_FakeSignedArtifactAuthority(lineage, artifact.to_dict()),
    )
    assert resolved is lineage
    assert resolved.identity_hash() == lineage.identity_hash()


def test_lineage_resolver_rejects_fabricated_caller_lineage(tmp_path):
    artifact, lineage, _source_bundle = _artifact_and_lineage(tmp_path)
    fabricated = replace(lineage, commit_sha="b" * 40)
    with pytest.raises(RoutingModelBindingProducerError, match="differs"):
        resolve_verified_routing_artifact_lineage_v2(
            lineage_document=fabricated.to_dict(),
            artifact_document=artifact.to_dict(),
            authority=_FakeSignedArtifactAuthority(lineage, artifact.to_dict()),
        )


def test_lineage_resolver_rejects_pointer_immutable_mismatch(tmp_path):
    artifact, lineage, _source_bundle = _artifact_and_lineage(tmp_path)

    class _MismatchedAuthority(_FakeSignedArtifactAuthority):
        def resolve(self):
            raise RoutingArtifactAuthorityError(
                "routing pointer and immutable manifest differ"
            )

    with pytest.raises(RoutingModelBindingProducerError, match="authority"):
        resolve_verified_routing_artifact_lineage_v2(
            lineage_document=lineage.to_dict(),
            artifact_document=artifact.to_dict(),
            authority=_MismatchedAuthority(lineage, artifact.to_dict()),
        )


def test_lineage_resolver_rejects_substituted_immutable_manifest(tmp_path):
    artifact, lineage, _source_bundle = _artifact_and_lineage(tmp_path)
    substituted = dict(artifact.to_dict())
    substituted["build_id"] = "attacker-build"
    with pytest.raises(RoutingModelBindingProducerError, match="authority"):
        resolve_verified_routing_artifact_lineage_v2(
            lineage_document=lineage.to_dict(),
            artifact_document=substituted,
            authority=_FakeSignedArtifactAuthority(lineage, artifact.to_dict()),
        )


def _fake_execution(*, payload, source_bundle, artifact, row, calls):
    calls.append(payload)
    empty_hash = sha256_json({})
    output = {"runtime_routing": _metadata(row=row)}
    result = {
        "schema_version": "leadpoet.model_sandbox_result.v2",
        "model_kind": "private",
        "operation": "metadata",
        "model_artifact_hash": artifact.model_artifact_hash,
        "model_manifest_hash": artifact.manifest_hash,
        "compatibility_image_digest": artifact.image_digest,
        "source_bundle_hash": source_bundle["archive_sha256"],
        "compatibility_policy_hash": _hash("1"),
        "compatibility_admission_hash": _hash("2"),
        "runtime_config_hash": _hash("3"),
        "input_hash": sha256_json(payload["input"]),
        "provider_evidence_cache_hash": empty_hash,
        "provider_evidence_cache_ref": "",
        "provider_evidence_mode": "",
        "provider_snapshot_archive_hash": empty_hash,
        "provider_snapshot_tree_hash": empty_hash,
        "provider_snapshot_manifest_hash": empty_hash,
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog_hash": empty_hash,
        "generated_provider_evidence_cache_hash": empty_hash,
        "trace_entries_hash": sha256_json([]),
        "output_hash": sha256_json(output),
        "output": output,
        "trace_entries": [],
        "generated_provider_evidence_cache": {},
        "consumer_runtime_probe": {"measured": True},
        "consumer_runtime_probe_hash": sha256_json({"measured": True}),
    }
    return MeasuredModelMetadataExecutionV2(
        payload=payload,
        result=result,
    )


def _producer_fixture(tmp_path):
    artifact, lineage, source_bundle = _artifact_and_lineage(tmp_path)
    binding, row = _binding_and_row()
    calls = []

    def execute(*, payload, job_id, purpose):
        assert job_id == "routing-observation-1"
        assert purpose == ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2
        return _fake_execution(
            payload=payload,
            source_bundle=source_bundle,
            artifact=artifact,
            row=row,
            calls=calls,
        )
    stages = []

    return (
        RoutingModelBindingObservationProducerV2(
            measured_metadata_executor=execute,
            record_stage=lambda **spec: stages.append(spec),
        ),
        artifact,
        lineage,
        source_bundle,
        binding,
        row,
        calls,
        stages,
    )


def test_producer_binds_measured_metadata_and_registers_standard_stage(tmp_path):
    producer, artifact, lineage, source_bundle, binding, _row, calls, stages = _producer_fixture(tmp_path)
    output = producer.produce(
        artifact_lineage=lineage,
        artifact_document=artifact.to_dict(),
        source_bundle=source_bundle,
        provider_bindings=(binding,),
        job_id="routing-observation-1",
    )
    assert output["schema_version"] == "leadpoet.routing_model_binding_producer_result.v2"
    assert output["artifact_lineage_hash"] == lineage.identity_hash()
    assert output["observation"]["artifact_lineage_hash"] == lineage.identity_hash()
    assert len(stages) == 1
    assert stages[0]["purpose"] == ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2
    assert stages[0]["output_root"] == sha256_json(output["observation"])
    assert stages[0]["input_root"] == output["observation"]["request_root"]
    assert len(calls) == 1
    assert calls[0]["operation"] == "metadata"
    assert calls[0]["provider_runtime_catalog"] == {}
    assert calls[0]["provider_evidence_cache"] == {}


def test_producer_rejects_fabricated_lineage_artifact_source_and_binding_sets(tmp_path):
    producer, artifact, lineage, source_bundle, binding, row, calls, _stages = _producer_fixture(tmp_path)
    with pytest.raises(RoutingModelBindingProducerError):
        producer.produce(
            artifact_lineage=replace(lineage, model_artifact_hash=_hash("f")),
            artifact_document=artifact.to_dict(),
            source_bundle=source_bundle,
            provider_bindings=(binding,),
            job_id="routing-observation-1",
        )
    bad_bundle = dict(source_bundle)
    bad_bundle["source_tree_hash"] = _hash("e")
    with pytest.raises(RoutingModelBindingProducerError):
        producer.produce(
            artifact_lineage=lineage,
            artifact_document=artifact.to_dict(),
            source_bundle=bad_bundle,
            provider_bindings=(binding,),
            job_id="routing-observation-1",
        )
    _extra, _extra_row = _binding_and_row(provider_id="sumble")
    with pytest.raises(RoutingModelBindingProducerError):
        producer.produce(
            artifact_lineage=lineage,
            artifact_document=artifact.to_dict(),
            source_bundle=source_bundle,
            provider_bindings=(binding, _extra),
            job_id="routing-observation-1",
        )
    # The extra binding is rejected after the measured metadata operation
    # returns.  That operation has an empty provider catalog and trace; no
    # provider call can occur on this path.
    assert len(calls) == 1
    assert calls[0]["provider_runtime_catalog"] == {}


@pytest.mark.asyncio
async def test_scoring_executor_runs_measured_metadata_and_records_stage(tmp_path, monkeypatch):
    for name, value in epoch_test_environment().items():
        monkeypatch.setenv(name, value)
    artifact, lineage, source_bundle = _artifact_and_lineage(tmp_path)
    binding, row = _binding_and_row()
    sandbox_calls = []

    class _Sandbox:
        def execute(self, payload, **kwargs):
            sandbox_calls.append((dict(payload), dict(kwargs)))
            return _fake_execution(
                payload=payload,
                source_bundle=source_bundle,
                artifact=artifact,
                row=row,
                calls=[],
            ).result

    context = ExecutionContextV2(
        job_id="routing-operation-1",
        purpose="research_lab.routing_model_binding_observation.v2",
        epoch_id=24_301,
        allowed_purposes=frozenset(
            {"research_lab.routing_model_binding_observation.v2"}
        ),
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail(
            "routing model observation must not call a provider"
        ),
        retry_policy_hashes={},
        model_sandbox=_Sandbox(),
        routing_artifact_lineage_resolver=lambda **_kwargs: lineage,
    )
    try:
        result = await executor(
            OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2,
            {
                "schema_version": "leadpoet.routing_model_binding_request.v2",
                "model_kind": "private",
                "artifact_lineage": lineage.to_dict(),
                "artifact": artifact.to_dict(),
                "source_bundle": source_bundle,
                "provider_bindings": [binding.to_dict()],
            },
            context,
        )
    finally:
        executor.close()
    assert result.output["artifact_lineage_hash"] == lineage.identity_hash()
    assert result.output["observation"]["artifact_lineage_hash"] == lineage.identity_hash()
    assert len(context.stage_receipts) == 1
    assert context.stage_receipts[0].purpose == (
        "research_lab.routing_model_binding_observation.v2"
    )
    assert context.transport_attempts == []
    assert len(sandbox_calls) == 1
    assert sandbox_calls[0][0]["provider_runtime_catalog"] == {}
    assert sandbox_calls[0][0]["provider_evidence_cache"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["artifact", "result", "trace", "binding"])
async def test_scoring_executor_rejects_substitution_before_any_provider_call(
    tmp_path, mutation, monkeypatch
):
    for name, value in epoch_test_environment().items():
        monkeypatch.setenv(name, value)
    artifact, lineage, source_bundle = _artifact_and_lineage(tmp_path)
    binding, row = _binding_and_row()
    sandbox_calls = []

    class _Sandbox:
        def execute(self, payload, **kwargs):
            sandbox_calls.append(dict(payload))
            execution = _fake_execution(
                payload=payload,
                source_bundle=source_bundle,
                artifact=artifact,
                row=row,
                calls=[],
            )
            if mutation == "result":
                execution.result["model_manifest_hash"] = _hash("f")
            elif mutation == "trace":
                execution.result["trace_entries"] = [{"provider": "forbidden"}]
            return execution.result

    context = ExecutionContextV2(
        job_id="routing-operation-substitution",
        purpose="research_lab.routing_model_binding_observation.v2",
        epoch_id=24_301,
        allowed_purposes=frozenset(
            {"research_lab.routing_model_binding_observation.v2"}
        ),
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider call is forbidden"),
        retry_policy_hashes={},
        model_sandbox=_Sandbox(),
        routing_artifact_lineage_resolver=lambda **_kwargs: lineage,
    )
    payload = {
        "schema_version": "leadpoet.routing_model_binding_request.v2",
        "model_kind": "private",
        "artifact_lineage": lineage.to_dict(),
        "artifact": artifact.to_dict(),
        "source_bundle": source_bundle,
        "provider_bindings": [
            (replace(binding, manifest_hash=_hash("f")).to_dict())
            if mutation == "binding"
            else binding.to_dict()
        ],
    }
    if mutation == "artifact":
        payload["artifact"] = {**artifact.to_dict(), "model_artifact_hash": _hash("f")}
    try:
        with pytest.raises(ValueError):
            await executor(OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2, payload, context)
    finally:
        executor.close()
    assert context.transport_attempts == []
    assert len(sandbox_calls) == (0 if mutation == "artifact" else 1)


def test_execution_job_manager_signs_synthetic_observation_stage(tmp_path, monkeypatch):
    for name, value in epoch_test_environment().items():
        monkeypatch.setenv(name, value)
    artifact, lineage, source_bundle = _artifact_and_lineage(tmp_path)
    binding, row = _binding_and_row()

    class _Sandbox:
        def execute(self, payload, **kwargs):
            return _fake_execution(
                payload=payload,
                source_bundle=source_bundle,
                artifact=artifact,
                row=row,
                calls=[],
            ).result

    scorer = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider call is forbidden"),
        retry_policy_hashes={},
        model_sandbox=_Sandbox(),
        routing_artifact_lineage_resolver=lambda **_kwargs: lineage,
    )

    def execute(operation, payload, context):
        return scorer(operation, payload, context)

    operations = {
        OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2: {
            "research_lab.routing_model_binding_observation.v2"
        }
    }
    manager, boot = _manager(execute, operations=operations)
    payload = {
        "schema_version": "leadpoet.routing_model_binding_request.v2",
        "model_kind": "private",
        "artifact_lineage": lineage.to_dict(),
        "artifact": artifact.to_dict(),
        "source_bundle": source_bundle,
        "provider_bindings": [binding.to_dict()],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    try:
        status = _run(
            manager,
            encoded,
            _manifest(
                encoded,
                operation=OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2,
                purpose="research_lab.routing_model_binding_observation.v2",
                job_id="routing-manager-job",
                input_artifact_hashes=[],
            ),
        )
        assert status["state"] == "succeeded"
        receipts = manager.receipts("routing-manager-job")
        assert len(receipts) == 2
        stage, final = receipts
        assert stage["purpose"] == "research_lab.routing_model_binding_observation.v2"
        assert stage["job_id"].startswith("stage:")
        assert final["parent_receipt_hashes"] == [stage["receipt_hash"]]
        assert final["input_root"] == sha256_bytes(encoded)
        assert manager.transport_attempts("routing-manager-job") == ()
        result_chunk = manager.result_chunk(job_id="routing-manager-job")
        result = json.loads(base64.b64decode(result_chunk["data_b64"]))
        assert final["output_root"] == sha256_json(result)
        verified = VerifiedRoutingModelBindingRequirements.from_attested(
            result["observation"], stage
        )
        assert verified.artifact_lineage_hash == lineage.identity_hash()
        graph = build_receipt_graph(
            root_receipt_hash=final["receipt_hash"],
            boot_identities=(boot,),
            receipts=receipts,
            transport_attempts=manager.transport_attempts("routing-manager-job"),
        )
        assert validate_receipt_graph(graph) == (
            stage["receipt_hash"],
            final["receipt_hash"],
        )
    finally:
        scorer.close()
