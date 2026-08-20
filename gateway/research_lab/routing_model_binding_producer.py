"""Protected producer for model-owned routing binding observations.

The routing admission code consumes a small, redacted observation.  This
module is the boundary that creates it: the exact private model is executed
through the already measured metadata operation, its signed execution receipt
is checked against the request and complete result, and only then are the
model's runtime bindings compared with the reviewed provider identities.

The executor and stage-receipt issuer are injected by the protected scoring
composition.  This keeps the module independent from the process-local job
manager while making the integration seam explicit.  A caller cannot supply a
runtime metadata document or a hash in place of a measured execution.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.research_lab.routing_experiment_artifacts import (
    RoutingArtifactAuthorityError,
    SignedRoutingArtifactAuthority,
    VerifiedRoutingArtifactLineage,
)
from gateway.research_lab.routing_model_binding_observation import (
    ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
    RoutingModelBindingObservationError,
    observe_routing_model_bindings_v2,
)
from gateway.tee.source_bundle_v2 import (
    SOURCE_BUNDLE_SCHEMA_VERSION,
    extract_source_bundle_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes
from research_lab.canonical import sha256_json
from research_lab.eval import (
    PrivateModelArtifactManifest,
    validate_private_model_artifact_manifest,
)
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    validate_provider_binding_identity,
    validate_sourcing_model_artifact_identity,
)


ROUTING_MODEL_BINDING_OBSERVATION_OPERATION_V2 = (
    "routing_model_binding_observation_v2"
)
ROUTING_MODEL_BINDING_PRODUCER_RESULT_SCHEMA_V2 = (
    "leadpoet.routing_model_binding_producer_result.v2"
)
MODEL_METADATA_MODULE_V2 = "research_lab_adapter"
MODEL_METADATA_CALLABLE_V2 = "adapter_metadata"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_BUNDLE_SCHEMA = SOURCE_BUNDLE_SCHEMA_VERSION

_MODEL_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "model_kind",
        "operation",
        "model_artifact_hash",
        "model_manifest_hash",
        "compatibility_image_digest",
        "source_bundle_hash",
        "compatibility_policy_hash",
        "compatibility_admission_hash",
        "runtime_config_hash",
        "input_hash",
        "provider_evidence_cache_hash",
        "provider_evidence_cache_ref",
        "provider_evidence_mode",
        "provider_snapshot_archive_hash",
        "provider_snapshot_tree_hash",
        "provider_snapshot_manifest_hash",
        "provider_cost_cap_microusd",
        "provider_call_cap",
        "provider_runtime_catalog_hash",
        "generated_provider_evidence_cache_hash",
        "trace_entries_hash",
        "output_hash",
        "output",
        "trace_entries",
        "generated_provider_evidence_cache",
        "consumer_runtime_probe",
        "consumer_runtime_probe_hash",
    }
)


class RoutingModelBindingProducerError(ValueError):
    """A measured model binding observation failed a trust boundary."""


def resolve_verified_routing_artifact_lineage_v2(
    *,
    lineage_document: Mapping[str, Any],
    artifact_document: Mapping[str, Any],
    authority: Any | None = None,
    authority_factory: Callable[[], Any] | None = None,
) -> VerifiedRoutingArtifactLineage:
    """Resolve the signed model pointer before accepting a routing request.

    A ``VerifiedRoutingArtifactLineage`` dataclass is only a value object.  It
    does not prove that its fields came from the signed model pointer.  This
    boundary therefore obtains the lineage from ``SignedRoutingArtifactAuthority``
    (or an injected authority with the same contract), compares the caller's
    documents byte-for-byte at their canonical mapping boundary, and asks the
    authority to verify the complete immutable model manifest.  The returned
    value is the authority's resolved object, never a caller-constructed
    dataclass.
    """

    if authority is not None and authority_factory is not None:
        raise RoutingModelBindingProducerError(
            "routing artifact authority and factory are mutually exclusive"
        )
    if not isinstance(lineage_document, Mapping):
        raise RoutingModelBindingProducerError(
            "routing artifact lineage document is invalid"
        )
    if not isinstance(artifact_document, Mapping):
        raise RoutingModelBindingProducerError(
            "routing artifact manifest document is invalid"
        )
    try:
        resolved_authority = (
            authority_factory()
            if authority_factory is not None
            else authority
            if authority is not None
            else SignedRoutingArtifactAuthority()
        )
        resolve = getattr(resolved_authority, "resolve", None)
        verify = getattr(resolved_authority, "verify", None)
        if not callable(resolve) or not callable(verify):
            raise RoutingModelBindingProducerError(
                "routing artifact authority contract is unavailable"
            )
        resolved = resolve()
        if not isinstance(resolved, VerifiedRoutingArtifactLineage):
            raise RoutingModelBindingProducerError(
                "routing artifact authority returned an unverified lineage"
            )
        if dict(lineage_document) != resolved.to_dict():
            raise RoutingModelBindingProducerError(
                "routing artifact lineage differs from signed authority"
            )
        # ``verify`` compares the complete private model manifest to the
        # authority's signed pointer/immutable document.  Passing the
        # authority's own identity here prevents a caller from substituting a
        # second identity while still reusing the resolved pointer.
        verification = verify(
            artifact=resolved.sourcing_model_identity(),
            manifest=dict(artifact_document),
        )
        if not isinstance(verification, Mapping) or verification.get("verified") is not True:
            raise RoutingModelBindingProducerError(
                "routing artifact manifest was not verified"
            )
        _validate_lineage_and_artifact(resolved, artifact_document)
        return resolved
    except RoutingModelBindingProducerError:
        raise
    except (RoutingArtifactAuthorityError, TypeError, ValueError, KeyError) as exc:
        raise RoutingModelBindingProducerError(
            "routing artifact signed authority verification failed"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - authority failures fail closed
        raise RoutingModelBindingProducerError(
            "routing artifact signed authority verification failed"
        ) from exc


@dataclass(frozen=True)
class MeasuredModelMetadataExecutionV2:
    """The protected executor's complete, non-redacted result boundary.

    ``result`` is the exact output returned by ``ScoringExecutorV2`` for the
    metadata operation. ``payload`` is included so the protected caller can
    prove that the measured request was the one it constructed, rather than
    trusting a caller-provided root.
    """

    payload: Mapping[str, Any]
    result: Mapping[str, Any]


MeasuredModelMetadataExecutorV2 = Callable[..., MeasuredModelMetadataExecutionV2]
StageRecorderV2 = Callable[..., None]


def _hash(value: Any, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise RoutingModelBindingProducerError(
            f"routing model binding producer {name} is invalid"
        )
    return normalized


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutingModelBindingProducerError(
            f"routing model binding producer {name} is invalid"
        )
    return value


def _validate_source_bundle(
    source_bundle: Mapping[str, Any],
    *,
    artifact_hash: str,
) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "archive_sha256",
        "source_tree_hash",
        "archive_size_bytes",
        "archive_b64",
    }
    if not isinstance(source_bundle, Mapping) or set(source_bundle) != required:
        raise RoutingModelBindingProducerError("source bundle fields are invalid")
    if source_bundle.get("schema_version") != _SOURCE_BUNDLE_SCHEMA:
        raise RoutingModelBindingProducerError("source bundle schema is invalid")
    archive_hash = _hash(source_bundle.get("archive_sha256"), "source archive hash")
    if source_bundle.get("source_tree_hash") != artifact_hash:
        raise RoutingModelBindingProducerError("source bundle tree differs from artifact")
    try:
        archive = base64.b64decode(str(source_bundle.get("archive_b64")), validate=True)
    except Exception as exc:  # noqa: BLE001 - boundary must be fail closed
        raise RoutingModelBindingProducerError("source bundle archive is invalid") from exc
    if not archive:
        raise RoutingModelBindingProducerError("source bundle archive is empty")

    if sha256_bytes(archive) != archive_hash:
        raise RoutingModelBindingProducerError("source bundle archive hash differs")
    if source_bundle.get("archive_size_bytes") != len(archive):
        raise RoutingModelBindingProducerError("source bundle archive size differs")
    # Reconstruct the tree in a task-owned temporary directory.  The measured
    # sandbox repeats this check, but doing it here binds the executor request
    # to the exact bytes that will be supplied to the model.
    try:
        with tempfile.TemporaryDirectory(prefix="lp-routing-model-observe-") as root:
            extract_source_bundle_v2(
                source_bundle,
                destination=Path(root) / "source",
                expected_source_tree_hash=artifact_hash,
            )
    except Exception as exc:  # noqa: BLE001 - convert to one boundary error
        raise RoutingModelBindingProducerError("source bundle tree is invalid") from exc
    return dict(source_bundle)


def _build_metadata_payload(
    *,
    artifact: PrivateModelArtifactManifest,
    source_bundle: Mapping[str, Any],
    model_kind: str,
) -> dict[str, Any]:
    if model_kind not in {"private", "candidate"}:
        raise RoutingModelBindingProducerError("model kind is invalid")
    # Keep this document byte-for-byte compatible with the existing measured
    # ModelSandboxV2 metadata operation.  All provider and snapshot state is
    # empty by contract, so this operation cannot make a provider call.
    return {
        "schema_version": "leadpoet.model_sandbox_request.v2",
        "model_kind": model_kind,
        "operation": "metadata",
        "artifact": artifact.to_dict(),
        "source_bundle": dict(source_bundle),
        "module_name": MODEL_METADATA_MODULE_V2,
        "callable_name": MODEL_METADATA_CALLABLE_V2,
        "input": {},
        "environment": {},
        "provider_evidence_cache": {},
        "provider_evidence_cache_ref": "",
        "provider_evidence_mode": "",
        "provider_snapshot_bundle": {},
        "provider_snapshot_tree_hash": "",
        "provider_snapshot_manifest_hash": "",
        "provider_cost_scope": "",
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog": {},
        "provider_catalog_evidence": {},
    }


def _validate_lineage_and_artifact(
    lineage: VerifiedRoutingArtifactLineage,
    artifact_document: Mapping[str, Any],
) -> PrivateModelArtifactManifest:
    if not isinstance(lineage, VerifiedRoutingArtifactLineage):
        raise RoutingModelBindingProducerError("artifact lineage is not verified")
    if validate_sourcing_model_artifact_identity(lineage.sourcing_model_identity()):
        raise RoutingModelBindingProducerError("artifact lineage is invalid")
    try:
        artifact = PrivateModelArtifactManifest.from_mapping(artifact_document)
    except Exception as exc:  # noqa: BLE001
        raise RoutingModelBindingProducerError("model artifact manifest is invalid") from exc
    if validate_private_model_artifact_manifest(artifact):
        raise RoutingModelBindingProducerError("model artifact manifest is invalid")
    expected = {
        "model_artifact_hash": lineage.model_artifact_hash,
        "git_commit_sha": lineage.commit_sha,
        "image_digest": lineage.image_digest,
        "config_hash": lineage.config_hash,
        "manifest_hash": lineage.manifest_hash,
        "build_id": lineage.build_id,
    }
    for field, expected_value in expected.items():
        if getattr(artifact, field) != expected_value:
            raise RoutingModelBindingProducerError(
                f"model artifact {field} differs from verified lineage"
            )
    if not _GIT_SHA_RE.fullmatch(artifact.git_commit_sha):
        raise RoutingModelBindingProducerError("model artifact commit is invalid")
    return artifact


def _validate_model_result(
    result: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    artifact: PrivateModelArtifactManifest,
    source_bundle: Mapping[str, Any],
    model_kind: str,
) -> Mapping[str, Any]:
    result = _require_mapping(result, "measured model result")
    if set(result) not in {_MODEL_RESULT_FIELDS, _MODEL_RESULT_FIELDS | {"sealed_artifacts"}}:
        raise RoutingModelBindingProducerError("measured model result fields are invalid")
    if result.get("schema_version") != "leadpoet.model_sandbox_result.v2":
        raise RoutingModelBindingProducerError("measured model result schema is invalid")
    if result.get("model_kind") != model_kind or result.get("operation") != "metadata":
        raise RoutingModelBindingProducerError("measured model result operation differs")
    if result.get("model_artifact_hash") != artifact.model_artifact_hash:
        raise RoutingModelBindingProducerError("measured model artifact differs")
    if result.get("model_manifest_hash") != artifact.manifest_hash:
        raise RoutingModelBindingProducerError("measured model manifest differs")
    if result.get("compatibility_image_digest") != artifact.image_digest:
        raise RoutingModelBindingProducerError("measured model image differs")
    if result.get("source_bundle_hash") != source_bundle["archive_sha256"]:
        raise RoutingModelBindingProducerError("measured source bundle differs")
    for field in (
        "compatibility_policy_hash",
        "compatibility_admission_hash",
        "runtime_config_hash",
    ):
        _hash(result.get(field), field)
    empty_hash = sha256_json({})
    if result.get("input_hash") != sha256_json(payload["input"]):
        raise RoutingModelBindingProducerError("measured model input differs")
    if any(
        result.get(field) != empty_hash
        for field in (
            "provider_evidence_cache_hash",
            "provider_snapshot_archive_hash",
            "provider_snapshot_tree_hash",
            "provider_snapshot_manifest_hash",
            "provider_runtime_catalog_hash",
            "generated_provider_evidence_cache_hash",
        )
    ):
        raise RoutingModelBindingProducerError("metadata provider state is not empty")
    if any(
        result.get(field) not in ("", 0, {})
        for field in ("provider_evidence_cache_ref", "provider_evidence_mode")
    ):
        raise RoutingModelBindingProducerError("metadata provider state is not isolated")
    if result.get("provider_cost_cap_microusd") != 0 or result.get("provider_call_cap") != 0:
        raise RoutingModelBindingProducerError("metadata provider caps are not empty")
    if result.get("generated_provider_evidence_cache") != {}:
        raise RoutingModelBindingProducerError("metadata provider cache is not empty")
    if result.get("trace_entries") != [] or result.get("trace_entries_hash") != sha256_json([]):
        raise RoutingModelBindingProducerError("metadata trace contains provider activity")
    output = _require_mapping(result.get("output"), "measured model output")
    if result.get("output_hash") != sha256_json(output):
        raise RoutingModelBindingProducerError("measured model output hash differs")
    if not isinstance(output.get("runtime_routing"), Mapping):
        raise RoutingModelBindingProducerError("runtime routing metadata is missing")
    probe = _require_mapping(result.get("consumer_runtime_probe"), "runtime probe")
    if result.get("consumer_runtime_probe_hash") != sha256_json(probe):
        raise RoutingModelBindingProducerError("runtime probe hash differs")
    if "sealed_artifacts" in result and result["sealed_artifacts"] != []:
        raise RoutingModelBindingProducerError("metadata result contains sealed artifacts")
    return result


class RoutingModelBindingObservationProducerV2:
    """Run measured metadata and register a standard stage receipt.

    The producer deliberately has no signing dependency. ``record_stage`` is
    normally ``ExecutionContextV2.record_stage``; the ExecutionJobManager
    issues the synthetic stage receipt and chains it to the final job receipt.
    """

    def __init__(
        self,
        *,
        measured_metadata_executor: MeasuredModelMetadataExecutorV2,
        record_stage: StageRecorderV2,
    ) -> None:
        if not callable(measured_metadata_executor) or not callable(record_stage):
            raise TypeError("routing model binding producer dependencies are invalid")
        self._execute = measured_metadata_executor
        self._record_stage = record_stage

    def produce(
        self,
        *,
        artifact_lineage: VerifiedRoutingArtifactLineage,
        artifact_document: Mapping[str, Any],
        source_bundle: Mapping[str, Any],
        provider_bindings: Sequence[ProviderBindingIdentity],
        job_id: str,
        model_kind: str = "private",
    ) -> dict[str, Any]:
        artifact = _validate_lineage_and_artifact(artifact_lineage, artifact_document)
        bundle = _validate_source_bundle(
            source_bundle,
            artifact_hash=artifact.model_artifact_hash,
        )
        if not isinstance(provider_bindings, Sequence) or isinstance(
            provider_bindings, (str, bytes)
        ) or not provider_bindings:
            raise RoutingModelBindingProducerError("provider bindings are invalid")
        for binding in provider_bindings:
            if not isinstance(binding, ProviderBindingIdentity) or validate_provider_binding_identity(binding):
                raise RoutingModelBindingProducerError("provider binding identity is invalid")
        payload = _build_metadata_payload(
            artifact=artifact,
            source_bundle=bundle,
            model_kind=model_kind,
        )
        try:
            execution = self._execute(
                payload=payload,
                job_id=str(job_id),
                purpose=ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
            )
        except Exception as exc:  # noqa: BLE001 - provider/model errors fail closed
            raise RoutingModelBindingProducerError("measured metadata operation failed") from exc
        if not isinstance(execution, MeasuredModelMetadataExecutionV2):
            raise RoutingModelBindingProducerError("measured metadata execution is invalid")
        if dict(execution.payload) != payload:
            raise RoutingModelBindingProducerError("measured metadata payload was substituted")
        result = _validate_model_result(
            execution.result,
            payload=payload,
            artifact=artifact,
            source_bundle=bundle,
            model_kind=model_kind,
        )
        try:
            observation = observe_routing_model_bindings_v2(
                runtime_metadata=result["output"]["runtime_routing"],
                provider_bindings=provider_bindings,
                artifact_lineage_hash=artifact_lineage.identity_hash(),
            )
        except RoutingModelBindingObservationError as exc:
            raise RoutingModelBindingProducerError("model binding observation failed") from exc
        try:
            self._record_stage(
                purpose=ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
                # The standard verifier treats this synthetic stage receipt
                # as the attestation for the redacted observation.  Its input
                # must therefore be the observation request root; the full
                # measured sandbox result remains bound through artifact
                # hashes and the enclosing job receipt.
                input_root=observation["request_root"],
                output_root=sha256_json(observation),
                artifact_hashes=(
                    result["model_artifact_hash"],
                    result["model_manifest_hash"],
                    result["source_bundle_hash"],
                    result["compatibility_policy_hash"],
                    result["compatibility_admission_hash"],
                    result["runtime_config_hash"],
                    result["provider_runtime_catalog_hash"],
                    result["trace_entries_hash"],
                    result["output_hash"],
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise RoutingModelBindingProducerError("observation stage registration failed") from exc
        return {
            "schema_version": ROUTING_MODEL_BINDING_PRODUCER_RESULT_SCHEMA_V2,
            "operation": ROUTING_MODEL_BINDING_OBSERVATION_OPERATION_V2,
            "artifact_lineage_hash": artifact_lineage.identity_hash(),
            "model_result": {
                "model_artifact_hash": result["model_artifact_hash"],
                "model_manifest_hash": result["model_manifest_hash"],
                "source_bundle_hash": result["source_bundle_hash"],
                "compatibility_policy_hash": result["compatibility_policy_hash"],
                "compatibility_admission_hash": result["compatibility_admission_hash"],
                "runtime_config_hash": result["runtime_config_hash"],
                "provider_runtime_catalog_hash": result["provider_runtime_catalog_hash"],
                "trace_entries_hash": result["trace_entries_hash"],
                "output_hash": result["output_hash"],
                "runtime_catalog_sha256": result["output"]["runtime_routing"]["catalog_sha256"],
                "runtime_policy_sha256": result["output"]["runtime_routing"]["policy_sha256"],
            },
            "observation": observation,
        }


__all__ = [
    "MODEL_METADATA_CALLABLE_V2",
    "MODEL_METADATA_MODULE_V2",
    "MeasuredModelMetadataExecutionV2",
    "ROUTING_MODEL_BINDING_OBSERVATION_OPERATION_V2",
    "ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2",
    "ROUTING_MODEL_BINDING_PRODUCER_RESULT_SCHEMA_V2",
    "RoutingModelBindingObservationProducerV2",
    "RoutingModelBindingProducerError",
    "resolve_verified_routing_artifact_lineage_v2",
]
