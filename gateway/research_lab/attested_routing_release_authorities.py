"""Fixed authority source for the packaged Research Lab routing release.

This module is part of the release package.  It reads only attested JSON
documents mounted at fixed release paths and uses the existing TEE and
Supabase clients.  It never imports a module named by configuration or by a
request.  The exact model verifier/evaluator/runner authority is deliberately
not fabricated here: the signed model release must publish those objects with
this package before startup can proceed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping

from gateway.research_lab.routing_release_builder import (
    ReviewedRoutingReleaseAuthoritySources,
    RoutingReleaseDependencyError,
)
from gateway.research_lab.routing_authority_bundle import (
    VerifiedRoutingAuthorityBundle,
    load_verified_routing_authority_bundle,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
)
from gateway.research_lab.routing_admission import (
    build_routing_admission_bundle_v2,
    validate_routing_admission_bundle_v2,
)
from gateway.research_lab.routing_execution_envelope import (
    build_routing_execution_envelope_v2,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
    VerifiedRoutingGoldLabels,
    verify_routing_json_kms_signature,
)
from gateway.research_lab.routing_experiment_runtime import (
    ReviewedProviderBrokerRoutingRunner,
    RoutingExperimentRuntimeConfig,
)
from gateway.research_lab.routing_experiment_store import (
    SupabaseRoutingExperimentStore,
)
from gateway.research_lab.routing_product_composition import (
    ReviewedRoutingReleaseInputs,
    RoutingProviderDispatchTeeRpc,
    build_attested_protected_authorities,
)
from gateway.research_lab.routing_provider_bindings import (
    ReviewedDeeplineActionCompiler,
)
from gateway.utils.tee_client import TEEClient
from research_lab.eval import DockerPrivateModelRunner, DockerPrivateModelSpec
from research_lab.docker_model_runner_transport import DockerModelRunnerTransport
from research_lab.model_runner_protocol import (
    ExactModelRunnerRegistration,
    ResearchLabModelRunnerProtocol,
)
from research_lab.routing_experiments import ProviderReceiptStore
from research_lab.canonical import sha256_json


_BUNDLE_PATH_ENV = "RESEARCH_LAB_ROUTING_AUTHORITY_BUNDLE_PATH"
_BUNDLE_KEYS_PATH_ENV = "RESEARCH_LAB_ROUTING_AUTHORITY_KEYS_PATH"
_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
_ARTIFACT_VARIANT_BRANCHES = {
    "baseline": "main",
    "challenger": "leadpoet-lab",
}
_MODEL_RELEASE_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "source_commit",
        "model_artifact_digest",
        "dependency_lock_sha256",
        "runtime_base_image_digest",
        "consumer_contract_sha256",
        "catalog_sha256",
        "policy_sha256",
        "candidate_profiles_sha256",
        "intent_profiles_sha256",
        "feature_schema_sha256",
        "candidate_waterfall_contract_sha256",
        "verifier_artifact_digest",
        "tool_binding_manifest_sha256",
        "llm_configuration_sha256",
        "release_identity_sha256",
    }
)


def _release_document(env_name: str, label: str) -> Mapping[str, Any]:
    raw_path = str(os.environ.get(env_name) or "").strip()
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RoutingReleaseDependencyError(
            f"routing release {label} document is not an attested regular file"
        )
    try:
        if path.stat().st_size > _MAX_DOCUMENT_BYTES:
            raise ValueError("oversize")
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - release input is fail closed
        raise RoutingReleaseDependencyError(
            f"routing release {label} document cannot be loaded"
        ) from exc
    if not isinstance(value, Mapping):
        raise RoutingReleaseDependencyError(
            f"routing release {label} document is not an object"
        )
    return dict(value)


def _pinned_keys() -> Mapping[str, str]:
    value = _release_document(_BUNDLE_KEYS_PATH_ENV, "authority key")
    if not value or any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise RoutingReleaseDependencyError("routing release authority key pins are invalid")
    return dict(value)


def _run(coro: Any) -> Any:
    """Run one TEE coroutine without nesting an event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: list[Any] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - preserve RPC failure
            error.append(exc)

    thread = threading.Thread(target=worker, name="routing-release-tee-rpc")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


class AttestedRoutingTeeJobRpc:
    """The six-method routing job surface over the existing TEE client."""

    def __init__(self, client: TEEClient) -> None:
        if not isinstance(client, TEEClient):
            raise RoutingReleaseDependencyError("routing release TEE client is invalid")
        self._client = client

    def submit_job(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(_run(self._client.scoring_v2_submit_job(dict(manifest))))

    def put_chunk(
        self, *, job_id: str, offset: int, data_b64: str, chunk_sha256: str
    ) -> Mapping[str, Any]:
        try:
            data = base64.b64decode(str(data_b64), validate=True)
        except Exception as exc:
            raise RoutingReleaseDependencyError("routing release TEE chunk is invalid") from exc
        result = _run(
            self._client.scoring_v2_put_chunk(
                job_id=str(job_id), offset=int(offset), data=data
            )
        )
        if str(result.get("chunk_sha256") or chunk_sha256) != str(chunk_sha256):
            raise RoutingReleaseDependencyError("routing release TEE chunk hash differs")
        return dict(result)

    def seal(self, job_id: str) -> Mapping[str, Any]:
        return dict(_run(self._client.scoring_v2_seal_job(str(job_id))))

    def status(self, job_id: str) -> Mapping[str, Any]:
        return dict(_run(self._client.scoring_v2_get_status(str(job_id))))

    def result(
        self, job_id: str, *, offset: int = 0, max_bytes: int = 512 * 1024
    ) -> Mapping[str, Any]:
        return dict(
            _run(
                self._client.scoring_v2_get_result(
                    str(job_id), offset=int(offset), max_bytes=int(max_bytes)
                )
            )
        )

    def receipts(self, job_id: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            dict(item)
            for item in _run(self._client.scoring_v2_get_receipts(str(job_id)))
        )


def _verify_gold_label(document: Mapping[str, Any], key_id: str) -> Mapping[str, Any]:
    """Verify labels with the purpose-specific KMS JSON verifier."""

    result = verify_routing_json_kms_signature(document, key_id)
    if result.get("verified") is not True or str(result.get("key_id") or "") != key_id:
        raise RoutingReleaseDependencyError("routing release gold-label signature is invalid")
    return dict(result)


class _BundleArtifactAuthority:
    """Verify only the exact signed artifact documents in the v2 bundle."""

    def __init__(
        self,
        *,
        lineages: tuple[VerifiedRoutingArtifactLineage, ...],
        bundle: Mapping[str, Any],
    ) -> None:
        registrations = bundle.get("artifact_registrations")
        if not isinstance(registrations, Mapping) or set(registrations) != {
            "baseline",
            "challenger",
        }:
            raise RoutingReleaseDependencyError(
                "routing release artifact registrations are incomplete"
            )
        if len(lineages) != 2:
            raise RoutingReleaseDependencyError(
                "routing release requires two signed artifact lineages"
            )
        self._lineages = tuple(lineages)
        self._manifests: dict[str, Mapping[str, Any]] = {}
        for variant, lineage in zip(("baseline", "challenger"), lineages):
            registration = registrations.get(variant)
            documents = (
                registration.get("documents")
                if isinstance(registration, Mapping)
                else None
            )
            manifest = (
                documents.get("artifact_pointer")
                if isinstance(documents, Mapping)
                else None
            )
            if not isinstance(manifest, Mapping) or str(
                manifest.get("manifest_hash") or ""
            ) != lineage.manifest_hash:
                raise RoutingReleaseDependencyError(
                    f"routing release {variant} artifact manifest differs"
                )
            self._manifests[lineage.identity_hash()] = dict(manifest)

    def verify(
        self,
        *,
        artifact: Any,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        identity = (
            artifact.to_dict()
            if callable(getattr(artifact, "to_dict", None))
            else dict(artifact)
        )
        expected_lineage = next(
            (
                lineage
                for lineage in self._lineages
                if lineage.sourcing_model_identity().to_dict() == identity
            ),
            None,
        )
        if expected_lineage is None or dict(manifest) != dict(
            self._manifests[expected_lineage.identity_hash()]
        ):
            raise RoutingReleaseDependencyError(
                "routing release artifact authority identity differs"
            )
        return {
            "verified": True,
            "model_artifact_hash": expected_lineage.model_artifact_hash,
            "manifest_hash": expected_lineage.manifest_hash,
            "commit_sha": expected_lineage.commit_sha,
            "pointer_document_hash": expected_lineage.pointer_document_hash,
            "artifact_lineage_hash": expected_lineage.identity_hash(),
            "image_digest": expected_lineage.image_digest,
            "build_id": expected_lineage.build_id,
            "signature_ref": expected_lineage.signature_ref,
            "key_id": expected_lineage.signature_key_id,
            "signing_algorithm": expected_lineage.signature_algorithm,
            "consumer_contract_binding_mode": "semantic_v1_required",
        }


def _load_signed_gold_labels(
    *, document: Mapping[str, Any], key_id: str, unit_refs: tuple[str, ...]
) -> VerifiedRoutingGoldLabels:
    required = {
        "schema_version",
        "labels",
        "label_set_hash",
        "provenance_hash",
        "manifest_uri",
        "signature_ref",
        "manifest_hash",
    }
    if set(document) != required or document.get("schema_version") != "leadpoet.routing_gold_labels.v1":
        raise RoutingReleaseDependencyError(
            "routing release gold-label document is invalid"
        )
    if tuple(unit_refs) != tuple(sorted(set(unit_refs))):
        raise RoutingReleaseDependencyError(
            "routing release gold-label unit refs are invalid"
        )
    try:
        verified = _verify_gold_label(document, key_id)
        labels = document["labels"]
        payload = dict(document)
        manifest_hash = str(payload.pop("manifest_hash") or "")
        if sha256_json(payload) != manifest_hash:
            raise ValueError("manifest hash")
        manifest_uri = str(document["manifest_uri"] or "")
        if (
            not manifest_uri.startswith("s3://")
            or manifest_uri.endswith("/current.json")
            or "/branches/" in manifest_uri
        ):
            raise ValueError("manifest URI")
        if not isinstance(labels, Mapping) or set(labels) != set(unit_refs):
            raise ValueError("labels")
        normalized = {str(key): value for key, value in labels.items()}
        if any(type(value) is not bool for value in normalized.values()):
            raise ValueError("label value")
    except Exception as exc:  # noqa: BLE001 - signed labels are fail closed
        if isinstance(exc, RoutingReleaseDependencyError):
            raise
        raise RoutingReleaseDependencyError(
            "routing release gold-label signature is invalid"
        ) from exc
    if verified.get("manifest_hash") != document["manifest_hash"]:
        raise RoutingReleaseDependencyError(
            "routing release gold-label hash differs"
        )
    return VerifiedRoutingGoldLabels(
        manifest_uri=str(document["manifest_uri"]),
        manifest_hash=str(document["manifest_hash"]),
        signature_ref=str(document["signature_ref"]),
        signing_key_id=str(key_id),
        label_set_hash=str(document["label_set_hash"]),
        labels=normalized,
        provenance_hash=str(document["provenance_hash"]),
    )


def _require_upstream_model_operations() -> None:
    """Fail closed until the signed artifact publishes model-owned operations."""

    required = (
        "build_host_capability_manifest",
        "evaluate_model_verifier_action",
    )
    missing = tuple(
        operation
        for operation in required
        if not callable(getattr(DockerModelRunnerTransport, operation, None))
    )
    if missing:
        raise RoutingReleaseDependencyError(
            "routing release upstream model adapter operations are unavailable: "
            + ", ".join(missing)
            + "; no Lab fallback is permitted"
        )


def _artifact_runner_registration(
    *,
    variant: str,
    lineage: VerifiedRoutingArtifactLineage,
    bundle: Mapping[str, Any],
    host_capability_manifest: Mapping[str, Any] | None = None,
) -> ExactModelRunnerRegistration:
    expected_branch = _ARTIFACT_VARIANT_BRANCHES.get(variant)
    if expected_branch is None or lineage.branch != expected_branch:
        raise RoutingReleaseDependencyError(
            f"routing release {variant} artifact branch is invalid"
        )
    registrations = bundle.get("artifact_registrations")
    registration = registrations.get(variant) if isinstance(registrations, Mapping) else None
    documents = registration.get("documents") if isinstance(registration, Mapping) else None
    pointer = documents.get("artifact_pointer") if isinstance(documents, Mapping) else None
    if not isinstance(pointer, Mapping):
        raise RoutingReleaseDependencyError(
            f"routing release {variant} signed artifact pointer is unavailable"
        )
    release_identity = pointer.get("model_release_identity")
    if not isinstance(release_identity, Mapping):
        raise RoutingReleaseDependencyError(
            f"routing release {variant} signed artifact is missing model_release_identity"
        )
    if not set(release_identity).issuperset(_MODEL_RELEASE_IDENTITY_FIELDS):
        raise RoutingReleaseDependencyError(
            f"routing release {variant} signed model_release_identity fields are incomplete"
        )
    if not isinstance(host_capability_manifest, Mapping):
        raise RoutingReleaseDependencyError(
            "routing release upstream host capability manifest is unavailable"
        )
    identity = lineage.sourcing_model_identity().to_dict()
    expected_identity_fields = {
        "source_commit": lineage.commit_sha,
        "model_artifact_digest": lineage.model_artifact_hash,
        "consumer_contract_sha256": lineage.routing_contract_hash.removeprefix("sha256:"),
        "catalog_sha256": lineage.routing_catalog_hash.removeprefix("sha256:"),
        "policy_sha256": lineage.routing_policy_hash.removeprefix("sha256:"),
        "feature_schema_sha256": lineage.feature_schema_hash.removeprefix("sha256:"),
    }
    if any(
        str(release_identity.get(key) or "").removeprefix("sha256:")
        != str(value).removeprefix("sha256:")
        for key, value in expected_identity_fields.items()
    ):
        raise RoutingReleaseDependencyError(
            f"routing release {variant} signed release identity differs from artifact lineage"
        )
    try:
        runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=lineage.image_digest,
                pull_before_run=False,
            )
        )
        transport = DockerModelRunnerTransport(runner)
        protocol = ResearchLabModelRunnerProtocol(
            transport=transport,
            expected_release_identity=dict(release_identity),
        )
        result = ExactModelRunnerRegistration(
            artifact_identity=identity,
            protocol=protocol,
            host_capability_manifest=dict(host_capability_manifest),
        )
    except Exception as exc:  # noqa: BLE001 - signed OCI boundary is fail closed
        raise RoutingReleaseDependencyError(
            f"routing release {variant} signed OCI runner registration failed"
        ) from exc
    try:
        result.validate_identity()
    except Exception as exc:  # noqa: BLE001 - exact registration is fail closed
        raise RoutingReleaseDependencyError(
            "routing release exact model runner registration identity failed"
        ) from exc
    return result


def _artifact_registrations(
    *,
    lineages: tuple[VerifiedRoutingArtifactLineage, ...],
    bundle: Mapping[str, Any],
    host_capability_manifest: Mapping[str, Any] | None = None,
) -> tuple[ExactModelRunnerRegistration, ...]:
    if len(lineages) != 2:
        raise RoutingReleaseDependencyError(
            "routing release requires exactly two signed model artifacts"
        )
    registrations = tuple(
        _artifact_runner_registration(
            variant=variant,
            lineage=lineage,
            bundle=bundle,
            host_capability_manifest=host_capability_manifest,
        )
        for variant, lineage in zip(("baseline", "challenger"), lineages)
    )
    if len({registration.key for registration in registrations}) != 2:
        raise RoutingReleaseDependencyError(
            "routing release model runner registrations are duplicated"
        )
    return registrations


class _ReviewedRunnerFactory:
    """Build one protected provider runner from exact release authorities."""

    def __init__(
        self,
        *,
        inputs: ReviewedRoutingReleaseInputs,
        model_observation: VerifiedRoutingModelBindingRequirements,
        protected_release_receipt: Mapping[str, Any],
        store_factory: Any,
        execution_envelope_factory: Any,
        protected_authorities_factory: Any,
    ) -> None:
        self._inputs = inputs
        self._model_observation = model_observation
        self._protected_release_receipt = dict(protected_release_receipt)
        self._store_factory = store_factory
        self._execution_envelope_factory = execution_envelope_factory
        self._protected_authorities_factory = protected_authorities_factory

    def validate_readiness(self) -> None:
        for value, label in (
            (self._store_factory, "store factory"),
            (self._execution_envelope_factory, "execution envelope factory"),
            (self._protected_authorities_factory, "protected authority factory"),
        ):
            if not callable(value):
                raise RoutingReleaseDependencyError(
                    f"routing release {label} is unavailable"
                )

    def __call__(self, spec: Any) -> ReviewedProviderBrokerRoutingRunner:
        try:
            envelope = self._execution_envelope_factory(spec)
            job_id = str(self._protected_release_receipt.get("job_id") or "")
            if not job_id:
                raise RoutingReleaseDependencyError(
                    "routing release protected job identity is unavailable"
                )
            admission = build_routing_admission_bundle_v2(
                job_id=job_id,
                spec=spec,
                envelope=envelope,
                artifact_lineage=self._inputs.artifact_lineage,
                gold_labels=self._inputs.gold_labels,
                binding_catalog=self._inputs.binding_catalog,
                unit_dataset=self._inputs.unit_dataset,
                model_binding_observation=self._inputs.model_binding_observation,
                protected_release_receipt=self._protected_release_receipt,
            )
            validate_routing_admission_bundle_v2(
                bundle=admission,
                spec=spec,
                envelope=envelope,
                artifact_lineage=self._inputs.artifact_lineage,
                gold_labels=self._inputs.gold_labels,
                binding_catalog=self._inputs.binding_catalog,
                unit_dataset=self._inputs.unit_dataset,
                model_binding_observation=self._inputs.model_binding_observation,
                protected_release_receipt=self._protected_release_receipt,
                job_id=job_id,
            )
            lineages_by_variant: dict[str, VerifiedRoutingArtifactLineage] = {}
            for variant in spec.variants:
                identity = variant.artifact.to_dict()
                lineage = next(
                    (
                        item
                        for item in self._inputs.artifact_lineages
                        if item.sourcing_model_identity().to_dict() == identity
                    ),
                    None,
                )
                if lineage is None:
                    raise RoutingReleaseDependencyError(
                        "routing release variant artifact lineage is unavailable"
                    )
                lineages_by_variant[variant.variant_id] = lineage
            protected = self._protected_authorities_factory()
            store = self._store_factory()
            if not isinstance(store, SupabaseRoutingExperimentStore):
                raise RoutingReleaseDependencyError(
                    "routing release store factory returned an invalid store"
                )
            parent_graph = {
                "receipts": [
                    dict(self._protected_release_receipt),
                    dict(self._model_observation.signed_receipt),
                ]
            }

            def validate_admission(
                candidate: Any, release: Mapping[str, Any]
            ) -> None:
                validate_routing_admission_bundle_v2(
                    bundle=candidate,
                    spec=spec,
                    envelope=envelope,
                    artifact_lineage=self._inputs.artifact_lineage,
                    gold_labels=self._inputs.gold_labels,
                    binding_catalog=self._inputs.binding_catalog,
                    unit_dataset=self._inputs.unit_dataset,
                    model_binding_observation=self._inputs.model_binding_observation,
                    protected_release_receipt=release,
                    job_id=job_id,
                )

            runner = ReviewedProviderBrokerRoutingRunner(
                config=RoutingExperimentRuntimeConfig.from_env(),
                store=store,
                artifact_lineage=self._inputs.artifact_lineage,
                artifact_lineages=lineages_by_variant,
                compiler=ReviewedDeeplineActionCompiler(
                    binding_catalog=self._inputs.binding_catalog,
                    unit_dataset=self._inputs.unit_dataset,
                ),
                model_binding_requirements=self._inputs.model_binding_observation,
                authorization_authority=protected.call_authorization_authority,
                dispatch_authority=protected.dispatch_authority,
                execution_envelope=envelope,
                admission_bundle=admission,
                protected_release_receipt=self._protected_release_receipt,
                authorization_parent_receipt_graphs=(parent_graph,),
                dispatch_parent_receipt_graphs=(parent_graph,),
                admission_validator=validate_admission,
            )
            runner.validate_composition()
            return runner
        except RoutingReleaseDependencyError:
            raise
        except Exception as exc:  # noqa: BLE001 - runner composition is fail closed
            raise RoutingReleaseDependencyError(
                "routing release reviewed runner construction failed"
            ) from exc


def _build_billing_rollup_factory() -> Any:
    def factory(spec: Any) -> Any:
        experiment_hash = str(spec.experiment_hash())

        def rollup(receipt_store: ProviderReceiptStore) -> Mapping[str, Any]:
            if not isinstance(receipt_store, ProviderReceiptStore):
                raise RoutingReleaseDependencyError(
                    "routing release billing receipt store is invalid"
                )
            rows = []
            for key in sorted(receipt_store.repository.keys()):
                receipt = receipt_store.repository.get(key)
                if receipt is not None:
                    rows.append(receipt.to_dict())
            total = sum(int(row.get("credit_microunits", 0)) for row in rows)
            payload = {
                "schema_version": "leadpoet.research_lab.routing_billing_rollup.v1",
                "experiment_hash": experiment_hash,
                "receipts": rows,
                "total_credit_microunits": total,
            }
            digest = sha256_json(payload)
            return {
                "rollup_id": "routing-billing-rollup:" + digest.split(":", 1)[1][:32],
                "rollup_hash": digest,
                "total_credit_microunits": total,
            }

        return rollup

    return factory


def load_reviewed_routing_release_authority_sources() -> ReviewedRoutingReleaseAuthoritySources:
    """Load the fixed source documents and concrete runtime authorities.

    The verifier, evaluation adapter, and runner transport must be exposed by
    the signed OCI artifact.  Host code composes only the protected TEE,
    store, envelope, and billing seams around those authorities.
    """

    bundle = _release_document(_BUNDLE_PATH_ENV, "authority bundle")
    keys = _pinned_keys()
    try:
        authority_bundle = load_verified_routing_authority_bundle(
            bundle, pinned_public_keys=keys
        )
    except Exception as exc:  # noqa: BLE001 - signed release is fail closed
        raise RoutingReleaseDependencyError(
            "routing release signed authority bundle is invalid"
        ) from exc
    if not isinstance(authority_bundle, VerifiedRoutingAuthorityBundle):
        raise RoutingReleaseDependencyError(
            "routing release signed authority bundle is unavailable"
        )
    lineages = tuple(authority_bundle.artifact_lineages)
    if len(lineages) != 2:
        raise RoutingReleaseDependencyError(
            "routing release requires exactly two signed artifact lineages"
        )
    if lineages[0].branch != "main" or lineages[1].branch != "leadpoet-lab":
        raise RoutingReleaseDependencyError(
            "routing release baseline and challenger artifact branches are invalid"
        )
    _require_upstream_model_operations()
    raise RoutingReleaseDependencyError(
        "routing release model-owned verifier and evaluator exports are not published"
    )



__all__ = [
    "AttestedRoutingTeeJobRpc",
    "load_reviewed_routing_release_authority_sources",
]
