"""Build the reviewed Lab routing dependency bundle from attested sources.

This is the release-owned composition layer.  It accepts only typed runtime
authorities and signed documents that were already delivered by the measured
release.  It does not import a provider, model, credential, endpoint, or
factory from request data or an environment-selected module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway.research_lab.routing_authority_bundle import (
    load_verified_routing_authority_bundle,
)
from gateway.research_lab.routing_experiment_artifacts import (
    SignedRoutingGoldLabelLoader,
    VerifiedRoutingGoldLabels,
)
from gateway.research_lab.routing_product_composition import (
    ReviewedRoutingReleaseInputs,
    RoutingProductCompositionError,
    RoutingProviderDispatchTeeRpc,
    validate_reviewed_release_inputs,
)
from gateway.research_lab.routing_experiment_worker import (
    ExactModelEvaluationAdapter,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
)
from gateway.research_lab.routing_experiment_runtime import (
    ReviewedProviderBrokerRoutingRunner,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.common_model_experiment import (
    ReviewedModelVerificationAuthority,
)
from gateway.research_lab.routing_provider_bindings import (
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.tee.protected_workflows import (
    DEFAULT_MANIFEST,
    load_manifest,
    verify_manifest,
)
from research_lab.model_runner_protocol import (
    ExactModelRunnerRegistration,
    ExactModelRunnerRegistry,
)
from research_lab.routing_experiments import RoutingExperimentArtifactAuthority


RELEASE_SOURCE_SCHEMA_VERSION = "leadpoet.research_lab.routing_release_sources.v1"
RELEASE_MODULE_SCHEMA_VERSION = "leadpoet.research_lab.routing_release_module.v1"
RELEASE_AUTHORITY_PROVIDER_MODULE = (
    "gateway.research_lab.attested_routing_release_authorities"
)


class RoutingReleaseDependencyError(RoutingProductCompositionError):
    """The attested release cannot compose a complete Lab dependency bundle."""


@dataclass(frozen=True)
class ReviewedRoutingReleaseAuthoritySources:
    """Inputs supplied by one measured, release-owned authority provider.

    The bundle, label document, and protected manifest hash are immutable
    release inputs.  The remaining fields are concrete authority objects from
    the same release; protocols are used only as type contracts and are
    checked before a dependency bundle is returned.
    """

    authority_bundle_document: Mapping[str, Any]
    authority_bundle_pinned_public_keys: Mapping[str, str]
    gold_label_document: Mapping[str, Any]
    gold_label_key_id: str
    gold_label_verifier: Callable[[Mapping[str, Any], str], Mapping[str, Any]]
    expected_label_set_hash: str
    expected_unit_refs: tuple[str, ...]
    model_binding_observation: VerifiedRoutingModelBindingRequirements
    protected_release_receipt: Mapping[str, Any]
    artifact_authority: RoutingExperimentArtifactAuthority
    model_verifier: ReviewedModelVerificationAuthority
    evaluation_adapter: ExactModelEvaluationAdapter
    scoring_job_rpc: Any
    call_authorization_job_rpc: Any
    dispatch_job_rpc: Any
    reviewed_runner_factory: Callable[
        [Any], ReviewedProviderBrokerRoutingRunner
    ]
    billing_rollup_factory: Callable[[Any], Callable[..., Mapping[str, Any]]]
    execution_envelope_factory: Callable[[Any], RoutingExperimentExecutionEnvelopeV2]
    store_factory: Callable[[], Any]
    protected_workflow_manifest_hash: str
    model_runner_registry: ExactModelRunnerRegistry | None = None
    # The registry is rebuilt from these exact, release-published registrations
    # when supplied.  Keeping the legacy field during the transition lets old
    # unit fixtures fail closed on the dual-artifact count without making a
    # caller-provided registry an authority for the new release path.
    model_runner_registrations: tuple[ExactModelRunnerRegistration, ...] | None = None


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutingReleaseDependencyError(f"routing release {label} is invalid")
    return value


def _require_callable(value: Any, label: str) -> Callable[..., Any]:
    if not callable(value):
        raise RoutingReleaseDependencyError(f"routing release {label} is unavailable")
    return value


def _require_rpc(value: Any, label: str) -> Any:
    if value is None:
        raise RoutingReleaseDependencyError(f"routing release {label} RPC is unavailable")
    for method in ("submit_job", "put_chunk", "seal", "status", "result", "receipts"):
        if not callable(getattr(value, method, None)):
            raise RoutingReleaseDependencyError(
                f"routing release {label} RPC is missing {method}"
            )
    return value


def _verify_protected_workflow_manifest(expected_hash: str) -> None:
    expected = str(expected_hash or "").strip().lower()
    manifest = load_manifest(DEFAULT_MANIFEST)
    if manifest.get("manifest_hash") != expected:
        raise RoutingReleaseDependencyError(
            "routing release protected workflow manifest identity differs"
        )
    try:
        verify_manifest(Path(__file__).resolve().parents[2], manifest)
    except Exception as exc:  # noqa: BLE001 - fail closed at the release boundary
        raise RoutingReleaseDependencyError(
            "routing release protected workflow manifest is not attested"
        ) from exc


def _load_signed_gold_labels(
    sources: ReviewedRoutingReleaseAuthoritySources,
    *,
    unit_dataset: VerifiedRoutingUnitDataset,
) -> VerifiedRoutingGoldLabels:
    document = _require_mapping(sources.gold_label_document, "gold-label document")
    manifest_uri = str(document.get("manifest_uri") or "").strip()
    loader = SignedRoutingGoldLabelLoader(
        manifest_uri=manifest_uri,
        loader=lambda _uri: dict(document),
        verifier=sources.gold_label_verifier,
        key_id=sources.gold_label_key_id,
    )
    refs = tuple(sources.expected_unit_refs)
    if refs != tuple(sorted(set(refs))) or set(refs) != set(unit_dataset.units):
        raise RoutingReleaseDependencyError(
            "routing release signed unit refs are not exact"
        )
    try:
        return loader.load(
            expected_label_set_hash=sources.expected_label_set_hash,
            expected_unit_refs=refs,
        )
    except Exception as exc:  # noqa: BLE001 - preserve one release error type
        raise RoutingReleaseDependencyError(
            "routing release signed gold labels are invalid"
        ) from exc


def build_reviewed_routing_release_dependencies(
    sources: ReviewedRoutingReleaseAuthoritySources,
    *,
    environment: Mapping[str, str] | None = None,
    expected_protected_workflow_manifest_hash: str | None = None,
) -> Any:
    """Compose the exact bootstrap dataclass used by gateway and consumer.

    This function is intentionally the only constructor used by the generated
    release module.  It verifies the signed artifact/binding/unit bundle and
    gold labels before constructing the Lab input dataclass, then validates all
    model, TEE, protected-release, and runtime identities before returning.
    """

    if not isinstance(sources, ReviewedRoutingReleaseAuthoritySources):
        raise RoutingReleaseDependencyError(
            "routing release authority sources are not typed"
        )
    expected_manifest_hash = (
        sources.protected_workflow_manifest_hash
        if expected_protected_workflow_manifest_hash is None
        else expected_protected_workflow_manifest_hash
    )
    _verify_protected_workflow_manifest(expected_manifest_hash)
    if not isinstance(sources.model_binding_observation, VerifiedRoutingModelBindingRequirements):
        raise RoutingReleaseDependencyError(
            "routing release model binding observation is unavailable"
        )
    if sources.model_runner_registrations is not None:
        registrations = tuple(sources.model_runner_registrations)
        if len(registrations) != 2:
            raise RoutingReleaseDependencyError(
                "routing release requires exactly two model runner registrations"
            )
        try:
            model_runner_registry = ExactModelRunnerRegistry(registrations)
        except Exception as exc:  # noqa: BLE001 - exact registration boundary
            raise RoutingReleaseDependencyError(
                "routing release exact model runner registrations are invalid"
            ) from exc
    elif isinstance(sources.model_runner_registry, ExactModelRunnerRegistry):
        # Legacy fixtures are retained only as an explicit compatibility seam;
        # the signed dual-artifact release path below still rejects one-lineage
        # bundles and therefore cannot activate through this fallback.
        model_runner_registry = sources.model_runner_registry
    else:
        raise RoutingReleaseDependencyError(
            "routing release exact model runner registry is unavailable"
        )
    if not callable(getattr(sources.artifact_authority, "verify", None)):
        raise RoutingReleaseDependencyError(
            "routing release artifact authority is unavailable"
        )
    for method in ("verify_company", "verify_intent", "verify_contact"):
        if not callable(getattr(sources.model_verifier, method, None)):
            raise RoutingReleaseDependencyError(
                f"routing release model verifier is missing {method}"
            )
    for method in ("build_decision_receipts", "build_evaluation"):
        if not callable(getattr(sources.evaluation_adapter, method, None)):
            raise RoutingReleaseDependencyError(
                f"routing release evaluation adapter is missing {method}"
            )
    for value, label in (
        (sources.gold_label_verifier, "gold-label verifier"),
        (sources.reviewed_runner_factory, "reviewed runner factory"),
        (sources.billing_rollup_factory, "billing authority"),
        (sources.execution_envelope_factory, "execution envelope factory"),
        (sources.store_factory, "store factory"),
    ):
        _require_callable(value, label)
    runner_readiness = getattr(sources.reviewed_runner_factory, "validate_readiness", None)
    if not callable(runner_readiness):
        raise RoutingReleaseDependencyError(
            "routing release reviewed runner readiness is unavailable"
        )
    try:
        runner_readiness()
    except Exception as exc:  # noqa: BLE001 - release authority must be ready
        raise RoutingReleaseDependencyError(
            "routing release reviewed runner readiness failed"
        ) from exc
    _require_rpc(sources.scoring_job_rpc, "model observation")
    _require_rpc(sources.call_authorization_job_rpc, "call authorization")
    _require_rpc(sources.dispatch_job_rpc, "provider dispatch")
    if not isinstance(sources.protected_release_receipt, Mapping):
        raise RoutingReleaseDependencyError(
            "routing release protected receipt is unavailable"
        )

    try:
        authority_bundle = load_verified_routing_authority_bundle(
            sources.authority_bundle_document,
            pinned_public_keys=sources.authority_bundle_pinned_public_keys,
        )
    except Exception as exc:  # noqa: BLE001 - signed package is fail closed
        if isinstance(exc, RoutingReleaseDependencyError):
            raise
        raise RoutingReleaseDependencyError(
            "routing release signed authority bundle is invalid"
        ) from exc
    if len(authority_bundle.artifact_lineages) != 2:
        raise RoutingReleaseDependencyError(
            "routing release requires two distinct signed leadpoet-lab artifacts"
        )
    if sources.model_runner_registrations is not None:
        expected_artifacts = {
            sha256_json(lineage.sourcing_model_identity().to_dict())
            for lineage in authority_bundle.artifact_lineages
        }
        actual_artifacts = {
            sha256_json(dict(registration.artifact_identity))
            for registration in sources.model_runner_registrations
        }
        if actual_artifacts != expected_artifacts:
            raise RoutingReleaseDependencyError(
                "routing release model runner registrations differ from signed artifacts"
            )
    gold_labels = _load_signed_gold_labels(
        sources,
        unit_dataset=authority_bundle.unit_dataset,
    )
    inputs = ReviewedRoutingReleaseInputs(
        artifact_lineage=authority_bundle.artifact_lineage,
        binding_catalog=authority_bundle.binding_catalog,
        unit_dataset=authority_bundle.unit_dataset,
        authority_bundle=authority_bundle,
        gold_labels=gold_labels,
        model_binding_observation=sources.model_binding_observation,
        protected_release_receipt=dict(sources.protected_release_receipt),
        artifact_authority=sources.artifact_authority,
        model_runner_registry=model_runner_registry,
        model_verifier=sources.model_verifier,
        evaluation_adapter=sources.evaluation_adapter,
        scoring_job_rpc=sources.scoring_job_rpc,
        call_authorization_job_rpc=sources.call_authorization_job_rpc,
        dispatch_job_rpc=(
            sources.dispatch_job_rpc
            if type(sources.dispatch_job_rpc) is RoutingProviderDispatchTeeRpc
            else RoutingProviderDispatchTeeRpc(sources.dispatch_job_rpc)
        ),
        artifact_lineages=authority_bundle.artifact_lineages,
    )
    try:
        validate_reviewed_release_inputs(inputs, environment=environment)
    except Exception as exc:  # noqa: BLE001 - release identity is fail closed
        raise RoutingReleaseDependencyError(
            "routing release input identity validation failed"
        ) from exc

    from gateway.research_lab.routing_product_bootstrap import (
        ReviewedRoutingBootstrapDependencies,
    )

    return ReviewedRoutingBootstrapDependencies(
        inputs=inputs,
        reviewed_runner_factory=sources.reviewed_runner_factory,
        billing_rollup_factory=sources.billing_rollup_factory,
        execution_envelope_factory=sources.execution_envelope_factory,
        store_factory=sources.store_factory,
    )


def render_generated_release_module(
    *,
    protected_workflow_manifest_hash: str,
) -> str:
    """Render the fixed release shim used by the attested package builder."""

    expected = str(protected_workflow_manifest_hash or "").strip().lower()
    if not expected.startswith("sha256:") or len(expected) != 71:
        raise RoutingReleaseDependencyError(
            "routing release protected workflow manifest hash is invalid"
        )
    return (
        '"""Generated by the reviewed Research Lab release builder.\n'
        'Do not edit: regenerate from the attested release inputs.\n"""\n\n'
        "from gateway.research_lab.routing_release_builder import (\n"
        "    RELEASE_MODULE_SCHEMA_VERSION,\n"
        "    RELEASE_AUTHORITY_PROVIDER_MODULE,\n"
        "    build_reviewed_routing_release_dependencies,\n"
        ")\n\n"
        f"RELEASE_MODULE_SCHEMA = {RELEASE_MODULE_SCHEMA_VERSION!r}\n"
        f"EXPECTED_PROTECTED_WORKFLOW_MANIFEST_HASH = {expected!r}\n\n"
        "def load_reviewed_routing_release_dependencies():\n"
        "    # The provider module is a fixed release input, not a request or env import.\n"
        "    from gateway.research_lab.attested_routing_release_authorities import (\n"
        "        load_reviewed_routing_release_authority_sources,\n"
        "    )\n"
        "    sources = load_reviewed_routing_release_authority_sources()\n"
        "    return build_reviewed_routing_release_dependencies(\n"
        "        sources,\n"
        "        expected_protected_workflow_manifest_hash=(\n"
        "            EXPECTED_PROTECTED_WORKFLOW_MANIFEST_HASH\n"
        "        ),\n"
        "    )\n"
    )


__all__ = [
    "RELEASE_AUTHORITY_PROVIDER_MODULE",
    "RELEASE_MODULE_SCHEMA_VERSION",
    "RoutingReleaseDependencyError",
    "ReviewedRoutingReleaseAuthoritySources",
    "build_reviewed_routing_release_dependencies",
    "render_generated_release_module",
]
