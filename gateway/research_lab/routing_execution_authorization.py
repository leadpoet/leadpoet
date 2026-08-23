"""Exact protected authorization contract for one routing provider call.

This is distinct from the final experiment-evaluation attestation.  The
protected scoring role signs the complete pre-dispatch identity.  A broker
request is admissible only when its action/body/time/cost identity has the
same canonical authorization hash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    validate_provider_binding_identity,
)


ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2 = "attest_routing_provider_call_v2"
ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2 = "research_lab.routing_provider_evidence.v2"
ROUTING_PROVIDER_AUTHORIZATION_SCHEMA_V2 = "leadpoet.routing_provider_call_grant.v2"
ROUTING_PROVIDER_AUTHORIZATION_RESULT_SCHEMA_V2 = (
    "leadpoet.routing_provider_call_grant_result.v2"
)
ROUTING_PROVIDER_AUTHORIZATION_REQUEST_SCHEMA_V2 = (
    "leadpoet.routing_provider_call_authorization_request.v2"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class RoutingProviderAuthorizationError(ValueError):
    """A provider authorization is incomplete or self-inconsistent."""


def _hash(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(text):
        raise RoutingProviderAuthorizationError(f"routing provider {name} is invalid")
    return text


def _ref(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not _REF_RE.fullmatch(text):
        raise RoutingProviderAuthorizationError(f"routing provider {name} is invalid")
    return text


@dataclass(frozen=True)
class RoutingProviderCallAuthorizationV2:
    # This is the durable admission job identity.  The protected authorization
    # execution is a separate ExecutionJobManager job and is not knowable when
    # this signed call document is built.
    admission_job_id: str
    experiment_hash: str
    experiment_id: str
    purpose: str
    envelope_hash: str
    admission_bundle_hash: str
    protected_release_hash: str
    protected_boot_identity_hash: str
    variant_id: str
    stage: str
    artifact_lineage_hash: str
    pointer_document_hash: str
    model_artifact_hash: str
    manifest_hash: str
    image_digest: str
    commit_sha: str
    build_id: str
    routing_contract_hash: str
    routing_catalog_hash: str
    routing_policy_hash: str
    feature_schema_hash: str
    verifier_contract_hash: str
    binding: ProviderBindingIdentity
    transport_id: str
    binding_catalog_manifest_hash: str
    binding_catalog_version: str
    action_id: str
    unit_ref: str
    unit_input_hash: str
    unit_dataset_manifest_hash: str
    unit_set_hash: str
    model_binding_observation_receipt_hash: str
    attempt: int
    core_request_fingerprint: str
    request_body_hash: str
    retry_policy_hash: str
    credit_cap_microunits: int
    timeout_ms: int
    claim_key: str
    claim_generation: int
    claim_fence_hash: str
    schema_version: str = ROUTING_PROVIDER_AUTHORIZATION_SCHEMA_V2

    def __post_init__(self) -> None:
        validate_routing_provider_call_authorization_v2(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["binding"] = self.binding.to_dict()
        return result

    def authorization_hash(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "RoutingProviderCallAuthorizationV2":
        validate_routing_provider_call_authorization_v2(payload)
        values = dict(payload)
        values["binding"] = ProviderBindingIdentity.from_mapping(values["binding"])
        return cls(**values)


def validate_routing_provider_call_authorization_v2(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "admission_job_id", "experiment_hash", "experiment_id", "purpose",
        "envelope_hash", "admission_bundle_hash", "protected_release_hash",
        "protected_boot_identity_hash", "variant_id", "stage",
        "artifact_lineage_hash", "pointer_document_hash", "model_artifact_hash",
        "manifest_hash", "image_digest", "commit_sha", "build_id",
        "routing_contract_hash", "routing_catalog_hash", "routing_policy_hash",
        "feature_schema_hash", "verifier_contract_hash", "binding",
        "transport_id", "binding_catalog_manifest_hash", "binding_catalog_version", "action_id",
        "unit_ref", "unit_input_hash", "unit_dataset_manifest_hash", "unit_set_hash",
        "model_binding_observation_receipt_hash",
        "attempt", "core_request_fingerprint", "request_body_hash", "retry_policy_hash",
        "credit_cap_microunits", "timeout_ms", "claim_key", "claim_generation",
        "claim_fence_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise RoutingProviderAuthorizationError("routing provider authorization fields are invalid")
    if payload.get("schema_version") != ROUTING_PROVIDER_AUTHORIZATION_SCHEMA_V2:
        raise RoutingProviderAuthorizationError("routing provider authorization schema is invalid")
    if payload.get("purpose") != ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2:
        raise RoutingProviderAuthorizationError("routing provider authorization purpose is invalid")
    for name in (
        "experiment_hash", "envelope_hash", "admission_bundle_hash",
        "protected_release_hash", "protected_boot_identity_hash",
        "artifact_lineage_hash", "pointer_document_hash",
        "model_artifact_hash", "manifest_hash",
        "routing_contract_hash", "routing_catalog_hash", "routing_policy_hash",
        "feature_schema_hash", "verifier_contract_hash", "binding_catalog_manifest_hash",
        "unit_input_hash", "unit_dataset_manifest_hash", "unit_set_hash",
        "model_binding_observation_receipt_hash",
        "core_request_fingerprint", "request_body_hash", "retry_policy_hash", "claim_key",
        "claim_fence_hash",
    ):
        _hash(payload.get(name), name)
    image_digest = str(payload.get("image_digest") or "")
    if (
        "@sha256:" not in image_digest
        or image_digest.endswith(":latest")
        or not re.search(r"@sha256:[0-9a-f]{64}$", image_digest)
    ):
        raise RoutingProviderAuthorizationError("routing provider image_digest is invalid")
    for name in (
        "admission_job_id", "experiment_id", "variant_id", "stage", "build_id", "transport_id",
        "binding_catalog_version",
        "action_id", "unit_ref",
    ):
        _ref(payload.get(name), name)
    if not _GIT_SHA_RE.fullmatch(str(payload.get("commit_sha") or "")):
        raise RoutingProviderAuthorizationError("routing provider commit_sha is invalid")
    binding = ProviderBindingIdentity.from_mapping(payload.get("binding") or {})
    # Reconstructing the typed identity runs its field checks at the same seam
    # used by the model contract.  It also prevents extra caller fields.
    if (
        set(payload.get("binding") or {}) != set(binding.to_dict())
        or validate_provider_binding_identity(binding)
    ):
        raise RoutingProviderAuthorizationError("routing provider binding fields are invalid")
    for name, maximum in (
        ("attempt", 64),
        ("credit_cap_microunits", 100_000_000),
        ("timeout_ms", 900_000),
        ("claim_generation", 2**63 - 1),
    ):
        value = payload.get(name)
        minimum = 1 if name in {"credit_cap_microunits", "timeout_ms", "claim_generation"} else 0
        if type(value) is not int or not minimum <= value <= maximum:
            raise RoutingProviderAuthorizationError(f"routing provider {name} is invalid")


def routing_provider_logical_operation_id_v2(
    *,
    experiment_hash: str,
    variant_id: str,
    unit_ref: str,
    tool_id: str,
    attempt: int,
    core_request_fingerprint: str,
    request_body_hash: str,
) -> str:
    """Derive the broker id from the signed call identity, without proof state."""

    for name, value in (
        ("experiment_hash", experiment_hash),
        ("core_request_fingerprint", core_request_fingerprint),
        ("request_body_hash", request_body_hash),
    ):
        _hash(value, name)
    for name, value in (("variant_id", variant_id), ("unit_ref", unit_ref), ("tool_id", tool_id)):
        _ref(value, name)
    if type(attempt) is not int or attempt < 0:
        raise RoutingProviderAuthorizationError("routing provider attempt is invalid")
    return sha256_json(
        {
            "schema_version": "leadpoet.routing_provider_logical_operation.v2",
            "experiment_hash": experiment_hash,
            "variant_id": variant_id,
            "unit_ref": unit_ref,
            "tool_id": tool_id,
            "attempt": attempt,
            "core_request_fingerprint": core_request_fingerprint,
            "request_body_hash": request_body_hash,
        }
    )


def routing_provider_dispatch_job_id_v2(proof: Mapping[str, Any]) -> str:
    """Derive the protected dispatch job from one signed authorization proof.

    The authorization job remains the signer of the parent grant. The
    dispatch job is a separate deterministic execution scope so its direct
    Supabase reservation and Deepline transport attempts can both be committed
    to the dispatch receipt without relabeling either transport.
    """

    if not isinstance(proof, Mapping):
        raise RoutingProviderAuthorizationError(
            "routing provider dispatch proof is invalid"
        )
    receipt = proof.get("authorization_receipt")
    if not isinstance(receipt, Mapping):
        raise RoutingProviderAuthorizationError(
            "routing provider dispatch receipt is invalid"
        )
    values = {
        "authorization_hash": str(proof.get("authorization_hash") or ""),
        "authorization_proof_hash": str(
            proof.get("authorization_proof_hash") or ""
        ),
        "authorization_receipt_hash": str(receipt.get("receipt_hash") or ""),
    }
    for name, value in values.items():
        _hash(value, name)
    return (
        "routing-dispatch:"
        + sha256_json(
            {
                "schema_version": "leadpoet.routing_provider_dispatch_job.v3",
                **values,
            }
        ).split(":", 1)[1][:32]
    )


def execute_routing_provider_call_authorization_v2(
    payload: Mapping[str, Any],
    *,
    authorization_job_id: str,
) -> dict[str, Any]:
    """Build the protected result for the actual authorization execution job.

    ``payload`` is signed before the protected scorer receives it and therefore
    cannot contain the scorer's ExecutionJobManager job ID without creating a
    circular identity.  The scorer supplies its measured ``context.job_id`` at
    this seam.  It is included in the signed result and subsequently bound to
    the standard execution receipt and broker request.
    """

    validate_routing_provider_call_authorization_v2(payload)
    authorization_job_id = _ref(authorization_job_id, "authorization_job_id")
    authorization_hash = sha256_json(dict(payload))
    result = {
        "schema_version": ROUTING_PROVIDER_AUTHORIZATION_RESULT_SCHEMA_V2,
        "attested": True,
        "operation": ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
        "purpose": ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
        "authorization_hash": authorization_hash,
        "experiment_hash": payload["experiment_hash"],
        "admission_job_id": payload["admission_job_id"],
        "authorization_job_id": authorization_job_id,
        "experiment_id": payload["experiment_id"],
        "purpose": payload["purpose"],
        "envelope_hash": payload["envelope_hash"],
        "admission_bundle_hash": payload["admission_bundle_hash"],
        "protected_release_hash": payload["protected_release_hash"],
        "protected_boot_identity_hash": payload["protected_boot_identity_hash"],
        "variant_id": payload["variant_id"],
        "binding_id": payload["binding"]["binding_id"],
        "action_id": payload["action_id"],
        "attempt": payload["attempt"],
        "request_body_hash": payload["request_body_hash"],
        "transport_id": payload["transport_id"],
        "retry_policy_hash": payload["retry_policy_hash"],
        "credit_cap_microunits": payload["credit_cap_microunits"],
        "timeout_ms": payload["timeout_ms"],
        "claim_generation": payload["claim_generation"],
        "claim_fence_hash": payload["claim_fence_hash"],
        "binding_catalog_manifest_hash": payload["binding_catalog_manifest_hash"],
        "binding_catalog_version": payload["binding_catalog_version"],
        "unit_dataset_manifest_hash": payload["unit_dataset_manifest_hash"],
        "unit_set_hash": payload["unit_set_hash"],
        "model_binding_observation_receipt_hash": payload["model_binding_observation_receipt_hash"],
    }
    result["output_root"] = sha256_json(result)
    return result


def build_routing_provider_authorization_request_v2(
    *,
    authorization: RoutingProviderCallAuthorizationV2,
    artifact_lineage: Any,
    model_binding_observation: Any,
    execution_envelope: Any,
    admission_bundle: Any,
    prepared_call: Any,
    protected_release_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete protected pre-dispatch authorization input."""

    from gateway.research_lab.routing_admission import RoutingAdmissionBundleV2
    from gateway.research_lab.routing_execution_envelope import (
        RoutingExperimentExecutionEnvelopeV2,
    )
    from gateway.research_lab.routing_experiment_artifacts import (
        VerifiedRoutingArtifactLineage,
    )
    from gateway.research_lab.routing_model_binding_observation import (
        VerifiedRoutingModelBindingRequirements,
    )
    from gateway.research_lab.routing_provider_bindings import (
        PreparedRoutingProviderCall,
    )

    if (
        type(artifact_lineage) is not VerifiedRoutingArtifactLineage
        or type(model_binding_observation)
        is not VerifiedRoutingModelBindingRequirements
        or type(execution_envelope) is not RoutingExperimentExecutionEnvelopeV2
        or type(admission_bundle) is not RoutingAdmissionBundleV2
        or type(prepared_call) is not PreparedRoutingProviderCall
        or not isinstance(protected_release_receipt, Mapping)
    ):
        raise RoutingProviderAuthorizationError(
            "routing provider authorization request authorities are invalid"
        )
    return {
        "schema_version": ROUTING_PROVIDER_AUTHORIZATION_REQUEST_SCHEMA_V2,
        "authorization": authorization.to_dict(),
        "artifact_lineage": artifact_lineage.to_dict(),
        "model_binding_observation": model_binding_observation.to_attested_dict(),
        "execution_envelope": execution_envelope.to_dict(),
        "admission_bundle": admission_bundle.to_dict(),
        "prepared_call": asdict(prepared_call),
        "protected_release_receipt": dict(protected_release_receipt),
    }


def validate_routing_provider_authorization_request_v2(
    payload: Mapping[str, Any],
    *,
    artifact_lineage: Any,
    binding_catalog: Any,
    unit_dataset: Any,
) -> tuple[
    RoutingProviderCallAuthorizationV2,
    Any,
    Any,
    Any,
    Mapping[str, Any],
]:
    """Parse and validate a complete request against protected authorities."""

    from gateway.research_lab.routing_admission import RoutingAdmissionBundleV2
    from gateway.research_lab.routing_execution_envelope import (
        RoutingExperimentExecutionEnvelopeV2,
    )
    from gateway.research_lab.routing_model_binding_observation import (
        VerifiedRoutingModelBindingRequirements,
    )
    from gateway.research_lab.routing_provider_bindings import (
        PreparedRoutingProviderCall,
    )
    from leadpoet_canonical.attested_v2 import validate_signed_execution_receipt

    expected = {
        "schema_version",
        "authorization",
        "artifact_lineage",
        "model_binding_observation",
        "execution_envelope",
        "admission_bundle",
        "prepared_call",
        "protected_release_receipt",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema_version")
        != ROUTING_PROVIDER_AUTHORIZATION_REQUEST_SCHEMA_V2
    ):
        raise RoutingProviderAuthorizationError(
            "routing provider authorization request fields are invalid"
        )
    try:
        grant = RoutingProviderCallAuthorizationV2.from_mapping(
            payload["authorization"]
        )
        observation = VerifiedRoutingModelBindingRequirements.from_attested_mapping(
            payload["model_binding_observation"]
        )
        envelope = RoutingExperimentExecutionEnvelopeV2.from_mapping(
            payload["execution_envelope"]
        )
        admission = RoutingAdmissionBundleV2.from_mapping(
            payload["admission_bundle"]
        )
        prepared = PreparedRoutingProviderCall.from_mapping(
            payload["prepared_call"]
        )
        protected_receipt = payload["protected_release_receipt"]
        if not isinstance(protected_receipt, Mapping):
            raise TypeError("protected receipt is not an object")
        validate_signed_execution_receipt(protected_receipt)
    except Exception as exc:  # noqa: BLE001 - protected boundary conversion
        raise RoutingProviderAuthorizationError(
            "routing provider authorization request is invalid"
        ) from exc
    if dict(payload["artifact_lineage"]) != artifact_lineage.to_dict():
        raise RoutingProviderAuthorizationError(
            "routing provider authorization artifact authority differs"
        )
    if (
        protected_receipt.get("receipt_hash") != admission.protected_receipt_hash
        or protected_receipt.get("role") != "gateway_scoring"
        or protected_receipt.get("purpose")
        != ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2
        or protected_receipt.get("status") != "succeeded"
        or protected_receipt.get("job_id") != admission.job_id
        or protected_receipt.get("commit_sha") != admission.protected_commit_sha
        or protected_receipt.get("config_hash") != admission.protected_config_hash
        or protected_receipt.get("pcr0") != admission.protected_pcr0
        or protected_receipt.get("build_manifest_hash")
        != admission.protected_build_manifest_hash
        or protected_receipt.get("dependency_lock_hash")
        != admission.protected_dependency_lock_hash
        or protected_receipt.get("boot_identity_hash")
        != admission.protected_boot_identity_hash
        or protected_receipt.get("enclave_pubkey")
        != admission.protected_enclave_pubkey
    ):
        raise RoutingProviderAuthorizationError(
            "routing provider protected release identity differs"
        )
    release = {
        "protected_receipt_hash": admission.protected_receipt_hash,
        "protected_commit_sha": admission.protected_commit_sha,
        "protected_pcr0": admission.protected_pcr0,
        "protected_build_manifest_hash": admission.protected_build_manifest_hash,
        "protected_dependency_lock_hash": admission.protected_dependency_lock_hash,
        "protected_config_hash": admission.protected_config_hash,
        "protected_boot_identity_hash": admission.protected_boot_identity_hash,
        "protected_enclave_pubkey": admission.protected_enclave_pubkey,
    }
    if admission.protected_release_hash != sha256_json(
        {"schema_version": "leadpoet.routing_protected_release.v2", **release}
    ):
        raise RoutingProviderAuthorizationError(
            "routing provider protected release hash differs"
        )
    validate_routing_provider_authorization_context_v2(
        grant,
        artifact_lineage=artifact_lineage,
        binding_catalog=binding_catalog,
        unit_dataset=unit_dataset,
        model_binding_observation=observation,
        execution_envelope=envelope,
        admission_bundle=admission,
        prepared_call=prepared,
    )
    return grant, observation, envelope, admission, dict(protected_receipt)


def validate_routing_provider_authorization_context_v2(
    authorization: RoutingProviderCallAuthorizationV2 | Mapping[str, Any],
    *,
    artifact_lineage: Any,
    binding_catalog: Any,
    unit_dataset: Any,
    model_binding_observation: Any,
    execution_envelope: Any,
    admission_bundle: Any,
    prepared_call: Any,
) -> None:
    """Validate a call grant against the protected routing authorities.

    ``execute_routing_provider_call_authorization_v2`` validates only the
    grant's own schema.  That is necessary, but not sufficient: a caller can
    otherwise construct a self-consistent grant for a different artifact,
    catalog, unit, observation, or compiled call.  The protected scorer calls
    this seam with the authority objects loaded by its composition.  It is
    intentionally strict about their concrete types and rebuilds all hashes
    from those objects before the grant is signed.

    This function performs no provider I/O and does not accept caller-supplied
    hashes as authority.  Signed artifact resolution remains the composition's
    responsibility; this seam requires the resulting immutable lineage object.
    """

    # Imports are local to keep the low-level grant schema independent of the
    # larger routing authority graph and avoid an import cycle at process boot.
    from gateway.research_lab.routing_admission import RoutingAdmissionBundleV2
    from gateway.research_lab.routing_execution_envelope import (
        RoutingExperimentExecutionEnvelopeV2,
    )
    from gateway.research_lab.routing_experiment_artifacts import (
        VerifiedRoutingArtifactLineage,
    )
    from gateway.research_lab.routing_model_binding_observation import (
        VerifiedRoutingModelBindingRequirements,
    )
    from gateway.research_lab.routing_provider_bindings import (
        DEEPLINE_ACTION_POLICIES,
        PreparedRoutingProviderCall,
        VerifiedRoutingBindingCatalog,
        VerifiedRoutingUnitDataset,
    )

    try:
        grant = (
            authorization
            if isinstance(authorization, RoutingProviderCallAuthorizationV2)
            else RoutingProviderCallAuthorizationV2.from_mapping(authorization)
        )
    except Exception as exc:  # noqa: BLE001 - protected boundary conversion
        raise RoutingProviderAuthorizationError(
            "routing provider authorization is invalid"
        ) from exc
    if (
        type(artifact_lineage) is not VerifiedRoutingArtifactLineage
        or type(binding_catalog) is not VerifiedRoutingBindingCatalog
        or type(unit_dataset) is not VerifiedRoutingUnitDataset
        or type(model_binding_observation)
        is not VerifiedRoutingModelBindingRequirements
        or type(execution_envelope) is not RoutingExperimentExecutionEnvelopeV2
        or type(admission_bundle) is not RoutingAdmissionBundleV2
        or type(prepared_call) is not PreparedRoutingProviderCall
    ):
        raise RoutingProviderAuthorizationError(
            "routing provider authority objects are not verified"
        )

    def same(actual: Any, expected: Any, name: str) -> None:
        if actual != expected:
            raise RoutingProviderAuthorizationError(
                f"routing provider authorization {name} differs"
            )

    # The model lineage is immutable and must be the object resolved by the
    # signed artifact authority before this helper is entered.
    lineage_values = {
        "artifact_lineage_hash": artifact_lineage.identity_hash(),
        "pointer_document_hash": artifact_lineage.pointer_document_hash,
        "model_artifact_hash": artifact_lineage.model_artifact_hash,
        "manifest_hash": artifact_lineage.manifest_hash,
        "image_digest": artifact_lineage.image_digest,
        "commit_sha": artifact_lineage.commit_sha,
        "build_id": artifact_lineage.build_id,
        "routing_contract_hash": artifact_lineage.routing_contract_hash,
        "routing_catalog_hash": artifact_lineage.routing_catalog_hash,
        "routing_policy_hash": artifact_lineage.routing_policy_hash,
        "feature_schema_hash": artifact_lineage.feature_schema_hash,
        "verifier_contract_hash": artifact_lineage.verifier_contract_hash,
    }
    for name, value in lineage_values.items():
        same(getattr(grant, name), value, name)

    envelope_hash = execution_envelope.envelope_hash()
    same(grant.envelope_hash, envelope_hash, "envelope_hash")
    same(grant.experiment_hash, execution_envelope.experiment_hash, "experiment_hash")
    same(
        execution_envelope.artifact_lineage_hash,
        artifact_lineage.identity_hash(),
        "envelope_artifact_lineage",
    )
    same(
        execution_envelope.pointer_document_hash,
        artifact_lineage.pointer_document_hash,
        "envelope_pointer_document",
    )

    observation = model_binding_observation
    same(
        observation.artifact_lineage_hash,
        artifact_lineage.identity_hash(),
        "model_observation_artifact_lineage",
    )
    same(
        execution_envelope.model_binding_observation,
        observation.to_attested_dict(),
        "model_binding_observation",
    )
    same(
        grant.model_binding_observation_receipt_hash,
        observation.observation_receipt_hash,
        "model_binding_observation_receipt_hash",
    )
    same(
        execution_envelope.model_binding_observation_receipt_hash,
        observation.observation_receipt_hash,
        "envelope_model_observation_receipt",
    )

    # The admission bundle is itself a content-addressed identity.  Every
    # component hash is checked as well, so a caller cannot pair a valid bundle
    # hash with a different authority object.
    same(grant.admission_bundle_hash, admission_bundle.identity_hash(), "admission_bundle_hash")
    same(grant.admission_job_id, admission_bundle.job_id, "admission_job_id")
    same(grant.experiment_id, admission_bundle.experiment_id, "experiment_id")
    same(
        grant.protected_release_hash,
        admission_bundle.protected_release_hash,
        "protected_release_hash",
    )
    same(
        grant.protected_boot_identity_hash,
        admission_bundle.protected_boot_identity_hash,
        "protected_boot_identity_hash",
    )
    for name, actual, expected in (
        ("admission_artifact_lineage", admission_bundle.artifact_lineage_hash, artifact_lineage.identity_hash()),
        ("admission_pointer_document", admission_bundle.pointer_document_hash, artifact_lineage.pointer_document_hash),
        ("admission_model_artifact", admission_bundle.model_artifact_hash, artifact_lineage.model_artifact_hash),
        ("admission_manifest", admission_bundle.immutable_manifest_hash, artifact_lineage.manifest_hash),
        ("admission_unit_dataset", admission_bundle.unit_dataset_manifest_hash, unit_dataset.manifest_hash),
        ("admission_unit_set", admission_bundle.unit_set_hash, unit_dataset.unit_set_hash),
        ("admission_catalog", admission_bundle.binding_catalog_manifest_hash, binding_catalog.manifest_hash),
        ("admission_catalog_version", admission_bundle.binding_catalog_version, binding_catalog.catalog_version),
        ("admission_model_observation", admission_bundle.model_binding_observation_hash, sha256_json(dict(observation.result))),
        ("admission_model_observation_receipt", admission_bundle.model_binding_observation_receipt_hash, observation.observation_receipt_hash),
    ):
        same(actual, expected, name)
    same(admission_bundle.envelope_hash, envelope_hash, "admission_envelope")
    same(admission_bundle.experiment_hash, execution_envelope.experiment_hash, "admission_experiment")

    # Resolve the exact binding and unit through the reviewed signed
    # authorities.  The prepared call is not trusted merely because its own
    # dataclass validation passed.
    try:
        manifest = binding_catalog.resolve(grant.binding)
        unit_input, unit_input_hash = unit_dataset.resolve(grant.unit_ref)
    except Exception as exc:  # noqa: BLE001 - authority boundary
        raise RoutingProviderAuthorizationError(
            "routing provider binding or unit is not reviewed"
        ) from exc
    same(grant.binding_catalog_manifest_hash, binding_catalog.manifest_hash, "binding_catalog_manifest_hash")
    same(grant.binding_catalog_version, binding_catalog.catalog_version, "binding_catalog_version")
    same(grant.unit_dataset_manifest_hash, unit_dataset.manifest_hash, "unit_dataset_manifest_hash")
    same(grant.unit_set_hash, unit_dataset.unit_set_hash, "unit_set_hash")
    same(grant.unit_input_hash, unit_input_hash, "unit_input_hash")
    same(grant.binding, prepared_call.binding, "binding")
    same(grant.action_id, prepared_call.action_id, "action_id")
    same(grant.transport_id, prepared_call.transport_id, "transport_id")
    same(grant.unit_ref, prepared_call.unit_ref, "unit_ref")
    same(grant.request_body_hash, prepared_call.request_body_hash, "request_body_hash")
    same(grant.retry_policy_hash, prepared_call.retry_policy_hash, "retry_policy_hash")
    same(grant.credit_cap_microunits, prepared_call.credit_ceiling_microunits, "credit_cap_microunits")
    same(grant.timeout_ms, prepared_call.timeout_ms, "timeout_ms")
    same(prepared_call.binding_manifest_hash, grant.binding.manifest_hash, "prepared_binding_manifest")
    same(prepared_call.binding_catalog_manifest_hash, binding_catalog.manifest_hash, "prepared_binding_catalog")
    same(prepared_call.binding_catalog_version, binding_catalog.catalog_version, "prepared_binding_catalog_version")
    same(prepared_call.unit_dataset_manifest_hash, unit_dataset.manifest_hash, "prepared_unit_dataset")
    same(prepared_call.unit_set_hash, unit_dataset.unit_set_hash, "prepared_unit_set")
    same(prepared_call.unit_input_hash, unit_input_hash, "prepared_unit_input")
    same(
        prepared_call.model_binding_requirements_hash,
        observation.resolve(
            binding=grant.binding,
            artifact_lineage_hash=artifact_lineage.identity_hash(),
        ),
        "prepared_model_binding_requirements",
    )
    same(prepared_call.model_binding_requirements_hash, manifest.model_binding_requirements_hash, "catalog_model_binding_requirements")
    same(prepared_call.action_id, manifest.action_id, "catalog_action_id")
    same(prepared_call.transport_id, manifest.transport_id, "catalog_transport_id")
    same(prepared_call.provider, manifest.binding.provider_id, "catalog_provider_id")
    same(prepared_call.operation, DEEPLINE_ACTION_POLICIES[prepared_call.action_id].operation, "catalog_operation")
    same(
        sha256_json(
            {
                "provider": prepared_call.provider,
                "operation": prepared_call.operation,
                "payload": dict(prepared_call.payload),
            }
        ),
        prepared_call.request_body_hash,
        "prepared_request_body",
    )
    same(
        sha256_json(
            {
                "schema_version": "leadpoet.routing_validation_context.v1",
                "action_id": prepared_call.action_id,
                "context": dict(prepared_call.validation_context),
            }
        ),
        prepared_call.validation_context_hash,
        "prepared_validation_context",
    )
    if grant.binding.binding_id not in set(admission_bundle.binding_ids):
        raise RoutingProviderAuthorizationError(
            "routing provider authorization binding is not admitted"
        )
    if not isinstance(unit_input, Mapping):
        raise RoutingProviderAuthorizationError("routing provider unit input is invalid")


__all__ = [
    "ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2",
    "ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2",
    "ROUTING_PROVIDER_AUTHORIZATION_SCHEMA_V2",
    "ROUTING_PROVIDER_AUTHORIZATION_RESULT_SCHEMA_V2",
    "ROUTING_PROVIDER_AUTHORIZATION_REQUEST_SCHEMA_V2",
    "RoutingProviderAuthorizationError",
    "RoutingProviderCallAuthorizationV2",
    "validate_routing_provider_call_authorization_v2",
    "routing_provider_logical_operation_id_v2",
    "routing_provider_dispatch_job_id_v2",
    "execute_routing_provider_call_authorization_v2",
    "build_routing_provider_authorization_request_v2",
    "validate_routing_provider_authorization_request_v2",
    "validate_routing_provider_authorization_context_v2",
]
