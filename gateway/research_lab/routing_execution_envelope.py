"""Immutable measured-execution identity for one routing experiment.

The model contract owns tool and provider identities.  The Lab runtime owns
the reviewed transport catalog, immutable unit dataset, and exact action
compiler.  This envelope binds those runtime authorities to one experiment
without adding transport-only fields to ``ProviderBindingIdentity``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping

from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
    VerifiedRoutingGoldLabels,
)
from gateway.research_lab.routing_provider_bindings import (
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
    routing_model_binding_identity_hash,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import RoutingExperimentV2Spec


ROUTING_EXECUTION_ENVELOPE_SCHEMA_V2 = (
    "leadpoet.research_lab.routing_execution_envelope.v2"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")


class RoutingExecutionEnvelopeError(ValueError):
    """The runtime authority envelope is incomplete or inconsistent."""


def _hash(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(text):
        raise RoutingExecutionEnvelopeError(f"routing execution {name} is invalid")
    return text


def _ref(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not _REF_RE.fullmatch(text):
        raise RoutingExecutionEnvelopeError(f"routing execution {name} is invalid")
    return text


@dataclass(frozen=True)
class RoutingExecutionBindingV2:
    """One model binding resolved through the signed runtime catalog."""

    binding_id: str
    provider_id: str
    tool_id: str
    binding_manifest_hash: str
    action_id: str
    compiler_family: str
    transport_id: str
    model_binding_requirements_hash: str
    output_contract_hash: str
    evidence_contract_hash: str
    retry_policy_hash: str
    credit_ceiling_microunits: int
    timeout_ms: int

    def __post_init__(self) -> None:
        for name in (
            "binding_id", "provider_id", "tool_id", "action_id",
            "compiler_family", "transport_id",
        ):
            _ref(getattr(self, name), name)
        for name in (
            "binding_manifest_hash", "model_binding_requirements_hash",
            "output_contract_hash", "evidence_contract_hash",
            "retry_policy_hash",
        ):
            _hash(getattr(self, name), name)
        if (
            type(self.credit_ceiling_microunits) is not int
            or not 1 <= self.credit_ceiling_microunits <= 100_000_000
            or type(self.timeout_ms) is not int
            or not 1 <= self.timeout_ms <= 900_000
        ):
            raise RoutingExecutionEnvelopeError(
                "routing execution binding limits are invalid"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoutingExecutionBindingV2":
        if not isinstance(value, Mapping) or set(value) != set(cls.__annotations__):
            raise RoutingExecutionEnvelopeError(
                "routing execution binding fields are invalid"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class RoutingExperimentExecutionEnvelopeV2:
    """Exact signed runtime authorities admitted for one immutable spec."""

    experiment_hash: str
    artifact_lineage_hash: str
    pointer_document_hash: str
    binding_catalog_manifest_hash: str
    binding_catalog_version: str
    unit_dataset_manifest_hash: str
    unit_set_hash: str
    gold_label_manifest_hash: str
    model_binding_observation_receipt_hash: str
    model_binding_observation: Mapping[str, Any]
    bindings: tuple[RoutingExecutionBindingV2, ...]
    schema_version: str = ROUTING_EXECUTION_ENVELOPE_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema_version != ROUTING_EXECUTION_ENVELOPE_SCHEMA_V2:
            raise RoutingExecutionEnvelopeError(
                "routing execution envelope schema is invalid"
            )
        for name in (
            "experiment_hash", "artifact_lineage_hash", "pointer_document_hash",
            "binding_catalog_manifest_hash", "unit_dataset_manifest_hash",
            "unit_set_hash", "gold_label_manifest_hash",
            "model_binding_observation_receipt_hash",
        ):
            _hash(getattr(self, name), name)
        _ref(self.binding_catalog_version, "binding catalog version")
        try:
            observation = VerifiedRoutingModelBindingRequirements.from_attested_mapping(
                self.model_binding_observation
            )
        except ValueError as exc:
            raise RoutingExecutionEnvelopeError(
                "routing execution model binding observation is invalid"
            ) from exc
        if (
            observation.observation_receipt_hash
            != self.model_binding_observation_receipt_hash
            or observation.artifact_lineage_hash != self.artifact_lineage_hash
        ):
            raise RoutingExecutionEnvelopeError(
                "routing execution model binding observation differs"
            )
        object.__setattr__(
            self,
            "model_binding_observation",
            MappingProxyType(observation.to_attested_dict()),
        )
        if not self.bindings:
            raise RoutingExecutionEnvelopeError(
                "routing execution envelope has no bindings"
            )
        ordered = tuple(sorted(self.bindings, key=lambda item: item.binding_id))
        if ordered != self.bindings or len({item.binding_id for item in ordered}) != len(ordered):
            raise RoutingExecutionEnvelopeError(
                "routing execution envelope bindings are not canonical"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_hash": self.experiment_hash,
            "artifact_lineage_hash": self.artifact_lineage_hash,
            "pointer_document_hash": self.pointer_document_hash,
            "binding_catalog_manifest_hash": self.binding_catalog_manifest_hash,
            "binding_catalog_version": self.binding_catalog_version,
            "unit_dataset_manifest_hash": self.unit_dataset_manifest_hash,
            "unit_set_hash": self.unit_set_hash,
            "gold_label_manifest_hash": self.gold_label_manifest_hash,
            "model_binding_observation_receipt_hash": (
                self.model_binding_observation_receipt_hash
            ),
            "model_binding_observation": {
                "result": dict(self.model_binding_observation["result"]),
                "receipt": dict(self.model_binding_observation["receipt"]),
            },
            "bindings": [item.to_dict() for item in self.bindings],
        }

    def envelope_hash(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "RoutingExperimentExecutionEnvelopeV2":
        expected = {
            "schema_version", "experiment_hash", "artifact_lineage_hash",
            "pointer_document_hash", "binding_catalog_manifest_hash",
            "binding_catalog_version", "unit_dataset_manifest_hash",
            "unit_set_hash", "gold_label_manifest_hash",
            "model_binding_observation_receipt_hash",
            "model_binding_observation", "bindings",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise RoutingExecutionEnvelopeError(
                "routing execution envelope fields are invalid"
            )
        rows = value.get("bindings")
        if not isinstance(rows, list):
            raise RoutingExecutionEnvelopeError(
                "routing execution envelope bindings are invalid"
            )
        return cls(
            schema_version=str(value["schema_version"]),
            experiment_hash=str(value["experiment_hash"]),
            artifact_lineage_hash=str(value["artifact_lineage_hash"]),
            pointer_document_hash=str(value["pointer_document_hash"]),
            binding_catalog_manifest_hash=str(value["binding_catalog_manifest_hash"]),
            binding_catalog_version=str(value["binding_catalog_version"]),
            unit_dataset_manifest_hash=str(value["unit_dataset_manifest_hash"]),
            unit_set_hash=str(value["unit_set_hash"]),
            gold_label_manifest_hash=str(value["gold_label_manifest_hash"]),
            model_binding_observation_receipt_hash=str(
                value["model_binding_observation_receipt_hash"]
            ),
            model_binding_observation=dict(value["model_binding_observation"]),
            bindings=tuple(RoutingExecutionBindingV2.from_mapping(item) for item in rows),
        )


def build_routing_execution_envelope_v2(
    *,
    spec: RoutingExperimentV2Spec,
    artifact_lineage: VerifiedRoutingArtifactLineage,
    binding_catalog: VerifiedRoutingBindingCatalog,
    unit_dataset: VerifiedRoutingUnitDataset,
    gold_labels: VerifiedRoutingGoldLabels,
    model_binding_observation: VerifiedRoutingModelBindingRequirements,
) -> RoutingExperimentExecutionEnvelopeV2:
    """Resolve every spec binding through the already verified authorities."""

    baseline = next(
        (
            variant
            for variant in spec.variants
            if variant.variant_id == spec.baseline_variant_id
        ),
        None,
    )
    if baseline is None or baseline.artifact.to_dict() != (
        artifact_lineage.sourcing_model_identity().to_dict()
    ):
        raise RoutingExecutionEnvelopeError(
            "routing execution baseline artifact differs from the release"
        )
    if spec.input.unit_input_set_hash != unit_dataset.unit_set_hash:
        raise RoutingExecutionEnvelopeError(
            "routing execution unit dataset differs from the spec"
        )
    if spec.input.gold_label_set_hash != gold_labels.label_set_hash:
        raise RoutingExecutionEnvelopeError(
            "routing execution gold labels differ from the spec"
        )
    rows = []
    for binding in sorted(spec.provider_bindings, key=lambda item: item.binding_id):
        manifest = binding_catalog.resolve(binding)
        rows.append(
            RoutingExecutionBindingV2(
                binding_id=binding.binding_id,
                provider_id=binding.provider_id,
                tool_id=binding.tool_id,
                binding_manifest_hash=manifest.binding.manifest_hash,
                action_id=manifest.action_id,
                compiler_family=manifest.compiler_family,
                transport_id=manifest.transport_id,
                model_binding_requirements_hash=manifest.model_binding_requirements_hash,
                output_contract_hash=manifest.output_contract_hash,
                evidence_contract_hash=manifest.evidence_contract_hash,
                retry_policy_hash=manifest.retry_policy_hash,
                credit_ceiling_microunits=manifest.credit_ceiling_microunits,
                timeout_ms=manifest.timeout_ms,
            )
        )
    envelope = RoutingExperimentExecutionEnvelopeV2(
        experiment_hash=spec.experiment_hash(),
        artifact_lineage_hash=artifact_lineage.identity_hash(),
        pointer_document_hash=artifact_lineage.pointer_document_hash,
        binding_catalog_manifest_hash=binding_catalog.manifest_hash,
        binding_catalog_version=binding_catalog.catalog_version,
        unit_dataset_manifest_hash=unit_dataset.manifest_hash,
        unit_set_hash=unit_dataset.unit_set_hash,
        gold_label_manifest_hash=gold_labels.manifest_hash,
        model_binding_observation_receipt_hash=(
            model_binding_observation.observation_receipt_hash
        ),
        model_binding_observation=model_binding_observation.to_attested_dict(),
        bindings=tuple(rows),
    )
    validate_routing_execution_envelope_v2(
        spec=spec,
        envelope=envelope,
        binding_catalog=binding_catalog,
    )
    return envelope


def validate_routing_execution_envelope_v2(
    *,
    spec: RoutingExperimentV2Spec,
    envelope: RoutingExperimentExecutionEnvelopeV2,
    binding_catalog: VerifiedRoutingBindingCatalog | None = None,
) -> None:
    """Cross-check the Lab-owned envelope with the exact model-owned spec."""

    if envelope.experiment_hash != spec.experiment_hash():
        raise RoutingExecutionEnvelopeError(
            "routing execution envelope experiment hash differs"
        )
    if envelope.unit_set_hash != spec.input.unit_input_set_hash:
        raise RoutingExecutionEnvelopeError(
            "routing execution envelope unit set differs"
        )
    model_bindings = {item.binding_id: item for item in spec.provider_bindings}
    try:
        observation = VerifiedRoutingModelBindingRequirements.from_attested_mapping(
            envelope.model_binding_observation
        )
    except ValueError as exc:
        raise RoutingExecutionEnvelopeError(
            "routing execution model binding observation is invalid"
        ) from exc
    expected_observation_ids = {
        routing_model_binding_identity_hash(item)
        for item in spec.provider_bindings
    }
    if (
        observation.artifact_lineage_hash != envelope.artifact_lineage_hash
        or set(observation.requirement_hash_by_binding_identity)
        != expected_observation_ids
    ):
        raise RoutingExecutionEnvelopeError(
            "routing execution model binding observation set differs"
        )
    envelope_bindings = {item.binding_id: item for item in envelope.bindings}
    if set(model_bindings) != set(envelope_bindings):
        raise RoutingExecutionEnvelopeError(
            "routing execution envelope binding set differs"
        )
    for binding_id, model_binding in model_bindings.items():
        runtime_binding = envelope_bindings[binding_id]
        if (
            runtime_binding.provider_id != model_binding.provider_id
            or runtime_binding.tool_id != model_binding.tool_id
            or runtime_binding.binding_manifest_hash != model_binding.manifest_hash
        ):
            raise RoutingExecutionEnvelopeError(
                "routing execution envelope model binding differs"
            )
        observed_requirements_hash = observation.resolve(
            binding=model_binding,
            artifact_lineage_hash=envelope.artifact_lineage_hash,
        )
        if runtime_binding.model_binding_requirements_hash != observed_requirements_hash:
            raise RoutingExecutionEnvelopeError(
                "routing execution model binding requirements differ"
            )
        ceiling = spec.credit_budget.provider_credit_ceilings.get(binding_id)
        if ceiling is None or runtime_binding.credit_ceiling_microunits > ceiling:
            raise RoutingExecutionEnvelopeError(
                "routing execution envelope binding budget differs"
            )
        if binding_catalog is not None:
            manifest = binding_catalog.resolve(model_binding)
            expected = {
                "action_id": manifest.action_id,
                "compiler_family": manifest.compiler_family,
                "transport_id": manifest.transport_id,
                "model_binding_requirements_hash": manifest.model_binding_requirements_hash,
                "output_contract_hash": manifest.output_contract_hash,
                "evidence_contract_hash": manifest.evidence_contract_hash,
                "retry_policy_hash": manifest.retry_policy_hash,
                "credit_ceiling_microunits": manifest.credit_ceiling_microunits,
                "timeout_ms": manifest.timeout_ms,
            }
            if any(getattr(runtime_binding, key) != value for key, value in expected.items()):
                raise RoutingExecutionEnvelopeError(
                    "routing execution envelope catalog binding differs"
                )


__all__ = [
    "ROUTING_EXECUTION_ENVELOPE_SCHEMA_V2",
    "RoutingExecutionEnvelopeError",
    "RoutingExecutionBindingV2",
    "RoutingExperimentExecutionEnvelopeV2",
    "build_routing_execution_envelope_v2",
    "validate_routing_execution_envelope_v2",
]
