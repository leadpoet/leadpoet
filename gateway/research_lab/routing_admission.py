"""Cryptographic admission identity for one measured routing experiment.

The routing worker may receive the individual authority documents from its
host, but a provider call must consume one immutable identity that commits to
all of them.  This module does not load credentials or call a provider.  It
only combines already verified authorities and the signed scoring-enclave
identity into a strict, hashable admission document.

Keeping this document separate from the provider-call grant is intentional:
the grant signs this exact hash, while the broker re-checks the complete
document before dispatch.  A caller therefore cannot substitute a catalog,
unit set, model observation, experiment, job, or release after grant issue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
    validate_routing_execution_envelope_v2,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
    VerifiedRoutingGoldLabels,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
)
from gateway.research_lab.routing_provider_bindings import (
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from leadpoet_canonical.attested_v2 import validate_signed_execution_receipt
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import RoutingExperimentV2Spec


ROUTING_ADMISSION_SCHEMA_V2 = "leadpoet.research_lab.routing_admission.v2"
ROUTING_ADMISSION_ROLE_V2 = "gateway_scoring"
ROUTING_ADMISSION_PURPOSE_V2 = "research_lab.routing_provider_evidence.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")


class RoutingAdmissionError(ValueError):
    """The routing admission identity is incomplete or substituted."""


def _hash(value: Any, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise RoutingAdmissionError(f"routing admission {name} is invalid")
    return normalized


def _ref(value: Any, name: str) -> str:
    normalized = str(value or "").strip()
    if not _REF_RE.fullmatch(normalized):
        raise RoutingAdmissionError(f"routing admission {name} is invalid")
    return normalized


@dataclass(frozen=True)
class RoutingAdmissionBundleV2:
    """Exact authorities and protected identity admitted for one provider call."""

    job_id: str
    experiment_id: str
    experiment_hash: str
    role: str
    purpose: str
    envelope_hash: str
    artifact_lineage_hash: str
    pointer_document_hash: str
    immutable_manifest_hash: str
    model_artifact_hash: str
    gold_label_manifest_hash: str
    gold_label_set_hash: str
    unit_dataset_manifest_hash: str
    unit_set_hash: str
    binding_catalog_manifest_hash: str
    binding_catalog_version: str
    model_binding_observation_hash: str
    model_binding_observation_receipt_hash: str
    binding_ids: tuple[str, ...]
    protected_release_hash: str
    protected_commit_sha: str
    protected_pcr0: str
    protected_build_manifest_hash: str
    protected_dependency_lock_hash: str
    protected_config_hash: str
    protected_boot_identity_hash: str
    protected_enclave_pubkey: str
    protected_receipt_hash: str
    schema_version: str = ROUTING_ADMISSION_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema_version != ROUTING_ADMISSION_SCHEMA_V2:
            raise RoutingAdmissionError("routing admission schema is invalid")
        for name in (
            "job_id", "experiment_id", "binding_catalog_version",
            "role", "purpose",
        ):
            _ref(getattr(self, name), name)
        for name in (
            "experiment_hash", "envelope_hash", "artifact_lineage_hash",
            "pointer_document_hash", "immutable_manifest_hash",
            "model_artifact_hash", "gold_label_manifest_hash",
            "gold_label_set_hash", "unit_dataset_manifest_hash", "unit_set_hash",
            "binding_catalog_manifest_hash", "model_binding_observation_hash",
            "model_binding_observation_receipt_hash", "protected_release_hash",
            "protected_build_manifest_hash", "protected_dependency_lock_hash",
            "protected_config_hash", "protected_boot_identity_hash",
            "protected_receipt_hash",
        ):
            _hash(getattr(self, name), name)
        if self.role != ROUTING_ADMISSION_ROLE_V2:
            raise RoutingAdmissionError("routing admission role is invalid")
        if self.purpose != ROUTING_ADMISSION_PURPOSE_V2:
            raise RoutingAdmissionError("routing admission purpose is invalid")
        if not _GIT_SHA_RE.fullmatch(self.protected_commit_sha):
            raise RoutingAdmissionError(
                "routing admission protected_commit_sha is invalid"
            )
        if not _PCR0_RE.fullmatch(str(self.protected_pcr0 or "")):
            raise RoutingAdmissionError("routing admission protected_pcr0 is invalid")
        if not re.fullmatch(r"^[0-9a-f]{64}$", self.protected_enclave_pubkey):
            raise RoutingAdmissionError(
                "routing admission protected_enclave_pubkey is invalid"
            )
        if not isinstance(self.binding_ids, tuple) or not self.binding_ids:
            raise RoutingAdmissionError("routing admission binding set is empty")
        if tuple(sorted(set(self.binding_ids))) != self.binding_ids:
            raise RoutingAdmissionError("routing admission binding set is not canonical")
        for item in self.binding_ids:
            _ref(item, "binding_id")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["binding_ids"] = list(self.binding_ids)
        return result

    def identity_hash(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoutingAdmissionBundleV2":
        expected = set(cls.__annotations__)
        if not isinstance(value, Mapping) or set(value) != expected:
            raise RoutingAdmissionError("routing admission fields are invalid")
        ids = value.get("binding_ids")
        if not isinstance(ids, list):
            raise RoutingAdmissionError("routing admission binding_ids are invalid")
        values = dict(value)
        values["binding_ids"] = tuple(str(item) for item in ids)
        return cls(**values)


def _protected_release_identity(
    value: Mapping[str, Any], *, job_id: str
) -> dict[str, str]:
    """Validate the complete signed scoring execution identity."""

    try:
        validate_signed_execution_receipt(value)
    except Exception as exc:
        raise RoutingAdmissionError("routing admission protected receipt is invalid") from exc
    if (
        value.get("role") != ROUTING_ADMISSION_ROLE_V2
        or value.get("purpose") != ROUTING_ADMISSION_PURPOSE_V2
        or value.get("status") != "succeeded"
        or value.get("job_id") != job_id
        or value.get("boot_identity_hash") == ""
    ):
        raise RoutingAdmissionError("routing admission protected release differs")
    if not _PCR0_RE.fullmatch(str(value.get("pcr0") or "")):
        raise RoutingAdmissionError("routing admission protected pcr0 is invalid")
    for name in (
        "build_manifest_hash",
        "dependency_lock_hash",
        "config_hash",
        "boot_identity_hash",
    ):
        _hash(value.get(name), f"protected {name}")
    release = {
        "protected_receipt_hash": str(value["receipt_hash"]),
        "protected_commit_sha": str(value["commit_sha"]),
        "protected_pcr0": str(value["pcr0"]),
        "protected_build_manifest_hash": str(value["build_manifest_hash"]),
        "protected_dependency_lock_hash": str(value["dependency_lock_hash"]),
        "protected_config_hash": str(value["config_hash"]),
        "protected_boot_identity_hash": str(value["boot_identity_hash"]),
        "protected_enclave_pubkey": str(value["enclave_pubkey"]),
    }
    release["protected_release_hash"] = sha256_json(
        {"schema_version": "leadpoet.routing_protected_release.v2", **release}
    )
    return release


def build_routing_admission_bundle_v2(
    *,
    job_id: str,
    spec: RoutingExperimentV2Spec,
    envelope: RoutingExperimentExecutionEnvelopeV2,
    artifact_lineage: VerifiedRoutingArtifactLineage,
    gold_labels: VerifiedRoutingGoldLabels,
    binding_catalog: VerifiedRoutingBindingCatalog,
    unit_dataset: VerifiedRoutingUnitDataset,
    model_binding_observation: VerifiedRoutingModelBindingRequirements,
    protected_release_receipt: Mapping[str, Any],
) -> RoutingAdmissionBundleV2:
    """Build one admission identity from already verified authorities."""

    try:
        validate_routing_execution_envelope_v2(
            spec=spec, envelope=envelope, binding_catalog=binding_catalog
        )
    except Exception as exc:
        raise RoutingAdmissionError("routing admission execution envelope is invalid") from exc
    if envelope.artifact_lineage_hash != artifact_lineage.identity_hash():
        raise RoutingAdmissionError("routing admission artifact lineage differs")
    if envelope.pointer_document_hash != artifact_lineage.pointer_document_hash:
        raise RoutingAdmissionError("routing admission pointer differs")
    if envelope.gold_label_manifest_hash != gold_labels.manifest_hash:
        raise RoutingAdmissionError("routing admission gold labels differ")
    if envelope.unit_dataset_manifest_hash != unit_dataset.manifest_hash:
        raise RoutingAdmissionError("routing admission unit dataset differs")
    if envelope.unit_set_hash != unit_dataset.unit_set_hash:
        raise RoutingAdmissionError("routing admission unit set differs")
    if envelope.binding_catalog_manifest_hash != binding_catalog.manifest_hash:
        raise RoutingAdmissionError("routing admission binding catalog differs")
    if envelope.binding_catalog_version != binding_catalog.catalog_version:
        raise RoutingAdmissionError("routing admission binding catalog version differs")
    if model_binding_observation.observation_receipt_hash != (
        envelope.model_binding_observation_receipt_hash
    ):
        raise RoutingAdmissionError("routing admission model observation differs")
    expected_ids = tuple(sorted(item.binding_id for item in spec.provider_bindings))
    if expected_ids != tuple(item.binding_id for item in envelope.bindings):
        raise RoutingAdmissionError("routing admission binding identities differ")
    release = _protected_release_identity(protected_release_receipt, job_id=job_id)
    bundle = RoutingAdmissionBundleV2(
        job_id=_ref(job_id, "job_id"),
        experiment_id=_ref(spec.experiment_id, "experiment_id"),
        experiment_hash=spec.experiment_hash(),
        role=ROUTING_ADMISSION_ROLE_V2,
        purpose=ROUTING_ADMISSION_PURPOSE_V2,
        envelope_hash=envelope.envelope_hash(),
        artifact_lineage_hash=artifact_lineage.identity_hash(),
        pointer_document_hash=artifact_lineage.pointer_document_hash,
        immutable_manifest_hash=artifact_lineage.manifest_hash,
        model_artifact_hash=artifact_lineage.model_artifact_hash,
        gold_label_manifest_hash=gold_labels.manifest_hash,
        gold_label_set_hash=gold_labels.label_set_hash,
        unit_dataset_manifest_hash=unit_dataset.manifest_hash,
        unit_set_hash=unit_dataset.unit_set_hash,
        binding_catalog_manifest_hash=binding_catalog.manifest_hash,
        binding_catalog_version=binding_catalog.catalog_version,
        # ``VerifiedRoutingModelBindingRequirements`` intentionally exposes
        # immutable mapping proxies.  Canonical hashing must serialize the
        # attested document, not the Python wrapper object.
        model_binding_observation_hash=sha256_json(
            dict(model_binding_observation.result)
        ),
        model_binding_observation_receipt_hash=(
            model_binding_observation.observation_receipt_hash
        ),
        binding_ids=expected_ids,
        **release,
    )
    return bundle


def validate_routing_admission_bundle_v2(
    *,
    bundle: RoutingAdmissionBundleV2,
    spec: RoutingExperimentV2Spec,
    envelope: RoutingExperimentExecutionEnvelopeV2,
    artifact_lineage: VerifiedRoutingArtifactLineage,
    gold_labels: VerifiedRoutingGoldLabels,
    binding_catalog: VerifiedRoutingBindingCatalog,
    unit_dataset: VerifiedRoutingUnitDataset,
    model_binding_observation: VerifiedRoutingModelBindingRequirements,
    protected_release_receipt: Mapping[str, Any],
    job_id: str,
) -> None:
    """Rebuild and compare the bundle before any provider call."""

    expected = build_routing_admission_bundle_v2(
        job_id=job_id,
        spec=spec,
        envelope=envelope,
        artifact_lineage=artifact_lineage,
        gold_labels=gold_labels,
        binding_catalog=binding_catalog,
        unit_dataset=unit_dataset,
        model_binding_observation=model_binding_observation,
        protected_release_receipt=protected_release_receipt,
    )
    if bundle.to_dict() != expected.to_dict():
        raise RoutingAdmissionError("routing admission bundle substitution detected")


__all__ = [
    "ROUTING_ADMISSION_SCHEMA_V2",
    "ROUTING_ADMISSION_ROLE_V2",
    "ROUTING_ADMISSION_PURPOSE_V2",
    "RoutingAdmissionError",
    "RoutingAdmissionBundleV2",
    "build_routing_admission_bundle_v2",
    "validate_routing_admission_bundle_v2",
]
