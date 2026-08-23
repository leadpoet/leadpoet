"""Cryptographically verified model-owned SourceAdd binding observations.

The gateway must not infer executable binding requirements from its runtime
catalog. The exact signed model sandbox emits one canonical requirement hash
for every model ``ProviderBindingIdentity``. This module accepts that result
only with a valid scoring-enclave execution receipt and exposes an immutable
lookup used by both envelope admission and provider pre-dispatch checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from leadpoet_canonical.attested_v2 import validate_signed_execution_receipt
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    validate_provider_binding_identity,
)


ROUTING_MODEL_BINDING_OBSERVATION_SCHEMA_V2 = (
    "leadpoet.routing_model_binding_observation.v2"
)
ROUTING_MODEL_BINDING_OBSERVATION_REQUEST_SCHEMA_V2 = (
    "leadpoet.routing_model_binding_observation_request.v2"
)
ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2 = (
    "research_lab.routing_model_binding_observation.v2"
)
ROUTING_MODEL_BINDING_REQUIREMENTS_SCHEMA_V2 = (
    "leadpoet.routing_model_binding_requirements.v2"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_SOURCE_ADD_MANIFEST_SCHEMA = "leadpoet.intent-source-binding-manifest:v1"

# These are the fields the probe consumes from the model runtime document.
# The model may add unrelated metadata fields, but it must not silently omit
# one of the binding-attestation fields or expose private bindings.
_MODEL_METADATA_REQUIRED_FIELDS = frozenset(
    {
        "compiler_version",
        "catalog_sha256",
        "policy_sha256",
        "source_add_manifest_attestations",
        "source_add_binding_manifests",
        "private_bindings_exposed",
    }
)
_MANIFEST_ROW_FIELDS = frozenset(
    {"tool_id", "revision", "manifest_sha256", "manifest"}
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "tool_id",
        "provider_id",
        "stage",
        "execution_mode",
        "cost_class",
        "unit_cost",
        "max_calls",
        "max_results",
        "timeout_seconds",
        "capabilities",
        "intent_categories",
        "evidence_types",
        "category_contracts",
        "binding_requirements",
    }
)
_CATEGORY_CONTRACT_FIELDS = frozenset(
    {"category", "capabilities", "evidence_types", "requirements"}
)
_ATTESTATION_FIELDS = frozenset({"tool_id", "revision", "manifest_sha256"})
_STAGES = frozenset({"candidate_acquisition", "intent_evidence"})
_SAFE_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_-]{1,79}$")


class RoutingModelBindingObservationError(ValueError):
    """The model binding observation or its signed receipt is not exact."""


def _hash(value: Any, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise RoutingModelBindingObservationError(
            f"routing model binding {name} is invalid"
        )
    return normalized


def _raw_or_prefixed_hash(value: Any, name: str) -> str:
    """Normalize model raw SHA-256 values and Lab ``sha256:`` values."""

    normalized = str(value or "").strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[7:]
    if not _RAW_HASH_RE.fullmatch(normalized):
        raise RoutingModelBindingObservationError(
            f"routing model binding {name} is invalid"
        )
    return normalized


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_TEXT_RE.fullmatch(value):
        raise RoutingModelBindingObservationError(
            f"routing model binding {name} is invalid"
        )
    return value


def _require_string_list(value: Any, name: str, *, maximum: int = 64) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise RoutingModelBindingObservationError(
            f"routing model binding {name} is invalid"
        )
    result = [_require_text(item, name) for item in value]
    if len(set(result)) != len(result):
        raise RoutingModelBindingObservationError(
            f"routing model binding {name} contains duplicates"
        )
    return result


def _source_add_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Recompute the model's public binding-manifest digest.

    The pinned model uses sorted, compact JSON and a raw hexadecimal digest
    for ``manifest_sha256``.  Recomputing it prevents a row from substituting
    a different manifest while retaining the original attestation value.
    """

    rendered = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def routing_model_binding_requirements_hash(
    binding_manifest: Mapping[str, Any],
) -> str:
    """Hash model-owned binding requirements using one closed schema.

    Requirements are sorted only for this derived hash.  The source model's
    manifest order remains attested separately by its manifest digest.
    """

    if not isinstance(binding_manifest, Mapping):
        raise RoutingModelBindingObservationError(
            "routing model binding manifest is invalid"
        )
    requirements = binding_manifest.get("binding_requirements")
    normalized = _require_string_list(
        requirements,
        "binding requirements",
        maximum=32,
    )
    payload = {
        "schema_version": ROUTING_MODEL_BINDING_REQUIREMENTS_SCHEMA_V2,
        "binding_requirements": sorted(normalized),
    }
    return sha256_json(payload)


def _validate_binding_manifest_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != _MANIFEST_ROW_FIELDS:
        raise RoutingModelBindingObservationError(
            "routing model source-add binding manifest row is invalid"
        )
    manifest = row.get("manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS:
        raise RoutingModelBindingObservationError(
            "routing model source-add binding manifest is invalid"
        )
    if manifest.get("schema_version") != _SOURCE_ADD_MANIFEST_SCHEMA:
        raise RoutingModelBindingObservationError(
            "routing model source-add binding manifest schema is invalid"
        )
    tool_id = _require_text(manifest.get("tool_id"), "manifest tool id")
    provider_id = manifest.get("provider_id")
    if not isinstance(provider_id, str) or not _SAFE_PROVIDER_RE.fullmatch(provider_id):
        raise RoutingModelBindingObservationError(
            "routing model source-add provider id is invalid"
        )
    stage = manifest.get("stage")
    if stage not in _STAGES:
        raise RoutingModelBindingObservationError(
            "routing model source-add stage is invalid"
        )
    expected_tool_id = (
        ("candidate" if stage == "candidate_acquisition" else "intent")
        + ".source_add."
        + provider_id
    )
    if tool_id != expected_tool_id:
        raise RoutingModelBindingObservationError(
            "routing model source-add tool/provider/stage differ"
        )
    for name in ("execution_mode", "cost_class"):
        _require_text(manifest.get(name), f"manifest {name}")
    for name, minimum, maximum in (
        ("max_calls", 1, 10000),
        ("max_results", 1, 10000),
    ):
        value = manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise RoutingModelBindingObservationError(
                f"routing model source-add {name} is invalid"
            )
    for name, minimum, maximum in (
        ("unit_cost", 0.0, 1_000_000.0),
        ("timeout_seconds", 0.001, 86_400.0),
    ):
        value = manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RoutingModelBindingObservationError(
                f"routing model source-add {name} is invalid"
            )
        if not math.isfinite(float(value)) or not minimum <= float(value) <= maximum:
            raise RoutingModelBindingObservationError(
                f"routing model source-add {name} is invalid"
            )
    for name in ("capabilities", "intent_categories", "evidence_types"):
        _require_string_list(manifest.get(name), f"manifest {name}")
    contracts = manifest.get("category_contracts")
    if not isinstance(contracts, list) or len(contracts) > 64:
        raise RoutingModelBindingObservationError(
            "routing model source-add category contracts are invalid"
        )
    categories: list[str] = []
    for contract in contracts:
        if not isinstance(contract, Mapping) or set(contract) != _CATEGORY_CONTRACT_FIELDS:
            raise RoutingModelBindingObservationError(
                "routing model source-add category contract is invalid"
            )
        category = _require_text(contract.get("category"), "category contract")
        categories.append(category)
        for name in ("capabilities", "evidence_types", "requirements"):
            _require_string_list(
                contract.get(name),
                f"category contract {name}",
                maximum=32,
            )
    if len(categories) != len(set(categories)) or categories != sorted(categories):
        raise RoutingModelBindingObservationError(
            "routing model source-add category contracts are not canonical"
        )
    _require_string_list(
        manifest.get("binding_requirements"),
        "binding requirements",
        maximum=32,
    )
    raw_manifest_hash = _raw_or_prefixed_hash(
        row.get("manifest_sha256"), "manifest hash"
    )
    if raw_manifest_hash != _source_add_manifest_digest(manifest):
        raise RoutingModelBindingObservationError(
            "routing model source-add manifest digest differs"
        )
    revision = row.get("revision")
    if not isinstance(revision, str) or revision != f"source-add-{raw_manifest_hash[:12]}":
        raise RoutingModelBindingObservationError(
            "routing model source-add revision differs"
        )
    if row.get("tool_id") != tool_id:
        raise RoutingModelBindingObservationError(
            "routing model source-add row tool differs"
        )
    return {
        "tool_id": tool_id,
        "revision": revision,
        "manifest_sha256": raw_manifest_hash,
        "manifest": dict(manifest),
    }


def observe_routing_model_bindings_v2(
    *,
    runtime_metadata: Mapping[str, Any],
    provider_bindings: Sequence[ProviderBindingIdentity],
    artifact_lineage_hash: str,
) -> dict[str, Any]:
    """Create a redacted model-binding observation before provider calls.

    This is deliberately a pure comparison.  It accepts only model-owned
    public metadata and typed binding identities; no runtime catalog or
    provider response can fill a missing or conflicting declaration.
    """

    if not isinstance(runtime_metadata, Mapping):
        raise RoutingModelBindingObservationError(
            "runtime routing metadata is invalid"
        )
    missing = _MODEL_METADATA_REQUIRED_FIELDS.difference(runtime_metadata)
    if missing:
        raise RoutingModelBindingObservationError(
            "runtime routing metadata fields are missing"
        )
    if runtime_metadata.get("private_bindings_exposed") is not False:
        raise RoutingModelBindingObservationError(
            "runtime routing metadata exposes private bindings"
        )
    _require_text(runtime_metadata.get("compiler_version"), "compiler version")
    _hash(runtime_metadata.get("catalog_sha256"), "catalog hash")
    _hash(runtime_metadata.get("policy_sha256"), "policy hash")

    rows = runtime_metadata.get("source_add_binding_manifests")
    if not isinstance(rows, list) or not rows:
        raise RoutingModelBindingObservationError(
            "runtime source-add binding manifests are invalid"
        )
    normalized_rows = [_validate_binding_manifest_row(row) for row in rows]
    if [row["tool_id"] for row in normalized_rows] != sorted(
        row["tool_id"] for row in normalized_rows
    ):
        raise RoutingModelBindingObservationError(
            "runtime source-add binding manifests are not canonical"
        )
    if len({row["tool_id"] for row in normalized_rows}) != len(normalized_rows):
        raise RoutingModelBindingObservationError(
            "runtime source-add binding manifests contain duplicates"
        )
    row_by_tool = {row["tool_id"]: row for row in normalized_rows}

    attestations = runtime_metadata.get("source_add_manifest_attestations")
    attestation_tool_ids = (
        [
            item.get("tool_id") if isinstance(item, Mapping) else None
            for item in attestations
        ]
        if isinstance(attestations, list)
        else None
    )
    if attestation_tool_ids is None or any(
        not isinstance(tool_id, str) for tool_id in attestation_tool_ids
    ) or attestation_tool_ids != sorted(row_by_tool):
        raise RoutingModelBindingObservationError(
            "runtime source-add manifest attestations are not canonical"
        )
    normalized_attestations = []
    for item in attestations:
        if not isinstance(item, Mapping) or set(item) != _ATTESTATION_FIELDS:
            raise RoutingModelBindingObservationError(
                "runtime source-add manifest attestation is invalid"
            )
        tool_id = item.get("tool_id")
        row = row_by_tool.get(tool_id)
        if row is None or item.get("revision") != row["revision"]:
            raise RoutingModelBindingObservationError(
                "runtime source-add manifest attestation differs"
            )
        if _raw_or_prefixed_hash(item.get("manifest_sha256"), "attestation hash") != row[
            "manifest_sha256"
        ]:
            raise RoutingModelBindingObservationError(
                "runtime source-add manifest attestation differs"
            )
        normalized_attestations.append(dict(item))

    if not isinstance(provider_bindings, Sequence) or isinstance(
        provider_bindings, (str, bytes)
    ) or not provider_bindings:
        raise RoutingModelBindingObservationError("provider bindings are invalid")
    identities: dict[str, ProviderBindingIdentity] = {}
    for binding in provider_bindings:
        if not isinstance(binding, ProviderBindingIdentity):
            raise RoutingModelBindingObservationError(
                "provider bindings must be typed identities"
            )
        errors = validate_provider_binding_identity(binding)
        if errors:
            raise RoutingModelBindingObservationError(
                "provider binding identity is invalid"
            )
        identity_hash = routing_model_binding_identity_hash(binding)
        if identity_hash in identities:
            raise RoutingModelBindingObservationError(
                "provider bindings contain duplicates"
            )
        identities[identity_hash] = binding
        row = row_by_tool.get(binding.tool_id)
        if row is None:
            raise RoutingModelBindingObservationError(
                "provider binding is not declared by model"
            )
        manifest = row["manifest"]
        if (
            binding.provider_id != manifest["provider_id"]
            or binding.tool_id != manifest["tool_id"]
            or _raw_or_prefixed_hash(binding.manifest_hash, "binding manifest hash")
            != row["manifest_sha256"]
        ):
            raise RoutingModelBindingObservationError(
                "provider binding does not match model manifest"
            )

    # The measured model metadata and the reviewed provider identity set are
    # one contract. A host cannot observe one binding while leaving another
    # model SourceAdd executable, or append an unmodelled binding.
    if set(row_by_tool) != {binding.tool_id for binding in identities.values()}:
        raise RoutingModelBindingObservationError(
            "provider binding set differs from model source-add set"
        )

    requirements = {
        identity_hash: routing_model_binding_requirements_hash(
            row_by_tool[binding.tool_id]["manifest"]
        )
        for identity_hash, binding in identities.items()
    }
    return build_routing_model_binding_observation_result_v2(
        artifact_lineage_hash=artifact_lineage_hash,
        requirement_hash_by_binding_identity=requirements,
    )


def routing_model_binding_identity_hash(binding: ProviderBindingIdentity) -> str:
    return sha256_json(
        {
            "schema_version": "leadpoet.routing_provider_binding.v1",
            "binding": binding.to_dict(),
        }
    )


def build_routing_model_binding_observation_result_v2(
    *,
    artifact_lineage_hash: str,
    requirement_hash_by_binding_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Build the canonical redacted output of the exact model sandbox probe."""

    lineage_hash = _hash(artifact_lineage_hash, "artifact lineage hash")
    rows = []
    for identity_hash, requirement_hash in sorted(
        requirement_hash_by_binding_identity.items()
    ):
        rows.append(
            {
                "binding_identity_hash": _hash(
                    identity_hash, "binding identity hash"
                ),
                "requirements_hash": _hash(
                    requirement_hash, "requirements hash"
                ),
            }
        )
    if not rows or len({row["binding_identity_hash"] for row in rows}) != len(rows):
        raise RoutingModelBindingObservationError(
            "routing model binding observation set is invalid"
        )
    request = {
        "schema_version": ROUTING_MODEL_BINDING_OBSERVATION_REQUEST_SCHEMA_V2,
        "artifact_lineage_hash": lineage_hash,
        "binding_identity_hashes": [row["binding_identity_hash"] for row in rows],
    }
    return {
        "schema_version": ROUTING_MODEL_BINDING_OBSERVATION_SCHEMA_V2,
        "artifact_lineage_hash": lineage_hash,
        "request_root": sha256_json(request),
        "requirements": rows,
    }


@dataclass(frozen=True, init=False)
class VerifiedRoutingModelBindingRequirements:
    """One immutable model observation that has a valid execution signature."""

    artifact_lineage_hash: str
    observation_receipt_hash: str
    requirement_hash_by_binding_identity: Mapping[str, str]
    result: Mapping[str, Any]
    signed_receipt: Mapping[str, Any]

    @classmethod
    def from_attested(
        cls,
        result: Mapping[str, Any],
        signed_receipt: Mapping[str, Any],
    ) -> "VerifiedRoutingModelBindingRequirements":
        if not isinstance(result, Mapping) or set(result) != {
            "schema_version",
            "artifact_lineage_hash",
            "request_root",
            "requirements",
        }:
            raise RoutingModelBindingObservationError(
                "routing model binding observation fields are invalid"
            )
        if result.get("schema_version") != ROUTING_MODEL_BINDING_OBSERVATION_SCHEMA_V2:
            raise RoutingModelBindingObservationError(
                "routing model binding observation schema is invalid"
            )
        artifact_lineage_hash = _hash(
            result.get("artifact_lineage_hash"), "artifact lineage hash"
        )
        rows = result.get("requirements")
        if not isinstance(rows, list) or not rows:
            raise RoutingModelBindingObservationError(
                "routing model binding observation set is invalid"
            )
        requirement_map: dict[str, str] = {}
        canonical_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "binding_identity_hash",
                "requirements_hash",
            }:
                raise RoutingModelBindingObservationError(
                    "routing model binding observation row is invalid"
                )
            identity_hash = _hash(
                row.get("binding_identity_hash"), "binding identity hash"
            )
            requirement_hash = _hash(
                row.get("requirements_hash"), "requirements hash"
            )
            if identity_hash in requirement_map:
                raise RoutingModelBindingObservationError(
                    "routing model binding observation has duplicate identities"
                )
            requirement_map[identity_hash] = requirement_hash
            canonical_rows.append(
                {
                    "binding_identity_hash": identity_hash,
                    "requirements_hash": requirement_hash,
                }
            )
        if canonical_rows != sorted(
            canonical_rows, key=lambda row: row["binding_identity_hash"]
        ):
            raise RoutingModelBindingObservationError(
                "routing model binding observation is not canonical"
            )
        expected_result = build_routing_model_binding_observation_result_v2(
            artifact_lineage_hash=artifact_lineage_hash,
            requirement_hash_by_binding_identity=requirement_map,
        )
        if dict(result) != expected_result:
            raise RoutingModelBindingObservationError(
                "routing model binding observation result differs"
            )
        try:
            validate_signed_execution_receipt(signed_receipt)
        except Exception as exc:
            raise RoutingModelBindingObservationError(
                "routing model binding observation signature is invalid"
            ) from exc
        if (
            signed_receipt.get("role") != "gateway_scoring"
            or signed_receipt.get("purpose")
            != ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2
            or signed_receipt.get("status") != "succeeded"
            or signed_receipt.get("input_root") != expected_result["request_root"]
            or signed_receipt.get("output_root") != sha256_json(expected_result)
        ):
            raise RoutingModelBindingObservationError(
                "routing model binding observation receipt differs"
            )
        receipt_hash = _hash(
            signed_receipt.get("receipt_hash"), "observation receipt hash"
        )
        value = object.__new__(cls)
        object.__setattr__(value, "artifact_lineage_hash", artifact_lineage_hash)
        object.__setattr__(value, "observation_receipt_hash", receipt_hash)
        object.__setattr__(
            value,
            "requirement_hash_by_binding_identity",
            MappingProxyType(dict(requirement_map)),
        )
        object.__setattr__(value, "result", MappingProxyType(dict(expected_result)))
        object.__setattr__(
            value, "signed_receipt", MappingProxyType(dict(signed_receipt))
        )
        return value

    @classmethod
    def from_attested_mapping(
        cls, value: Mapping[str, Any]
    ) -> "VerifiedRoutingModelBindingRequirements":
        if not isinstance(value, Mapping) or set(value) != {"result", "receipt"}:
            raise RoutingModelBindingObservationError(
                "routing model binding attested document is invalid"
            )
        result = value.get("result")
        receipt = value.get("receipt")
        if not isinstance(result, Mapping) or not isinstance(receipt, Mapping):
            raise RoutingModelBindingObservationError(
                "routing model binding attested document is invalid"
            )
        return cls.from_attested(result, receipt)

    def to_attested_dict(self) -> dict[str, Any]:
        return {"result": dict(self.result), "receipt": dict(self.signed_receipt)}

    def resolve(
        self,
        *,
        binding: ProviderBindingIdentity,
        artifact_lineage_hash: str,
    ) -> str:
        if artifact_lineage_hash != self.artifact_lineage_hash:
            raise RoutingModelBindingObservationError(
                "routing model binding observation belongs to another artifact"
            )
        result = self.requirement_hash_by_binding_identity.get(
            routing_model_binding_identity_hash(binding)
        )
        if result is None:
            raise RoutingModelBindingObservationError(
                "routing model binding requirements were not observed"
            )
        return result


__all__ = [
    "ROUTING_MODEL_BINDING_REQUIREMENTS_SCHEMA_V2",
    "ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2",
    "ROUTING_MODEL_BINDING_OBSERVATION_REQUEST_SCHEMA_V2",
    "ROUTING_MODEL_BINDING_OBSERVATION_SCHEMA_V2",
    "RoutingModelBindingObservationError",
    "VerifiedRoutingModelBindingRequirements",
    "build_routing_model_binding_observation_result_v2",
    "observe_routing_model_bindings_v2",
    "routing_model_binding_requirements_hash",
    "routing_model_binding_identity_hash",
]
