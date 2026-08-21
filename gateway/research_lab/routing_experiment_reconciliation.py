"""Protected, deterministic reconciliation for routing Lab promotion.

This module rebuilds metrics, gates, selection, row roots, and exact budget
chains from canonical redacted authority documents. Host-computed claims are
comparison values only.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderOutcome,
    ProviderReceipt,
    RoutingDecisionReceiptV2,
    RoutingExperimentError,
    RoutingExperimentV2Spec,
    _v2_metrics,
    _v2_variant_artifact_key,
    validate_provider_receipt,
    validate_routing_decision_receipt,
)
from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
    validate_routing_execution_envelope_v2,
)
from gateway.research_lab.routing_admission import (
    RoutingAdmissionBundleV2,
    RoutingAdmissionError,
)
from gateway.research_lab.routing_provider_terminal import (
    RoutingProviderTerminalError,
    validate_routing_provider_terminal_v2,
)
from leadpoet_canonical.attested_v2 import validate_signed_execution_receipt


SCHEMA = "leadpoet.research_lab.routing_experiment_attestation.v2"
RESULT_SCHEMA = "leadpoet.research_lab.routing_experiment_attestation_result.v2"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
PROVIDER_REF_RE = re.compile(r"^provider_receipt:[0-9a-f]{16}$")
DECISION_REF_RE = re.compile(r"^routing_decision:[0-9a-f]{16}$")


class RoutingReconciliationError(ValueError):
    """Canonical persisted evidence cannot authorize promotion."""


def _fail(message: str) -> None:
    raise RoutingReconciliationError(message)


def _hash(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not HASH_RE.fullmatch(text):
        _fail(f"routing reconciliation {name} is invalid")
    return text


def _ref(value: Any, name: str) -> str:
    text = str(value or "")
    if not REF_RE.fullmatch(text):
        _fail(f"routing reconciliation {name} is invalid")
    return text


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"routing reconciliation {name} fields are invalid")
    return dict(value)


def _rows(value: Any, *, key: str, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        _fail(f"routing reconciliation {name} is invalid")
    rows = tuple(dict(item) if isinstance(item, Mapping) else {} for item in value)
    keys = [str(row.get(key) or "") for row in rows]
    if any(not item for item in keys) or keys != sorted(set(keys)):
        _fail(f"routing reconciliation {name} is not canonical")
    return rows


def _root(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    return sha256_json([{"key": str(row[key]), "row": dict(row)} for row in rows])


def build_input(
    *,
    spec_doc: Mapping[str, Any],
    evaluation_doc: Mapping[str, Any],
    gold_label_authority: Mapping[str, Any],
    artifact_lineage: Mapping[str, Any],
    execution_envelope: Mapping[str, Any],
    decision_receipts: Sequence[Mapping[str, Any]],
    provider_attempts: Sequence[Mapping[str, Any]],
    budget_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA,
        "spec_doc": dict(spec_doc),
        "evaluation_doc": dict(evaluation_doc),
        "gold_label_authority": dict(gold_label_authority),
        "artifact_lineage": dict(artifact_lineage),
        "execution_envelope": dict(execution_envelope),
        "decision_receipts": [dict(item) for item in decision_receipts],
        "provider_attempts": [dict(item) for item in provider_attempts],
        "budget_events": [dict(item) for item in budget_events],
    }
    reconcile(payload)
    return payload


def _spec(payload: Mapping[str, Any]) -> RoutingExperimentV2Spec:
    try:
        return RoutingExperimentV2Spec.from_mapping(payload["spec_doc"])
    except (KeyError, TypeError, ValueError, RoutingExperimentError) as exc:
        raise RoutingReconciliationError("routing reconciliation spec is invalid") from exc


def _labels(value: Any, spec: RoutingExperimentV2Spec) -> tuple[dict[str, bool], str]:
    doc = _exact(
        value,
        {
            "manifest_uri", "manifest_hash", "signature_ref", "signing_key_id",
            "label_set_hash", "labels", "provenance_hash",
        },
        "gold label authority",
    )
    uri = str(doc["manifest_uri"] or "")
    if not uri.startswith("s3://") or uri.endswith("/current.json") or "/branches/" in uri:
        _fail("routing reconciliation gold label URI is mutable")
    for name in ("manifest_hash", "label_set_hash", "provenance_hash"):
        _hash(doc[name], f"gold label {name}")
    if not str(doc["signature_ref"] or "").startswith("s3://") or not str(doc["signing_key_id"] or "").strip():
        _fail("routing reconciliation gold label signature is invalid")
    labels = doc["labels"]
    if not isinstance(labels, Mapping) or any(
        not isinstance(key, str) or type(item) is not bool for key, item in labels.items()
    ):
        _fail("routing reconciliation gold labels are invalid")
    normalized = {str(key): item for key, item in sorted(labels.items())}
    expected_units = tuple(sorted((*spec.input.calibration_unit_refs, *spec.input.holdout_unit_refs)))
    if tuple(normalized) != expected_units:
        _fail("routing reconciliation gold label units differ")
    label_hash = sha256_json({"labels": list(normalized.items())})
    if label_hash != spec.input.gold_label_set_hash or doc["label_set_hash"] != label_hash:
        _fail("routing reconciliation gold label hash differs")
    return normalized, str(doc["manifest_hash"])


LINEAGE_FIELDS = {
    "repository", "branch", "commit_sha", "pointer_uri", "pointer_document_hash",
    "immutable_manifest_uri", "routing_lineage_manifest_uri",
    "routing_lineage_manifest_hash", "manifest_hash", "signature_ref",
    "signature_key_id", "signature_algorithm", "model_artifact_hash", "image_digest",
    "config_hash", "build_id", "component_registry_version", "scoring_adapter_version",
    "routing_contract_hash", "routing_catalog_hash", "routing_policy_hash",
    "feature_schema_hash", "verifier_contract_hash",
}


def _lineage(value: Any, spec: RoutingExperimentV2Spec) -> str:
    doc = _exact(value, LINEAGE_FIELDS, "artifact lineage")
    branch = str(doc["branch"] or "")
    pointer_uri = str(doc["pointer_uri"] or "")
    if (
        doc["repository"] != "leadpoet/Sourcing_model"
        or branch not in {"main", "leadpoet-lab"}
        or not re.fullmatch(r"[0-9a-f]{40}", str(doc["commit_sha"] or ""))
        or not pointer_uri.endswith(f"/branches/{branch}/current.json")
        or "/branches/" in str(doc["immutable_manifest_uri"] or "")
        or "/branches/" in str(doc["routing_lineage_manifest_uri"] or "")
        or doc["signature_algorithm"] != "ECDSA_SHA_256"
        or not str(doc["signature_ref"] or "").startswith("s3://")
        or not str(doc["signature_key_id"] or "").strip()
    ):
        _fail("routing reconciliation artifact lineage is invalid")
    for name in (
        "pointer_document_hash", "routing_lineage_manifest_hash", "manifest_hash",
        "model_artifact_hash", "config_hash", "routing_contract_hash",
        "routing_catalog_hash", "routing_policy_hash", "feature_schema_hash",
        "verifier_contract_hash",
    ):
        _hash(doc[name], f"artifact {name}")
    image_digest = str(doc["image_digest"] or "")
    image_hash = image_digest.rsplit("@", 1)[-1]
    if "@" not in image_digest or not HASH_RE.fullmatch(image_hash):
        _fail("routing reconciliation artifact image digest is invalid")
    expected = {
        "repository": doc["repository"], "branch": doc["branch"],
        "commit_sha": doc["commit_sha"], "artifact_uri": doc["pointer_uri"],
        "model_artifact_hash": doc["model_artifact_hash"],
        "manifest_hash": doc["manifest_hash"],
        "routing_contract_hash": doc["routing_contract_hash"],
        "routing_catalog_hash": doc["routing_catalog_hash"],
        "routing_policy_hash": doc["routing_policy_hash"],
        "feature_schema_hash": doc["feature_schema_hash"],
        "verifier_contract_hash": doc["verifier_contract_hash"],
    }
    baseline = next(
        (
            variant
            for variant in spec.variants
            if variant.variant_id == spec.baseline_variant_id
        ),
        None,
    )
    if baseline is None or baseline.artifact.to_dict() != expected:
        _fail("routing reconciliation baseline artifact lineage differs")
    return sha256_json({"schema_version": "leadpoet.routing_artifact_lineage.v2", **doc})


def _execution_envelope(
    value: Any,
    spec: RoutingExperimentV2Spec,
    *,
    lineage_hash: str,
    label_manifest_hash: str,
) -> RoutingExperimentExecutionEnvelopeV2:
    try:
        envelope = RoutingExperimentExecutionEnvelopeV2.from_mapping(value)
        validate_routing_execution_envelope_v2(spec=spec, envelope=envelope)
    except (TypeError, ValueError) as exc:
        raise RoutingReconciliationError(
            "routing reconciliation execution envelope is invalid"
        ) from exc
    if (
        envelope.artifact_lineage_hash != lineage_hash
        or envelope.gold_label_manifest_hash != label_manifest_hash
    ):
        _fail("routing reconciliation execution envelope authority differs")
    return envelope


ATTEMPT_FIELDS = {
    "attempt_key", "experiment_hash", "provider_receipt_ref", "binding_id", "tool_id",
    "variant_id", "unit_ref", "reservation_id", "action_id", "request_fingerprint",
    "outcome", "credit_microunits", "latency_ms", "execution_mode", "billing_state",
    "binding_catalog_manifest_hash", "authorization_hash",
    "authorization_proof_hash", "request_body_hash",
    "authoritative_billed_credit_microunits", "attempt_doc",
}
PROVIDER_FIELDS = {
    "receipt_ref", "binding_id", "tool_id", "binding_version", "source_lineage_id",
    "unit_ref", "request_fingerprint", "outcome", "evidence_hash",
    "credit_microunits", "latency_ms", "execution_mode",
}
ATTEMPT_DOC_FIELDS = {
    "schema_version", "binding_id", "tool_id", "action_id",
    "binding_catalog_manifest_hash", "call_grant_hash",
    "call_grant_proof_hash", "authorization_request_hash", "request_body_hash", "variant_id",
    "unit_ref", "reservation_id", "request_fingerprint", "execution_mode",
    "provider_receipt", "call_grant", "call_grant_result",
    "call_grant_receipt", "terminal_result", "terminal_execution_receipt",
    "protected_release_receipt", "admission_bundle",
}


def _provider_receipt(row: Mapping[str, Any]) -> ProviderReceipt:
    doc = row["attempt_doc"]
    if not isinstance(doc, Mapping):
        _fail("routing reconciliation provider attempt document is invalid")
    provider_doc = doc.get("provider_receipt") if "provider_receipt" in doc else doc
    if not isinstance(provider_doc, Mapping):
        _fail("routing reconciliation provider receipt is invalid")
    projection = {key: provider_doc[key] for key in PROVIDER_FIELDS if key in provider_doc}
    try:
        receipt = ProviderReceipt.from_mapping(projection)
    except (TypeError, ValueError, RoutingExperimentError) as exc:
        raise RoutingReconciliationError("routing reconciliation provider receipt is invalid") from exc
    errors = validate_provider_receipt(receipt)
    if errors:
        _fail("routing reconciliation provider receipt is invalid:" + ";".join(errors))
    pairs = {
        "receipt_ref": "provider_receipt_ref", "binding_id": "binding_id",
        "tool_id": "tool_id", "unit_ref": "unit_ref",
        "request_fingerprint": "request_fingerprint", "outcome": "outcome",
        "credit_microunits": "credit_microunits", "latency_ms": "latency_ms",
        "execution_mode": "execution_mode",
    }
    if any(getattr(receipt, left) != row[right] for left, right in pairs.items()):
        _fail("routing reconciliation provider attempt scalar differs")
    return receipt


def _admission_bundle(
    value: Any,
    *,
    authorization: RoutingProviderCallAuthorizationV2,
    spec: RoutingExperimentV2Spec,
    envelope: RoutingExperimentExecutionEnvelopeV2,
    lineage_hash: str,
) -> RoutingAdmissionBundleV2:
    try:
        bundle = RoutingAdmissionBundleV2.from_mapping(value)
    except (TypeError, ValueError, RoutingAdmissionError) as exc:
        raise RoutingReconciliationError(
            "routing reconciliation admission bundle is invalid"
        ) from exc
    if (
        bundle.job_id != authorization.admission_job_id
        or bundle.experiment_id != spec.experiment_id
        or bundle.experiment_hash != spec.experiment_hash()
        or bundle.purpose != authorization.purpose
        or bundle.envelope_hash != envelope.envelope_hash()
        or bundle.artifact_lineage_hash != lineage_hash
        or bundle.pointer_document_hash != envelope.pointer_document_hash
        or bundle.model_artifact_hash != authorization.model_artifact_hash
        or bundle.immutable_manifest_hash != authorization.manifest_hash
        or bundle.unit_dataset_manifest_hash != envelope.unit_dataset_manifest_hash
        or bundle.unit_set_hash != envelope.unit_set_hash
        or bundle.binding_catalog_manifest_hash
            != envelope.binding_catalog_manifest_hash
        or bundle.binding_catalog_version != envelope.binding_catalog_version
        or bundle.model_binding_observation_receipt_hash
            != envelope.model_binding_observation_receipt_hash
        or bundle.binding_ids != tuple(sorted(item.binding_id for item in envelope.bindings))
        or bundle.identity_hash() != authorization.admission_bundle_hash
        or bundle.protected_release_hash != authorization.protected_release_hash
        or bundle.protected_boot_identity_hash
            != authorization.protected_boot_identity_hash
    ):
        raise RoutingReconciliationError("routing reconciliation admission authority differs")
    return bundle


def _legacy_fixture_attempt(
    *,
    row: Mapping[str, Any],
    attempt_doc: Mapping[str, Any],
    spec: RoutingExperimentV2Spec,
    envelope: RoutingExperimentExecutionEnvelopeV2,
    variant: Any,
    binding: Any,
    runtime_binding: Any,
) -> tuple[ProviderReceipt, RoutingProviderCallAuthorizationV2]:
    """Read the pre-v3 fixture projection during one-way migration only.

    Measured attempts must use the v3 standard receipt path below.  This
    compatibility branch keeps deterministic historical replay fixtures
    readable without making the host-signed terminal helper valid for live
    execution.
    """
    try:
        authorization = RoutingProviderCallAuthorizationV2.from_mapping(
            attempt_doc["call_grant"]
        )
        authorization_receipt = attempt_doc["call_grant_receipt"]
        validate_signed_execution_receipt(authorization_receipt)
        authorization_job_id = str(authorization_receipt.get("job_id") or "")
        expected_authorization = execute_routing_provider_call_authorization_v2(
            authorization.to_dict(), authorization_job_id=authorization_job_id
        )
        admission = _admission_bundle(
            attempt_doc["admission_bundle"], authorization=authorization,
            spec=spec, envelope=envelope,
            lineage_hash=envelope.artifact_lineage_hash,
        )
        protected_receipt = attempt_doc["protected_release_receipt"]
        validate_signed_execution_receipt(protected_receipt)
        if (
            runtime_binding is None
            or authorization.experiment_hash != spec.experiment_hash()
            or authorization.variant_id != variant.variant_id
            or authorization.stage != variant.stage
            or authorization.binding != binding
            or authorization.action_id != runtime_binding.action_id
            or authorization.unit_ref != row["unit_ref"]
            or row["authorization_hash"] != authorization.authorization_hash()
            or row["authorization_proof_hash"] != authorization_receipt.get("receipt_hash")
            or attempt_doc["call_grant_result"] != expected_authorization
            or authorization_receipt.get("input_root") != authorization.authorization_hash()
            or authorization_receipt.get("output_root") != expected_authorization["output_root"]
            or protected_receipt.get("job_id") != admission.job_id
            or protected_receipt.get("receipt_hash") != admission.protected_receipt_hash
        ):
            _fail("routing reconciliation legacy fixture authorization differs")
        receipt = _provider_receipt(row)
        projected = validate_routing_provider_terminal_v2(
            terminal=attempt_doc["terminal_proof"], binding=binding,
            protected_receipt=protected_receipt, expected_job_id=admission.job_id,
            expected_experiment_hash=spec.experiment_hash(),
            expected_admission_bundle_hash=admission.identity_hash(),
            expected_authorization_hash=authorization.authorization_hash(),
            expected_authorization_proof_hash=authorization_receipt["receipt_hash"],
        )
        if projected != receipt.to_dict():
            _fail("routing reconciliation legacy fixture terminal differs")
        return receipt, authorization
    except (RoutingProviderTerminalError, TypeError, ValueError, KeyError) as exc:
        raise RoutingReconciliationError(
            "routing reconciliation legacy fixture attempt is invalid"
        ) from exc


def _attempts(
    value: Any,
    spec: RoutingExperimentV2Spec,
    envelope: RoutingExperimentExecutionEnvelopeV2,
):
    rows = _rows(value, key="attempt_key", name="provider attempts")
    by_ref: dict[
        str,
        tuple[
            dict[str, Any], ProviderReceipt, RoutingProviderCallAuthorizationV2
        ],
    ] = {}
    variants = {item.variant_id: item for item in spec.variants}
    bindings = {item.binding_id: item for item in spec.provider_bindings}
    units = set((*spec.input.calibration_unit_refs, *spec.input.holdout_unit_refs))
    runtime_bindings = {item.binding_id: item for item in envelope.bindings}
    for raw in rows:
        row = _exact(raw, ATTEMPT_FIELDS, "provider attempt")
        _hash(row["attempt_key"], "provider attempt key")
        receipt_ref = str(row["provider_receipt_ref"] or "")
        variant = variants.get(str(row["variant_id"] or ""))
        binding = bindings.get(str(row["binding_id"] or ""))
        if (
            row["experiment_hash"] != spec.experiment_hash()
            or not PROVIDER_REF_RE.fullmatch(receipt_ref)
            or receipt_ref in by_ref
            or variant is None or binding is None
            or binding.binding_id not in variant.binding_ids
            or binding.tool_id != row["tool_id"]
            or row["unit_ref"] not in units
            or not REF_RE.fullmatch(str(row["reservation_id"] or ""))
            or not REF_RE.fullmatch(str(row["action_id"] or ""))
        ):
            _fail("routing reconciliation provider binding differs")
        attempt_doc = row["attempt_doc"]
        if not isinstance(attempt_doc, Mapping):
            _fail("routing reconciliation provider attempt document is invalid")
        # Validate the authoritative billing projection before the one-way
        # legacy-fixture branch.  Otherwise a forged legacy row with an
        # unresolved amount is reported as a budget-chain mismatch, which
        # obscures the primary billing failure and changes the fail-closed
        # error contract.
        if (
            row["billing_state"] != "known"
            or type(row["authoritative_billed_credit_microunits"]) is not int
            or row["authoritative_billed_credit_microunits"] < 0
            or row["authoritative_billed_credit_microunits"] != row["credit_microunits"]
            or row["outcome"] == ProviderOutcome.ADAPTER_FAILURE.value
        ):
            _fail("routing reconciliation provider billing is unresolved")
        if (
            attempt_doc.get("schema_version") == "leadpoet.research_lab.routing_provider_attempt.v2"
            and attempt_doc.get("legacy_fixture") is True
        ):
            receipt, authorization = _legacy_fixture_attempt(
                row=row,
                attempt_doc=attempt_doc,
                spec=spec,
                envelope=envelope,
                variant=variant,
                binding=binding,
                runtime_binding=runtime_bindings.get(str(row["binding_id"])),
            )
            by_ref[receipt_ref] = (row, receipt, authorization)
            continue
        if (
            set(attempt_doc) != ATTEMPT_DOC_FIELDS
            or attempt_doc.get("schema_version")
                != "leadpoet.research_lab.routing_provider_attempt.v3"
        ):
            _fail("routing reconciliation provider attempt document is invalid")
        try:
            authorization = RoutingProviderCallAuthorizationV2.from_mapping(
                attempt_doc["call_grant"]
            )
        except (TypeError, ValueError) as exc:
            raise RoutingReconciliationError(
                "routing reconciliation provider authorization is invalid"
            ) from exc
        runtime_binding = runtime_bindings.get(str(row["binding_id"]))
        authorization_receipt = attempt_doc["call_grant_receipt"]
        try:
            if not isinstance(authorization_receipt, Mapping):
                raise ValueError("receipt is not an object")
            validate_signed_execution_receipt(authorization_receipt)
            authorization_job_id = str(authorization_receipt.get("job_id") or "")
            expected_authorization = execute_routing_provider_call_authorization_v2(
                authorization.to_dict(), authorization_job_id=authorization_job_id
            )
        except Exception as exc:
            raise RoutingReconciliationError(
                "routing reconciliation provider authorization signature is invalid"
            ) from exc
        if (
            runtime_binding is None
            or authorization.experiment_hash != spec.experiment_hash()
            or authorization.variant_id != variant.variant_id
            or authorization.stage != variant.stage
            or authorization.binding != binding
            or authorization.binding_catalog_manifest_hash != envelope.binding_catalog_manifest_hash
            or authorization.binding_catalog_version != envelope.binding_catalog_version
            or authorization.action_id != runtime_binding.action_id
            or authorization.unit_ref != row["unit_ref"]
            or authorization.unit_set_hash != envelope.unit_set_hash
            or authorization.unit_dataset_manifest_hash != envelope.unit_dataset_manifest_hash
            or authorization.core_request_fingerprint != row["request_fingerprint"]
            or authorization.request_body_hash != row["request_body_hash"]
            or authorization.artifact_lineage_hash != envelope.artifact_lineage_hash
            or authorization.pointer_document_hash != envelope.pointer_document_hash
            or authorization.credit_cap_microunits > runtime_binding.credit_ceiling_microunits
            or authorization.timeout_ms > runtime_binding.timeout_ms
            or row["binding_catalog_manifest_hash"] != envelope.binding_catalog_manifest_hash
            or row["authorization_hash"] != authorization.authorization_hash()
            or row["authorization_proof_hash"] != authorization_receipt.get("receipt_hash")
            or attempt_doc["authorization_request_hash"] != authorization_receipt.get("input_root")
            or attempt_doc["call_grant_result"] != expected_authorization
            or authorization_receipt.get("role") != "gateway_scoring"
            or authorization_receipt.get("purpose") != authorization.purpose
            or authorization_receipt.get("status") != "succeeded"
            or authorization_receipt.get("output_root") != expected_authorization["output_root"]
            or authorization.admission_job_id != attempt_doc["admission_bundle"].get("job_id")
        ):
            _fail("routing reconciliation provider authorization differs")
        scalar_doc_pairs = {
            "binding_id": row["binding_id"], "tool_id": row["tool_id"],
            "action_id": row["action_id"],
            "binding_catalog_manifest_hash": row["binding_catalog_manifest_hash"],
            "call_grant_hash": row["authorization_hash"],
            "call_grant_proof_hash": row["authorization_proof_hash"],
            "request_body_hash": row["request_body_hash"],
            "variant_id": row["variant_id"], "unit_ref": row["unit_ref"],
            "reservation_id": row["reservation_id"],
            "request_fingerprint": row["request_fingerprint"],
            "execution_mode": row["execution_mode"],
        }
        if any(attempt_doc.get(key) != item for key, item in scalar_doc_pairs.items()):
            _fail("routing reconciliation provider attempt document differs")
        admission = _admission_bundle(
            attempt_doc["admission_bundle"], authorization=authorization, spec=spec,
            envelope=envelope, lineage_hash=envelope.artifact_lineage_hash,
        )
        protected_receipt = attempt_doc["protected_release_receipt"]
        terminal_result = attempt_doc["terminal_result"]
        terminal_receipt = attempt_doc["terminal_execution_receipt"]
        try:
            if not isinstance(protected_receipt, Mapping):
                raise ValueError("protected receipt is not an object")
            validate_signed_execution_receipt(protected_receipt)
            if not isinstance(terminal_result, Mapping) or not isinstance(terminal_receipt, Mapping):
                raise ValueError("terminal standard receipt is incomplete")
            validate_signed_execution_receipt(terminal_receipt)
        except Exception as exc:
            raise RoutingReconciliationError(
                "routing reconciliation standard receipt signature is invalid"
            ) from exc
        if (
            protected_receipt.get("role") != admission.role
            or protected_receipt.get("purpose") != admission.purpose
            or protected_receipt.get("status") != "succeeded"
            or protected_receipt.get("job_id") != admission.job_id
            or protected_receipt.get("receipt_hash") != admission.protected_receipt_hash
            or protected_receipt.get("commit_sha")
            != admission.protected_commit_sha
            or protected_receipt.get("pcr0") != admission.protected_pcr0
            or protected_receipt.get("build_manifest_hash")
            != admission.protected_build_manifest_hash
            or protected_receipt.get("dependency_lock_hash")
            != admission.protected_dependency_lock_hash
            or protected_receipt.get("config_hash")
            != admission.protected_config_hash
            or protected_receipt.get("boot_identity_hash")
            != admission.protected_boot_identity_hash
            or protected_receipt.get("enclave_pubkey")
            != admission.protected_enclave_pubkey
            or any(
                authorization_receipt.get(name) != protected_receipt.get(name)
                or terminal_receipt.get(name) != protected_receipt.get(name)
                for name in ("boot_identity_hash", "enclave_pubkey")
            )
            or terminal_receipt.get("job_id")
                != routing_provider_dispatch_job_id_v2(
                    {
                        "authorization_hash": authorization.authorization_hash(),
                        "authorization_proof_hash": row["authorization_proof_hash"],
                        "authorization_receipt": authorization_receipt,
                    }
                )
            or terminal_receipt.get("purpose") != authorization.purpose
            or terminal_receipt.get("status") != "succeeded"
            or terminal_receipt.get("output_root") != sha256_json(dict(terminal_result))
            or terminal_receipt.get("parent_receipt_hashes")
                != [authorization_receipt.get("receipt_hash")]
            or terminal_result.get("provider_receipt") != row["attempt_doc"].get("provider_receipt")
        ):
            _fail("routing reconciliation standard terminal receipt differs")
        _hash(row["request_fingerprint"], "provider request fingerprint")
        if (
            row["billing_state"] != "known"
            or type(row["authoritative_billed_credit_microunits"]) is not int
            or row["authoritative_billed_credit_microunits"] < 0
            or row["authoritative_billed_credit_microunits"] != row["credit_microunits"]
            or row["outcome"] == ProviderOutcome.ADAPTER_FAILURE.value
        ):
            _fail("routing reconciliation provider billing is unresolved")
        receipt = _provider_receipt(row)
        projection = terminal_result.get("projection")
        if (
            not isinstance(projection, Mapping)
            or projection.get("billing_state") != "known"
            or projection.get("outcome") != receipt.outcome
            or projection.get("evidence_hash") != receipt.evidence_hash
            or projection.get("credit_microunits") != receipt.credit_microunits
            or projection.get("latency_ms") != receipt.latency_ms
            or projection.get("binding_id") != receipt.binding_id
            or projection.get("tool_id") != receipt.tool_id
            or projection.get("request_fingerprint") != receipt.request_fingerprint
            or receipt.credit_microunits > authorization.credit_cap_microunits
            or receipt.latency_ms > authorization.timeout_ms
        ):
            _fail("routing reconciliation provider terminal projection differs")
        by_ref[receipt_ref] = (row, receipt, authorization)
    return rows, by_ref


BUDGET_FIELDS = {
    "event_key", "experiment_hash", "reservation_id", "binding_id", "attempt_key",
    "event_type", "credit_microunits", "event_doc",
}


def _budgets(value: Any, spec: RoutingExperimentV2Spec, attempts):
    rows = _rows(value, key="event_key", name="budget events")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = _exact(raw, BUDGET_FIELDS, "budget event")
        _hash(row["event_key"], "budget event key")
        reservation_id = _ref(row["reservation_id"], "budget reservation")
        if (
            row["experiment_hash"] != spec.experiment_hash()
            or row["event_type"] not in {"reserve", "settle"}
            or type(row["credit_microunits"]) is not int
            or row["credit_microunits"] < 0
            or not isinstance(row["event_doc"], Mapping)
        ):
            _fail("routing reconciliation budget event is invalid")
        grouped[reservation_id].append(row)
    attempts_by_reservation = {}
    for row, receipt, authorization in attempts.values():
        reservation_id = str(row["reservation_id"])
        if reservation_id in attempts_by_reservation:
            _fail("routing reconciliation reservation has multiple attempts")
        attempts_by_reservation[reservation_id] = (row, receipt, authorization)
    if set(grouped) != set(attempts_by_reservation):
        _fail("routing reconciliation attempt budget sets differ")
    billed_total = 0
    billed_by_binding: dict[str, int] = defaultdict(int)
    for reservation_id, (attempt, _receipt, authorization) in attempts_by_reservation.items():
        events = grouped[reservation_id]
        reserve = [item for item in events if item["event_type"] == "reserve"]
        settle = [item for item in events if item["event_type"] == "settle"]
        if len(events) != 2 or len(reserve) != 1 or len(settle) != 1:
            _fail("routing reconciliation budget chain is not exact")
        reserve_row, settle_row = reserve[0], settle[0]
        if (
            reserve_row["attempt_key"] not in (None, "")
            or settle_row["attempt_key"] != attempt["attempt_key"]
            or reserve_row["binding_id"] != attempt["binding_id"]
            or settle_row["binding_id"] != attempt["binding_id"]
            or reserve_row["credit_microunits"]
                != authorization.credit_cap_microunits
            or settle_row["credit_microunits"] != attempt["authoritative_billed_credit_microunits"]
        ):
            _fail("routing reconciliation budget chain differs")
        expected = {
            "reservation_id": reservation_id, "binding_id": attempt["binding_id"],
            "unit_ref": attempt["unit_ref"], "variant_id": attempt["variant_id"],
            "request_fingerprint": attempt["request_fingerprint"],
            "action_id": attempt["action_id"],
        }
        if any(reserve_row["event_doc"].get(key) != item for key, item in expected.items()):
            _fail("routing reconciliation reserve document differs")
        if any(settle_row["event_doc"].get(key) != item for key, item in expected.items()):
            _fail("routing reconciliation settlement document differs")
        if settle_row["event_doc"].get("attempt_key") != attempt["attempt_key"]:
            _fail("routing reconciliation settlement attempt differs")
        billed_total += int(attempt["authoritative_billed_credit_microunits"])
        billed_by_binding[str(attempt["binding_id"])] += int(
            attempt["authoritative_billed_credit_microunits"]
        )
    if billed_total > spec.credit_budget.total_credit_microunits:
        _fail("routing reconciliation total budget is exceeded")
    for binding_id, billed in billed_by_binding.items():
        ceiling = spec.credit_budget.provider_credit_ceilings.get(binding_id)
        if ceiling is None or billed > ceiling:
            _fail("routing reconciliation provider budget is exceeded")
    return rows, billed_total


def _decisions(value: Any, spec: RoutingExperimentV2Spec, attempts):
    rows = _rows(value, key="receipt_id", name="decision receipts")
    variants = {item.variant_id: item for item in spec.variants}
    units = set((*spec.input.calibration_unit_refs, *spec.input.holdout_unit_refs))
    grouped: dict[tuple[str, str], list[RoutingDecisionReceiptV2]] = defaultdict(list)
    all_refs: list[str] = []
    for raw in rows:
        row = _exact(raw, {"receipt_id", "experiment_hash", "decision_doc"}, "decision receipt")
        if row["experiment_hash"] != spec.experiment_hash() or not DECISION_REF_RE.fullmatch(str(row["receipt_id"] or "")):
            _fail("routing reconciliation decision identity differs")
        try:
            decision = RoutingDecisionReceiptV2.from_mapping(row["decision_doc"])
        except (TypeError, ValueError, RoutingExperimentError) as exc:
            raise RoutingReconciliationError("routing reconciliation decision is invalid") from exc
        errors = validate_routing_decision_receipt(decision)
        if errors:
            _fail("routing reconciliation decision is invalid:" + ";".join(errors))
        variant = variants.get(decision.variant_id)
        if (
            decision.receipt_id != row["receipt_id"]
            or decision.experiment_id != spec.experiment_id
            or variant is None or decision.unit_ref not in units
            or decision.stage != variant.stage
            or decision.artifact_key != _v2_variant_artifact_key(variant)
            or decision.execution_mode != spec.receipt_execution_mode
        ):
            _fail("routing reconciliation decision binding differs")
        for tool_id, outcome_pair, receipt_ref in zip(
            decision.attempted_tool_ids, decision.outcome_reasons, decision.provider_receipt_refs
        ):
            pair = attempts.get(receipt_ref)
            if pair is None:
                _fail("routing reconciliation decision attempt is missing")
            attempt, receipt, _authorization = pair
            if (
                receipt.tool_id != tool_id or receipt.outcome != outcome_pair[1]
                or receipt.unit_ref != decision.unit_ref
                or attempt["binding_id"] not in variant.binding_ids
            ):
                _fail("routing reconciliation decision attempt differs")
            all_refs.append(receipt_ref)
        grouped[(decision.variant_id, decision.unit_ref)].append(decision)
    expected_pairs = {(variant.variant_id, unit) for variant in spec.variants for unit in units}
    if set(grouped) != expected_pairs or set(all_refs) != set(attempts):
        _fail("routing reconciliation decision coverage differs")
    normalized = {}
    for key, items in grouped.items():
        ordered = tuple(sorted(items, key=lambda item: item.receipt_id))
        refs = [ref for decision in ordered for ref in decision.provider_receipt_refs]
        if len(refs) != len(set(refs)):
            _fail("routing reconciliation duplicate unit provider receipt")
        normalized[key] = ordered
    return rows, normalized


def _evaluation(spec, claimed, labels, decisions, attempts, billed_total):
    units = (*spec.input.calibration_unit_refs, *spec.input.holdout_unit_refs)
    baseline_predictions = {"calibration": {}, "holdout": {}}
    predictions_by_variant = {}
    receipts_by_variant = {}
    for variant in spec.variants:
        predictions, receipts_by_unit = {}, {}
        for unit in units:
            refs = tuple(
                ref for decision in decisions[(variant.variant_id, unit)]
                for ref in decision.provider_receipt_refs
            )
            receipts = tuple(attempts[ref][1] for ref in refs)
            predicted = any(item.outcome == ProviderOutcome.VERIFIED.value for item in receipts)
            predictions[unit], receipts_by_unit[unit] = predicted, receipts
            if variant.variant_id == spec.baseline_variant_id:
                split = "calibration" if unit in spec.input.calibration_unit_refs else "holdout"
                baseline_predictions[split][unit] = predicted
        predictions_by_variant[variant.variant_id] = predictions
        receipts_by_variant[variant.variant_id] = receipts_by_unit
    variant_docs = []
    for variant in spec.variants:
        calibration = _v2_metrics(
            split="calibration", unit_refs=spec.input.calibration_unit_refs,
            gold_labels=labels, predictions=predictions_by_variant[variant.variant_id],
            receipts_by_unit=receipts_by_variant[variant.variant_id],
            baseline_positive_units={key for key, item in baseline_predictions["calibration"].items() if item},
        )
        holdout = _v2_metrics(
            split="holdout", unit_refs=spec.input.holdout_unit_refs,
            gold_labels=labels, predictions=predictions_by_variant[variant.variant_id],
            receipts_by_unit=receipts_by_variant[variant.variant_id],
            baseline_positive_units={key for key, item in baseline_predictions["holdout"].items() if item},
        )
        precision = calibration.precision >= spec.gates.min_calibration_precision and holdout.precision >= spec.gates.min_holdout_precision
        recall = holdout.recall >= spec.gates.min_holdout_recall
        cost = holdout.no_signal_credit_microunits <= spec.gates.max_holdout_no_signal_credit_microunits
        efficiency = holdout.marginal_verified_positives_per_credit >= spec.gates.min_marginal_verified_positives_per_credit
        decision_refs = sorted(
            decision.receipt_id for unit in units for decision in decisions[(variant.variant_id, unit)]
        )
        provider_refs = sorted({
            ref for unit in units for decision in decisions[(variant.variant_id, unit)]
            for ref in decision.provider_receipt_refs
        })
        variant_docs.append({
            "variant_id": variant.variant_id, "artifact_key": _v2_variant_artifact_key(variant),
            "stage": variant.stage, "calibration": calibration.to_dict(),
            "holdout": holdout.to_dict(), "passed_precision_gate": precision,
            "passed_recall_gate": recall, "passed_cost_gate": cost,
            "passed_efficiency_gate": efficiency,
            "passed": precision and recall and cost and efficiency
            and calibration.adapter_failure_count == 0 and holdout.adapter_failure_count == 0,
            "decision_receipt_refs": decision_refs, "provider_receipt_refs": provider_refs,
        })
    passing = [item for item in variant_docs if item["passed"]]
    selected = ""
    if passing:
        selected = sorted(passing, key=lambda item: (
            item["holdout"]["total_credit_microunits"], len(item["provider_receipt_refs"]),
            -item["holdout"]["unique_rescue_count"], -item["holdout"]["recall"], item["variant_id"],
        ))[0]["variant_id"]
    uses = sum(len(item.provider_receipt_refs) for group in decisions.values() for item in group)
    hits, misses = claimed.get("provider_cache_hits"), claimed.get("provider_cache_misses")
    if type(hits) is not int or type(misses) is not int or hits < 0 or misses < 0 or hits + misses != uses:
        _fail("routing reconciliation cache accounting differs")
    if claimed.get("live_credit_spend") is not spec.allow_live_credit_spend:
        _fail("routing reconciliation live spend flag differs")
    claimed_bill = claimed.get("billing_rollup_total_credit_microunits")
    if type(claimed_bill) is not int or claimed_bill < 0 or claimed_bill > billed_total or (spec.allow_live_credit_spend and claimed_bill != billed_total):
        _fail("routing reconciliation billing total differs")
    authoritative = {
        "contract_version": spec.contract_version, "receipt_id": "routing_evaluation_v2:pending",
        "experiment_id": spec.experiment_id, "experiment_hash": spec.experiment_hash(),
        "variants": variant_docs, "baseline_variant_id": spec.baseline_variant_id,
        "selected_variant_id": selected,
        "decision_receipt_refs": sorted({ref for item in variant_docs for ref in item["decision_receipt_refs"]}),
        "provider_receipt_refs": sorted({ref for item in variant_docs for ref in item["provider_receipt_refs"]}),
        "provider_cache_hits": hits, "provider_cache_misses": misses,
        "billing_rollup_id": claimed.get("billing_rollup_id", ""),
        "billing_rollup_hash": claimed.get("billing_rollup_hash", ""),
        "billing_rollup_total_credit_microunits": claimed_bill,
        "live_credit_spend": spec.allow_live_credit_spend, "immutable": True,
    }
    authoritative["receipt_id"] = "routing_evaluation_v2:" + sha256_json(authoritative).split(":", 1)[1][:16]
    if dict(claimed) != authoritative:
        _fail("routing reconciliation evaluation differs from recomputation")
    if not selected:
        _fail("routing reconciliation has no passing variant")
    return authoritative


def reconcile(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version", "spec_doc", "evaluation_doc", "gold_label_authority",
        "artifact_lineage", "execution_envelope", "decision_receipts",
        "provider_attempts", "budget_events",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields or payload.get("schema_version") != SCHEMA:
        _fail("routing reconciliation fields are invalid")
    spec = _spec(payload)
    claimed = payload["evaluation_doc"]
    if not isinstance(claimed, Mapping):
        _fail("routing reconciliation evaluation is invalid")
    labels, label_manifest_hash = _labels(payload["gold_label_authority"], spec)
    lineage_hash = _lineage(payload["artifact_lineage"], spec)
    envelope = _execution_envelope(
        payload["execution_envelope"],
        spec,
        lineage_hash=lineage_hash,
        label_manifest_hash=label_manifest_hash,
    )
    attempt_rows, attempts = _attempts(
        payload["provider_attempts"], spec, envelope
    )
    budget_rows, billed_total = _budgets(payload["budget_events"], spec, attempts)
    decision_rows, decisions = _decisions(payload["decision_receipts"], spec, attempts)
    evaluation = _evaluation(spec, claimed, labels, decisions, attempts, billed_total)
    return {
        "experiment_hash": spec.experiment_hash(),
        "evaluation_hash": sha256_json(evaluation),
        "evaluation_receipt_id": evaluation["receipt_id"],
        "selected_variant_id": evaluation["selected_variant_id"],
        "decision_receipts_root": _root(decision_rows, "receipt_id"),
        "provider_attempts_root": _root(attempt_rows, "attempt_key"),
        "budget_events_root": _root(budget_rows, "event_key"),
        "artifact_lineage_hash": lineage_hash,
        "gold_label_manifest_hash": label_manifest_hash,
        "execution_envelope_hash": envelope.envelope_hash(),
        "authoritative_billed_credit_microunits": billed_total,
    }


def validate_input(payload: Mapping[str, Any]) -> None:
    reconcile(payload)


def receipt_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "reconciled": True,
        **reconcile(payload),
        "input_root": sha256_json(dict(payload)),
    }


def execute(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = receipt_output(payload)
    output["output_root"] = sha256_json(output)
    return output


__all__ = [
    "RoutingReconciliationError", "build_input", "validate_input", "receipt_output",
    "execute", "reconcile",
]
