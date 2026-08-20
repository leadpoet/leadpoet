"""Signed terminal evidence and billing projection for routing provider calls.

``ProviderReceipt`` is a host projection used by the pure evaluator.  It is
not itself an authority.  This module makes the projection a function of a
coordinator-signed provider evidence record and a scoring-enclave execution
receipt.  Reconciliation can therefore reject a fabricated receipt even when
all of its scalar fields look plausible.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from gateway.tee.provider_evidence_v2 import (
    ProviderEvidenceV2Error,
    validate_signed_provider_evidence_record,
)
from leadpoet_canonical.attested_v2 import (
    create_signed_execution_receipt,
    validate_signed_execution_receipt,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    ProviderOutcome,
    ProviderReceipt,
    ReceiptExecutionMode,
    validate_provider_receipt,
)


ROUTING_PROVIDER_TERMINAL_SCHEMA_V2 = (
    "leadpoet.research_lab.routing_provider_terminal.v2"
)
ROUTING_PROVIDER_TERMINAL_PURPOSE_V2 = "research_lab.routing_provider_evidence.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
_TERMINAL_BODY_FIELDS = {
    "schema_version", "job_id", "experiment_hash", "admission_bundle_hash",
    "authorization_hash", "authorization_proof_hash", "binding", "variant_id",
    "unit_ref", "request_fingerprint", "terminal_status", "provider_record",
    "provider_record_hash", "coordinator_boot_identity", "billing_projection",
    "billing_projection_hash",
}


class RoutingProviderTerminalError(ValueError):
    """Signed provider terminal evidence is invalid or substituted."""


def _hash(value: Any, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise RoutingProviderTerminalError(f"routing provider terminal {name} is invalid")
    return normalized


def _ref(value: Any, name: str) -> str:
    normalized = str(value or "").strip()
    if not _REF_RE.fullmatch(normalized):
        raise RoutingProviderTerminalError(f"routing provider terminal {name} is invalid")
    return normalized


def _projection(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "receipt_ref", "outcome", "evidence_hash", "credit_microunits",
        "latency_ms", "billing_state",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RoutingProviderTerminalError("routing billing projection fields are invalid")
    outcome = str(value["outcome"] or "")
    if outcome not in {item.value for item in ProviderOutcome}:
        raise RoutingProviderTerminalError("routing billing projection outcome is invalid")
    if str(value["billing_state"] or "") != "known":
        raise RoutingProviderTerminalError("routing billing projection is not authoritative")
    receipt_ref = str(value["receipt_ref"] or "")
    if not receipt_ref.startswith("provider_receipt:"):
        raise RoutingProviderTerminalError("routing billing projection receipt is invalid")
    _hash(value["evidence_hash"], "billing evidence hash")
    for name, maximum in (("credit_microunits", 100_000_000), ("latency_ms", 900_000)):
        current = value[name]
        if type(current) is not int or not 0 <= current <= maximum:
            raise RoutingProviderTerminalError(f"routing billing projection {name} is invalid")
    return {
        "receipt_ref": receipt_ref,
        "outcome": outcome,
        "evidence_hash": str(value["evidence_hash"]),
        "credit_microunits": int(value["credit_microunits"]),
        "latency_ms": int(value["latency_ms"]),
        "billing_state": "known",
    }


def _expected_receipt_ref(
    *,
    binding: ProviderBindingIdentity,
    unit_ref: str,
    request_fingerprint: str,
    projection: Mapping[str, Any],
) -> str:
    identity = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id,
        "unit_ref": unit_ref,
        "request_fingerprint": request_fingerprint,
        "outcome": projection["outcome"],
        "evidence_hash": projection["evidence_hash"],
        "credit_microunits": projection["credit_microunits"],
        "latency_ms": projection["latency_ms"],
        "execution_mode": ReceiptExecutionMode.MEASURED_LAB.value,
    }
    return "provider_receipt:" + sha256_json(identity).split(":", 1)[1][:16]


def build_routing_provider_terminal_body_v2(
    *,
    job_id: str,
    experiment_hash: str,
    admission_bundle_hash: str,
    authorization_hash: str,
    authorization_proof_hash: str,
    binding: ProviderBindingIdentity,
    variant_id: str,
    unit_ref: str,
    request_fingerprint: str,
    terminal_status: str,
    provider_record: Mapping[str, Any],
    coordinator_boot_identity: Mapping[str, Any],
    billing_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the redacted terminal body from a signed provider record."""

    for name, value in (
        ("job_id", job_id), ("variant_id", variant_id), ("unit_ref", unit_ref),
    ):
        _ref(value, name)
    for name, value in (
        ("experiment_hash", experiment_hash),
        ("admission_bundle_hash", admission_bundle_hash),
        ("authorization_hash", authorization_hash),
        ("authorization_proof_hash", authorization_proof_hash),
        ("request_fingerprint", request_fingerprint),
    ):
        _hash(value, name)
    if terminal_status not in {"authenticated_response", "transport_failure"}:
        raise RoutingProviderTerminalError("routing provider terminal status is invalid")
    try:
        record = validate_signed_provider_evidence_record(
            provider_record, boot_identity=coordinator_boot_identity
        )
    except (ProviderEvidenceV2Error, TypeError, ValueError) as exc:
        raise RoutingProviderTerminalError(
            "routing provider terminal evidence record is invalid"
        ) from exc
    projection = _projection(billing_projection)
    if projection["receipt_ref"] != _expected_receipt_ref(
        binding=binding,
        unit_ref=unit_ref,
        request_fingerprint=request_fingerprint,
        projection=projection,
    ):
        raise RoutingProviderTerminalError("routing billing projection receipt differs")
    if terminal_status == "transport_failure" and (
        projection["outcome"] != ProviderOutcome.ADAPTER_FAILURE.value
        or record["evidence"] not in {"transport_failure", "replay_miss"}
    ):
        raise RoutingProviderTerminalError(
            "routing transport failure must project as adapter_failure"
        )
    if terminal_status == "authenticated_response" and record["evidence"] == "transport_failure":
        raise RoutingProviderTerminalError(
            "routing authenticated terminal has transport-failure evidence"
        )
    if record["request_fingerprint"] != request_fingerprint.split(":", 1)[-1]:
        raise RoutingProviderTerminalError("routing provider terminal request differs")
    if projection["evidence_hash"] != record["record_hash"]:
        raise RoutingProviderTerminalError(
            "routing billing projection evidence is not the signed provider record"
        )
    if record["transport_attempt_hash"] == "":
        raise RoutingProviderTerminalError("routing provider terminal transport proof is missing")
    body = {
        "schema_version": ROUTING_PROVIDER_TERMINAL_SCHEMA_V2,
        "job_id": str(job_id),
        "experiment_hash": str(experiment_hash),
        "admission_bundle_hash": str(admission_bundle_hash),
        "authorization_hash": str(authorization_hash),
        "authorization_proof_hash": str(authorization_proof_hash),
        "binding": binding.to_dict(),
        "variant_id": str(variant_id),
        "unit_ref": str(unit_ref),
        "request_fingerprint": str(request_fingerprint),
        "terminal_status": str(terminal_status),
        "provider_record": dict(record),
        "provider_record_hash": str(record["record_hash"]),
        "coordinator_boot_identity": dict(coordinator_boot_identity),
        "billing_projection": projection,
        "billing_projection_hash": sha256_json(projection),
    }
    return body


def sign_routing_provider_terminal_v2(
    *,
    body: Mapping[str, Any],
    protected_receipt: Mapping[str, Any],
    enclave_pubkey: str,
    sign_digest: Callable[[bytes], Any],
) -> dict[str, Any]:
    """Sign a terminal body using the same scoring receipt primitive."""

    if not isinstance(body, Mapping) or set(body) != _TERMINAL_BODY_FIELDS:
        raise RoutingProviderTerminalError("routing provider terminal body fields are invalid")
    try:
        validate_signed_execution_receipt(protected_receipt)
    except Exception as exc:
        raise RoutingProviderTerminalError("routing protected receipt is invalid") from exc
    if (
        protected_receipt.get("role") != "gateway_scoring"
        or protected_receipt.get("purpose") != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
        or protected_receipt.get("status") != "succeeded"
        or protected_receipt.get("job_id") != body.get("job_id")
        or str(enclave_pubkey or "").lower()
        != str(protected_receipt.get("enclave_pubkey") or "").lower()
    ):
        raise RoutingProviderTerminalError("routing protected receipt scope differs")
    normalized = dict(body)
    input_root = sha256_json(normalized)
    receipt_body = {
        "schema_version": "leadpoet.attested_execution_receipt.v2",
        "role": "gateway_scoring",
        "purpose": ROUTING_PROVIDER_TERMINAL_PURPOSE_V2,
        "job_id": str(body["job_id"]),
        "epoch_id": int(protected_receipt["epoch_id"]),
        "sequence": int(protected_receipt["sequence"]) + 1,
        "commit_sha": protected_receipt["commit_sha"],
        "pcr0": protected_receipt["pcr0"],
        "build_manifest_hash": protected_receipt["build_manifest_hash"],
        "dependency_lock_hash": protected_receipt["dependency_lock_hash"],
        "config_hash": protected_receipt["config_hash"],
        "boot_identity_hash": protected_receipt["boot_identity_hash"],
        "input_root": input_root,
        "output_root": sha256_json(normalized),
        "transport_root": _hash(body["provider_record_hash"], "transport root"),
        "host_operation_root": _hash(body["authorization_hash"], "host operation root"),
        "artifact_root": _hash(body["billing_projection_hash"], "artifact root"),
        "parent_receipt_hashes": [str(protected_receipt["receipt_hash"])],
        "status": "succeeded",
        "failure_code": None,
        "issued_at": protected_receipt["issued_at"],
    }
    signed = create_signed_execution_receipt(
        body=receipt_body, enclave_pubkey=enclave_pubkey, sign_digest=sign_digest
    )
    return {"body": normalized, "receipt": signed}


def validate_routing_provider_terminal_v2(
    *,
    terminal: Mapping[str, Any],
    binding: ProviderBindingIdentity,
    protected_receipt: Mapping[str, Any],
    expected_job_id: str,
    expected_experiment_hash: str,
    expected_admission_bundle_hash: str,
    expected_authorization_hash: str,
    expected_authorization_proof_hash: str,
) -> dict[str, Any]:
    """Verify terminal signature and derive the only accepted host receipt."""

    if not isinstance(terminal, Mapping) or set(terminal) != {"body", "receipt"}:
        raise RoutingProviderTerminalError("routing provider terminal fields are invalid")
    body = terminal.get("body")
    receipt = terminal.get("receipt")
    if (
        not isinstance(body, Mapping)
        or set(body) != _TERMINAL_BODY_FIELDS
        or body.get("schema_version") != ROUTING_PROVIDER_TERMINAL_SCHEMA_V2
    ):
        raise RoutingProviderTerminalError("routing provider terminal body is invalid")
    if body.get("job_id") != expected_job_id or body.get("experiment_hash") != expected_experiment_hash:
        raise RoutingProviderTerminalError("routing provider terminal job differs")
    for name in (
        "experiment_hash", "admission_bundle_hash", "authorization_hash",
        "authorization_proof_hash", "request_fingerprint",
    ):
        _hash(body.get(name), name)
    for name in ("job_id", "variant_id", "unit_ref"):
        _ref(body.get(name), name)
    if body.get("admission_bundle_hash") != expected_admission_bundle_hash:
        raise RoutingProviderTerminalError("routing provider terminal admission differs")
    if body.get("authorization_hash") != expected_authorization_hash or body.get("authorization_proof_hash") != expected_authorization_proof_hash:
        raise RoutingProviderTerminalError("routing provider terminal authorization differs")
    projection = _projection(body.get("billing_projection") or {})
    try:
        body_binding = ProviderBindingIdentity.from_mapping(body.get("binding") or {})
    except Exception as exc:
        raise RoutingProviderTerminalError("routing provider terminal binding is invalid") from exc
    if body_binding != binding:
        raise RoutingProviderTerminalError("routing provider terminal binding differs")
    if projection["receipt_ref"] != _expected_receipt_ref(
        binding=binding,
        unit_ref=str(body["unit_ref"]),
        request_fingerprint=str(body["request_fingerprint"]),
        projection=projection,
    ):
        raise RoutingProviderTerminalError("routing billing projection receipt differs")
    if body.get("billing_projection_hash") != sha256_json(projection):
        raise RoutingProviderTerminalError("routing provider billing proof hash differs")
    if body.get("provider_record_hash") != (body.get("provider_record") or {}).get("record_hash"):
        raise RoutingProviderTerminalError("routing provider evidence hash differs")
    try:
        provider_record = validate_signed_provider_evidence_record(
            body.get("provider_record") or {},
            boot_identity=body.get("coordinator_boot_identity") or {},
        )
    except Exception as exc:
        raise RoutingProviderTerminalError(
            "routing provider evidence signature is invalid"
        ) from exc
    if (
        provider_record["request_fingerprint"]
        != str(body["request_fingerprint"]).split(":", 1)[-1]
        or body.get("billing_projection", {}).get("evidence_hash")
        != provider_record["record_hash"]
    ):
        raise RoutingProviderTerminalError("routing provider terminal evidence differs")
    try:
        validate_signed_execution_receipt(receipt)
    except Exception as exc:
        raise RoutingProviderTerminalError("routing provider terminal signature is invalid") from exc
    if (
        receipt.get("role") != "gateway_scoring"
        or receipt.get("purpose") != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
        or receipt.get("status") != "succeeded"
        or receipt.get("job_id") != expected_job_id
        or receipt.get("enclave_pubkey") != protected_receipt.get("enclave_pubkey")
        or receipt.get("input_root") != sha256_json(dict(body))
        or receipt.get("output_root") != sha256_json(dict(body))
        or receipt.get("host_operation_root") != expected_authorization_hash
        or receipt.get("artifact_root") != body.get("billing_projection_hash")
        or receipt.get("parent_receipt_hashes")
        != [protected_receipt.get("receipt_hash")]
        or any(
            receipt.get(name) != protected_receipt.get(name)
            for name in (
                "commit_sha", "pcr0", "build_manifest_hash", "dependency_lock_hash",
                "config_hash", "boot_identity_hash",
            )
        )
    ):
        raise RoutingProviderTerminalError("routing provider terminal receipt differs")
    identity = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id,
        "unit_ref": str(body["unit_ref"]),
        "request_fingerprint": str(body["request_fingerprint"]),
        "outcome": projection["outcome"],
        "evidence_hash": projection["evidence_hash"],
        "credit_microunits": projection["credit_microunits"],
        "latency_ms": projection["latency_ms"],
        "execution_mode": ReceiptExecutionMode.MEASURED_LAB.value,
    }
    receipt_identity = dict(identity)
    receipt_identity["receipt_ref"] = "provider_receipt:" + sha256_json(
        receipt_identity
    ).split(":", 1)[1][:16]
    result = ProviderReceipt(**receipt_identity)
    errors = validate_provider_receipt(result)
    if errors:
        raise RoutingProviderTerminalError("routing provider projected receipt is invalid")
    return result.to_dict()


__all__ = [
    "ROUTING_PROVIDER_TERMINAL_SCHEMA_V2",
    "ROUTING_PROVIDER_TERMINAL_PURPOSE_V2",
    "RoutingProviderTerminalError",
    "build_routing_provider_terminal_body_v2",
    "sign_routing_provider_terminal_v2",
    "validate_routing_provider_terminal_v2",
]
