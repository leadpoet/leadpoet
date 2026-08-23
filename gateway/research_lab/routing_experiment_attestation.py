"""Deterministic protected-operation contract for routing experiment promotion.

The scoring enclave receives hashes and redacted receipts only.  It never
receives a provider request, a provider response body, or a credential.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from research_lab.canonical import sha256_json


ROUTING_EXPERIMENT_ATTESTATION_OPERATION_V2 = "attest_routing_experiment_v2"
ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2 = "research_lab.routing_experiment.v2"
ROUTING_EXPERIMENT_ATTESTATION_SCHEMA_V2 = (
    "leadpoet.research_lab.routing_experiment_attestation.v2"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")


class RoutingExperimentAttestationError(ValueError):
    """The protected routing-evaluation payload is not self-consistent."""


def build_routing_experiment_attestation_input_v2(
    *,
    experiment_hash: str,
    evaluation_hash: str,
    evaluation_receipt_id: str,
    selected_variant_id: str,
    decision_receipt_refs: tuple[str, ...],
    provider_receipt_refs: tuple[str, ...],
    decision_receipts_root: str,
    provider_attempts_root: str,
    budget_events_root: str,
    billing_rollup_hash: str = "",
    billing_rollup_total_credit_microunits: int = 0,
) -> dict[str, Any]:
    """Build the exact redacted input committed by the scoring enclave."""

    payload = {
        "schema_version": ROUTING_EXPERIMENT_ATTESTATION_SCHEMA_V2,
        "experiment_hash": str(experiment_hash),
        "evaluation_hash": str(evaluation_hash),
        "evaluation_receipt_id": str(evaluation_receipt_id),
        "selected_variant_id": str(selected_variant_id),
        "decision_receipt_refs": list(decision_receipt_refs),
        "provider_receipt_refs": list(provider_receipt_refs),
        "decision_receipts_root": str(decision_receipts_root),
        "provider_attempts_root": str(provider_attempts_root),
        "budget_events_root": str(budget_events_root),
        "billing_rollup_hash": str(billing_rollup_hash),
        "billing_rollup_total_credit_microunits": billing_rollup_total_credit_microunits,
    }
    validate_routing_experiment_attestation_input_v2(payload)
    return payload


def validate_routing_experiment_attestation_input_v2(payload: Mapping[str, Any]) -> None:
    """Reject incomplete, duplicate, or unbound promotion evidence."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "experiment_hash",
        "evaluation_hash",
        "evaluation_receipt_id",
        "selected_variant_id",
        "decision_receipt_refs",
        "provider_receipt_refs",
        "decision_receipts_root",
        "provider_attempts_root",
        "budget_events_root",
        "billing_rollup_hash",
        "billing_rollup_total_credit_microunits",
    }:
        raise RoutingExperimentAttestationError("routing experiment attestation fields are invalid")
    if payload.get("schema_version") != ROUTING_EXPERIMENT_ATTESTATION_SCHEMA_V2:
        raise RoutingExperimentAttestationError("routing experiment attestation schema is invalid")
    for field_name in (
        "experiment_hash",
        "evaluation_hash",
        "decision_receipts_root",
        "provider_attempts_root",
        "budget_events_root",
    ):
        if not _HASH_RE.fullmatch(str(payload.get(field_name) or "")):
            raise RoutingExperimentAttestationError(
                f"routing experiment attestation {field_name} is invalid"
            )
    billing_hash = str(payload.get("billing_rollup_hash") or "")
    if billing_hash and not _HASH_RE.fullmatch(billing_hash):
        raise RoutingExperimentAttestationError("routing experiment attestation billing hash is invalid")
    for field_name in ("evaluation_receipt_id", "selected_variant_id"):
        if not _REF_RE.fullmatch(str(payload.get(field_name) or "")):
            raise RoutingExperimentAttestationError(
                f"routing experiment attestation {field_name} is invalid"
            )
    for field_name, prefix in (
        ("decision_receipt_refs", "routing_decision:"),
        ("provider_receipt_refs", "provider_receipt:"),
    ):
        refs = payload.get(field_name)
        if not isinstance(refs, list) or not refs or any(
            not isinstance(item, str) or not item.startswith(prefix) for item in refs
        ):
            raise RoutingExperimentAttestationError(
                f"routing experiment attestation {field_name} is invalid"
            )
        if refs != sorted(set(refs)):
            raise RoutingExperimentAttestationError(
                f"routing experiment attestation {field_name} is not canonical"
            )
    if (
        type(payload.get("billing_rollup_total_credit_microunits")) is not int
        or int(payload["billing_rollup_total_credit_microunits"]) < 0
    ):
        raise RoutingExperimentAttestationError("routing experiment attestation billing total is invalid")


def routing_experiment_attestation_receipt_output_v2(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact object whose hash is the scoring receipt output root.

    The enclave signs this object directly.  ``execute_*`` adds that signed
    output root as a convenience field for the host/store boundary, rather
    than creating a self-referential hash field in the enclave response.
    """

    validate_routing_experiment_attestation_input_v2(payload)
    normalized = dict(payload)
    input_root = sha256_json(normalized)
    return {
        "schema_version": "leadpoet.research_lab.routing_experiment_attestation_result.v2",
        "reconciled": True,
        "experiment_hash": normalized["experiment_hash"],
        "evaluation_hash": normalized["evaluation_hash"],
        "evaluation_receipt_id": normalized["evaluation_receipt_id"],
        "selected_variant_id": normalized["selected_variant_id"],
        "decision_receipts_root": normalized["decision_receipts_root"],
        "provider_attempts_root": normalized["provider_attempts_root"],
        "budget_events_root": normalized["budget_events_root"],
        "input_root": input_root,
    }


def execute_routing_experiment_attestation_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the immutable evaluation commitment for an attested run."""

    output = routing_experiment_attestation_receipt_output_v2(payload)
    output["output_root"] = sha256_json(output)
    return output


# Security upgrade: the initial PR93 draft accepted caller-computed roots and
# pass flags. Keep its private helpers above for review history, but replace
# every public entry point with the canonical document reconciler. This avoids
# deleting an existing module while making the hash-only path unreachable.
from gateway.research_lab import routing_experiment_reconciliation as _secure  # noqa: E402


def build_routing_experiment_attestation_input_v2(  # type: ignore[no-redef]
    *,
    spec_doc: Mapping[str, Any],
    evaluation_doc: Mapping[str, Any],
    gold_label_authority: Mapping[str, Any],
    artifact_lineage: Mapping[str, Any],
    execution_envelope: Mapping[str, Any],
    decision_receipts: tuple[Mapping[str, Any], ...],
    provider_attempts: tuple[Mapping[str, Any], ...],
    budget_events: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    try:
        return _secure.build_input(
            spec_doc=spec_doc,
            evaluation_doc=evaluation_doc,
            gold_label_authority=gold_label_authority,
            artifact_lineage=artifact_lineage,
            execution_envelope=execution_envelope,
            decision_receipts=decision_receipts,
            provider_attempts=provider_attempts,
            budget_events=budget_events,
        )
    except _secure.RoutingReconciliationError as exc:
        raise RoutingExperimentAttestationError(str(exc)) from exc


def validate_routing_experiment_attestation_input_v2(  # type: ignore[no-redef]
    payload: Mapping[str, Any],
) -> None:
    try:
        _secure.validate_input(payload)
    except _secure.RoutingReconciliationError as exc:
        raise RoutingExperimentAttestationError(str(exc)) from exc


def routing_experiment_attestation_receipt_output_v2(  # type: ignore[no-redef]
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _secure.receipt_output(payload)
    except _secure.RoutingReconciliationError as exc:
        raise RoutingExperimentAttestationError(str(exc)) from exc


def execute_routing_experiment_attestation_v2(  # type: ignore[no-redef]
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _secure.execute(payload)
    except _secure.RoutingReconciliationError as exc:
        raise RoutingExperimentAttestationError(str(exc)) from exc


__all__ = [
    "ROUTING_EXPERIMENT_ATTESTATION_OPERATION_V2",
    "ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2",
    "ROUTING_EXPERIMENT_ATTESTATION_SCHEMA_V2",
    "RoutingExperimentAttestationError",
    "build_routing_experiment_attestation_input_v2",
    "validate_routing_experiment_attestation_input_v2",
    "routing_experiment_attestation_receipt_output_v2",
    "execute_routing_experiment_attestation_v2",
]
