"""Protected, deterministic post-call normalizer for routing providers.

This module is deliberately separate from :mod:`routing_provider_terminal`.
The latter is a legacy host projection and signing helper.  This boundary
accepts only the complete pre-call grant, the immutable prepared call, the
compiler-owned broker request/result, and coordinator-signed provider
evidence.  It derives the provider receipt inside the boundary and returns
commitments for the normal ``ExecutionJobManagerV2`` receipt.  It does not
sign a second, custom terminal receipt and it never accepts caller supplied
outcome or cost values.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime
import json
import re
from typing import Any, Mapping

from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    routing_provider_dispatch_job_id_v2,
)
from gateway.research_lab.routing_provider_bindings import (
    DEEPLINE_ACTION_POLICIES,
    PreparedRoutingProviderCall,
    ReviewedDeeplineActionCompiler,
    RoutingProviderBindingError,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.tee.provider_broker_v2 import (
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2Error,
    validate_routing_authorization_proof_v2,
)
from gateway.tee.provider_evidence_v2 import (
    ProviderEvidenceV2Error,
    validate_signed_provider_evidence_record,
)
from leadpoet_canonical.attested_v2 import (
    sha256_bytes,
    validate_transport_attempt,
)
from research_lab.canonical import sha256_json
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    ProviderReceipt,
    ReceiptExecutionMode,
    validate_provider_receipt,
)


ROUTING_PROVIDER_TERMINAL_OPERATION_V2 = "routing_provider_terminal_v2"
ROUTING_PROVIDER_TERMINAL_RESULT_SCHEMA_V2 = (
    "leadpoet.routing_provider_terminal_result.v2"
)
ROUTING_PROVIDER_TERMINAL_PURPOSE_V2 = "research_lab.routing_provider_evidence.v2"
# The dispatch operation is deliberately separate from the older terminal
# normalizer.  The host supplies only the signed grant, compiler projection,
# and exact broker request; the scoring enclave obtains the coordinator result
# through its measured provider path before invoking the same normalizer.
ROUTING_PROVIDER_DISPATCH_OPERATION_V2 = "routing_provider_dispatch_v2"
ROUTING_PROVIDER_DISPATCH_REQUEST_SCHEMA_V2 = (
    "leadpoet.routing_provider_dispatch_request.v2"
)
ROUTING_PROVIDER_DISPATCH_PURPOSE_V2 = ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
ROUTING_BUDGET_RESERVATION_SCHEMA_V3 = (
    "leadpoet.research_lab.routing_budget_reservation.v3"
)
ROUTING_BUDGET_RESERVATION_RESULT_SCHEMA_V3 = (
    "leadpoet.research_lab.routing_budget_reservation_result.v3"
)
ROUTING_BUDGET_RESERVATION_PROOF_SCHEMA_V3 = (
    "leadpoet.research_lab.routing_budget_reservation_proof.v3"
)
ROUTING_BUDGET_RESERVATION_PURPOSE_V3 = (
    ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")


class ProtectedRoutingProviderTerminalError(ValueError):
    """A protected routing terminal input is not exact or authoritative."""


def build_routing_budget_reservation_v3(
    *,
    authorization: RoutingProviderCallAuthorizationV2,
    prepared_call: PreparedRoutingProviderCall,
    lease_seconds: int,
) -> dict[str, Any]:
    """Build the one exact durable reservation required before dispatch.

    The document contains no bearer capability. The database verifies the
    queue-bound claim key and generation against its current lease. The
    scoring enclave rebuilds this document from the signed authorization, so
    a host cannot reserve a different binding, request, or credit amount.
    """

    if not isinstance(authorization, RoutingProviderCallAuthorizationV2):
        raise ProtectedRoutingProviderTerminalError(
            "routing budget authorization is invalid"
        )
    if not isinstance(prepared_call, PreparedRoutingProviderCall):
        raise ProtectedRoutingProviderTerminalError(
            "routing budget prepared call is invalid"
        )
    if prepared_call.binding != authorization.binding:
        raise ProtectedRoutingProviderTerminalError(
            "routing budget binding differs"
        )
    if (
        prepared_call.action_id != authorization.action_id
        or prepared_call.unit_ref != authorization.unit_ref
        or prepared_call.request_body_hash != authorization.request_body_hash
        or prepared_call.binding_catalog_manifest_hash
        != authorization.binding_catalog_manifest_hash
        or prepared_call.credit_ceiling_microunits
        != authorization.credit_cap_microunits
        or prepared_call.timeout_ms != authorization.timeout_ms
    ):
        raise ProtectedRoutingProviderTerminalError(
            "routing budget prepared call differs"
        )
    if (
        type(lease_seconds) is not int
        or not 1 <= lease_seconds <= 3600
        or lease_seconds * 1000 < authorization.timeout_ms
    ):
        raise ProtectedRoutingProviderTerminalError(
            "routing budget lease is invalid"
        )
    authorization_hash = authorization.authorization_hash()
    reservation_id = (
        "routing-reservation:"
        + authorization_hash.split(":", 1)[1][:32]
    )
    event_doc = {
        "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
        "reservation_id": reservation_id,
        "binding_id": authorization.binding.binding_id,
        "call_grant_hash": authorization_hash,
        "action_id": authorization.action_id,
        "tool_id": authorization.binding.tool_id,
        "variant_id": authorization.variant_id,
        "unit_ref": authorization.unit_ref,
        "attempt": authorization.attempt,
        "request_fingerprint": authorization.core_request_fingerprint,
        "request_body_hash": authorization.request_body_hash,
        "binding_catalog_manifest_hash": (
            authorization.binding_catalog_manifest_hash
        ),
    }
    event_key = sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
            "kind": "reserve",
            "reservation_id": reservation_id,
            "parts": [authorization_hash],
        }
    )
    return {
        "schema_version": ROUTING_BUDGET_RESERVATION_SCHEMA_V3,
        "event_key": event_key,
        "reservation_id": reservation_id,
        "experiment_hash": authorization.experiment_hash,
        "binding_id": authorization.binding.binding_id,
        "claim_key": authorization.claim_key,
        "claim_generation": authorization.claim_generation,
        "claim_fence_hash": authorization.claim_fence_hash,
        "credit_microunits": authorization.credit_cap_microunits,
        "lease_seconds": lease_seconds,
        "event_doc": event_doc,
    }


def validate_routing_budget_reservation_v3(
    value: Mapping[str, Any],
    *,
    authorization: RoutingProviderCallAuthorizationV2,
    prepared_call: PreparedRoutingProviderCall,
) -> dict[str, Any]:
    """Return the exact reservation after rebuilding its signed identity."""

    if not isinstance(value, Mapping):
        raise ProtectedRoutingProviderTerminalError(
            "routing budget reservation is invalid"
        )
    lease_seconds = value.get("lease_seconds")
    if type(lease_seconds) is not int:
        raise ProtectedRoutingProviderTerminalError(
            "routing budget reservation lease is invalid"
        )
    expected = build_routing_budget_reservation_v3(
        authorization=authorization,
        prepared_call=prepared_call,
        lease_seconds=lease_seconds,
    )
    if dict(value) != expected:
        raise ProtectedRoutingProviderTerminalError(
            "routing budget reservation differs"
        )
    return expected


def validate_routing_budget_reservation_result_v3(
    value: Mapping[str, Any],
    *,
    reservation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact PostgREST result returned by the durable authority."""

    required = {
        "schema_version",
        "reserved",
        "idempotent",
        "reservation_id",
        "event_key",
        "experiment_hash",
        "binding_id",
        "claim_key",
        "claim_generation",
        "credit_microunits",
        "lease_expires_at",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ProtectedRoutingProviderTerminalError(
            "routing budget reservation result fields are invalid"
        )
    if (
        value.get("schema_version")
        != ROUTING_BUDGET_RESERVATION_RESULT_SCHEMA_V3
        or value.get("reserved") is not True
        or type(value.get("idempotent")) is not bool
        or value.get("reservation_id") != reservation.get("reservation_id")
        or value.get("event_key") != reservation.get("event_key")
        or value.get("experiment_hash") != reservation.get("experiment_hash")
        or value.get("binding_id") != reservation.get("binding_id")
        or value.get("claim_key") != reservation.get("claim_key")
        or value.get("claim_generation") != reservation.get("claim_generation")
        or value.get("credit_microunits") != reservation.get("credit_microunits")
    ):
        raise ProtectedRoutingProviderTerminalError(
            "routing budget reservation result differs"
        )
    expires_at = str(value.get("lease_expires_at") or "")
    try:
        parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtectedRoutingProviderTerminalError(
            "routing budget reservation expiry is invalid"
        ) from exc
    if parsed_expiry.tzinfo is None:
        raise ProtectedRoutingProviderTerminalError(
            "routing budget reservation expiry is invalid"
        )
    return dict(value)


def routing_budget_reservation_proof_v3(
    *,
    reservation_result: Mapping[str, Any],
    response_hash: str,
    transport_attempt_hash: str,
) -> dict[str, Any]:
    """Build the redacted proof committed by the protected dispatch receipt."""

    response_hash = _hash(response_hash, "budget reservation response_hash")
    transport_attempt_hash = _hash(
        transport_attempt_hash,
        "budget reservation transport_attempt_hash",
    )
    return {
        "schema_version": ROUTING_BUDGET_RESERVATION_PROOF_SCHEMA_V3,
        "reservation_id": str(reservation_result["reservation_id"]),
        "event_key": str(reservation_result["event_key"]),
        "experiment_hash": str(reservation_result["experiment_hash"]),
        "binding_id": str(reservation_result["binding_id"]),
        "claim_key": str(reservation_result["claim_key"]),
        "claim_generation": int(reservation_result["claim_generation"]),
        "credit_microunits": int(reservation_result["credit_microunits"]),
        "lease_expires_at": str(reservation_result["lease_expires_at"]),
        "response_hash": response_hash,
        "transport_attempt_hash": transport_attempt_hash,
    }


def _hash(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(text):
        raise ProtectedRoutingProviderTerminalError(
            f"routing protected terminal {name} is invalid"
        )
    return text


def _canonical_body(value: Any, *, name: str) -> bytes:
    try:
        body = base64.b64decode(str(value or ""), validate=True)
    except Exception as exc:  # noqa: BLE001 - boundary conversion
        raise ProtectedRoutingProviderTerminalError(
            f"routing protected terminal {name} is not base64"
        ) from exc
    if not body or len(body) > 8 * 1024 * 1024:
        raise ProtectedRoutingProviderTerminalError(
            f"routing protected terminal {name} is out of bounds"
        )
    try:
        document = json.loads(body)
    except Exception as exc:  # noqa: BLE001 - boundary conversion
        raise ProtectedRoutingProviderTerminalError(
            f"routing protected terminal {name} is not JSON"
        ) from exc
    try:
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - boundary conversion
        raise ProtectedRoutingProviderTerminalError(
            f"routing protected terminal {name} cannot be canonicalized"
        ) from exc
    if canonical != body:
        raise ProtectedRoutingProviderTerminalError(
            f"routing protected terminal {name} is not canonical JSON"
        )
    return body


def _prepared_projection(prepared: PreparedRoutingProviderCall) -> dict[str, Any]:
    if type(prepared) is not PreparedRoutingProviderCall:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal prepared call is not compiler-owned"
        )
    try:
        projection = dict(prepared.authorization_projection())
        body = {
            "provider": prepared.provider,
            "operation": prepared.operation,
            "payload": dict(prepared.payload),
        }
        if sha256_json(body) != prepared.request_body_hash:
            raise ValueError("request body hash differs")
        expected_context_hash = sha256_json(
            {
                "schema_version": "leadpoet.routing_validation_context.v1",
                "action_id": prepared.action_id,
                "context": dict(prepared.validation_context),
            }
        )
        if expected_context_hash != prepared.validation_context_hash:
            raise ValueError("validation context hash differs")
        return projection
    except Exception as exc:  # noqa: BLE001 - boundary conversion
        if isinstance(exc, ProtectedRoutingProviderTerminalError):
            raise
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal prepared call is inconsistent"
        ) from exc


def prepared_routing_provider_call_from_mapping(
    value: Mapping[str, Any],
) -> PreparedRoutingProviderCall:
    """Decode the exact compiler projection transported in a job payload."""

    expected = {
        "binding", "binding_manifest_hash", "binding_catalog_manifest_hash",
        "binding_catalog_version", "unit_ref", "unit_input_hash",
        "unit_dataset_manifest_hash", "unit_set_hash",
        "model_binding_requirements_hash", "action_id", "transport_id",
        "provider", "operation", "payload", "validation_context",
        "validation_context_hash", "request_body_hash", "timeout_ms",
        "credit_ceiling_microunits", "max_results", "retry_policy_hash",
        "evidence_contract_hash", "output_contract_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal prepared call fields are invalid"
        )
    try:
        fields = dict(value)
        fields["binding"] = ProviderBindingIdentity.from_mapping(fields["binding"])
        prepared = PreparedRoutingProviderCall(**fields)
    except Exception as exc:  # noqa: BLE001 - boundary conversion
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal prepared call is invalid"
        ) from exc
    _prepared_projection(prepared)
    return prepared


def _validate_transport_identity(
    *,
    attempt: Mapping[str, Any],
    request: Mapping[str, Any],
    raw_response_body: bytes,
) -> None:
    try:
        validate_transport_attempt(attempt)
    except Exception as exc:  # noqa: BLE001 - boundary conversion
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal transport attempt is invalid"
        ) from exc
    expected = {
        "logical_operation_id": request.get("logical_operation_id"),
        "job_id": request.get("job_id"),
        "purpose": request.get("purpose"),
        "provider_id": request.get("provider_id"),
        "attempt_number": request.get("attempt_number"),
        "method": "POST",
        "timeout_ms": request.get("timeout_ms"),
        "retry_policy_hash": request.get("retry_policy_hash"),
    }
    if any(attempt.get(name) != value for name, value in expected.items()):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal transport identity differs"
        )
    try:
        parsed = json.loads(base64.b64decode(str(request["body_b64"]), validate=True))
        path = str(request["url"]).split("?", 1)[0]
    except Exception as exc:  # noqa: BLE001 - boundary conversion
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal request is not canonical"
        ) from exc
    del parsed, path
    request_body = base64.b64decode(str(request["body_b64"]), validate=True)
    if attempt.get("body_hash") != sha256_bytes(request_body):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal request body hash differs"
        )
    if attempt.get("terminal_status") != "authenticated_response":
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal attempt is not authenticated"
        )
    if attempt.get("response_hash") != sha256_bytes(raw_response_body):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal response hash differs"
        )


def execute_protected_routing_provider_terminal_v2(
    *,
    authorization_proof: Mapping[str, Any],
    prepared_call: PreparedRoutingProviderCall,
    broker_request: Mapping[str, Any],
    broker_result: Mapping[str, Any],
    provider_record: Mapping[str, Any],
    trusted_coordinator_boot_identity: Mapping[str, Any],
    raw_response_body: bytes,
    binding_catalog: VerifiedRoutingBindingCatalog,
    unit_dataset: VerifiedRoutingUnitDataset,
) -> dict[str, Any]:
    """Normalize one authenticated provider response inside the boundary.

    The function intentionally accepts no ``outcome`` or ``cost`` parameter.
    Both values are derived from the closed compiler and signed provider
    record. ``trusted_coordinator_boot_identity`` must come from authenticated
    protected composition; the caller must not replace it with a host-built
    identity. The ordinary execution job manager owns the final receipt and
    roots; this function never signs a custom transport/host/artifact receipt.
    """

    if type(binding_catalog) is not VerifiedRoutingBindingCatalog:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal binding catalog is not reviewed"
        )
    if type(unit_dataset) is not VerifiedRoutingUnitDataset:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal unit dataset is not reviewed"
        )
    # The compiler is created only after the protected boundary has accepted
    # the exact signed catalog and immutable unit dataset.  Callers cannot
    # inject an adapter or a custom response validator.
    compiler = ReviewedDeeplineActionCompiler(
        binding_catalog=binding_catalog,
        unit_dataset=unit_dataset,
    )
    _prepared_projection(prepared_call)
    try:
        validate_routing_authorization_proof_v2(
            authorization_proof, broker_request
        )
        authorization = RoutingProviderCallAuthorizationV2.from_mapping(
            authorization_proof["authorization"]
        )
        authorization_receipt = authorization_proof["authorization_receipt"]
        if not isinstance(authorization_receipt, Mapping):
            raise TypeError("authorization receipt is not an object")
    except Exception as exc:  # noqa: BLE001 - signature and identity boundary
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal authorization proof is invalid"
        ) from exc
    try:
        policy = DEEPLINE_ACTION_POLICIES[prepared_call.action_id]
        expected_prepared = compiler.prepare(
            binding=prepared_call.binding,
            unit_ref=prepared_call.unit_ref,
            authorization_credit_microunits=authorization.credit_cap_microunits,
            authorization_timeout_ms=authorization.timeout_ms,
            expected_model_binding_requirements_hash=(
                prepared_call.model_binding_requirements_hash
            ),
            phase=(
                "conditional_confirmation"
                if prepared_call.action_id == "builtwith_domain_lookup"
                else "initial"
            ),
            execution_mode=(
                "offline_replay"
                if prepared_call.action_id == "builtwith_domain_lookup"
                else "measured_lab"
            ),
        )
    except (KeyError, RoutingProviderBindingError, TypeError, ValueError) as exc:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal prepared call is not in the reviewed catalog"
        ) from exc
    if expected_prepared != prepared_call:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal prepared call differs from reviewed catalog"
        )
    if (
        prepared_call.provider != policy.provider
        or prepared_call.operation != policy.operation
    ):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal model provider identity differs"
        )
    if (
        broker_request.get("schema_version") != PROVIDER_BROKER_SCHEMA_VERSION
        or broker_request.get("purpose") != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
        or broker_request.get("job_id")
        != routing_provider_dispatch_job_id_v2(authorization_proof)
        or broker_request.get("provider_id") != authorization.transport_id
        or broker_request.get("attempt_number") != authorization.attempt
        or broker_request.get("timeout_ms") != authorization.timeout_ms
        or prepared_call.binding != authorization.binding
        or prepared_call.transport_id != authorization.transport_id
        or prepared_call.action_id != authorization.action_id
        or prepared_call.request_body_hash != authorization.request_body_hash
        or prepared_call.credit_ceiling_microunits != authorization.credit_cap_microunits
        or prepared_call.timeout_ms != authorization.timeout_ms
        or prepared_call.retry_policy_hash != authorization.retry_policy_hash
        or prepared_call.unit_ref != authorization.unit_ref
    ):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal authorization/prepared projection differs"
        )
    request_body = _canonical_body(broker_request.get("body_b64"), name="request body")
    if sha256_json(json.loads(request_body)) != prepared_call.request_body_hash:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal request body differs"
        )
    if request_body != base64.b64decode(str(broker_request.get("body_b64")), validate=True):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal request body changed"
        )
    if not isinstance(raw_response_body, (bytes, bytearray)):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal response body is invalid"
        )
    response_body = bytes(raw_response_body)
    if not broker_result.get("terminal_status") == "authenticated_response":
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal requires authenticated provider response"
        )
    try:
        result_body = base64.b64decode(
            str(broker_result.get("body_b64") or ""), validate=True
        )
    except Exception as exc:  # noqa: BLE001
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal broker response body is invalid"
        ) from exc
    if result_body != response_body:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal broker response differs"
        )
    try:
        record = validate_signed_provider_evidence_record(
            provider_record, boot_identity=trusted_coordinator_boot_identity
        )
    except (ProviderEvidenceV2Error, TypeError, ValueError) as exc:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal provider evidence signature is invalid"
        ) from exc
    record_body_hash = _hash(record.get("body_hash"), "provider body hash")
    response_hash = sha256_bytes(response_body)
    if record_body_hash != response_hash:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal signed body hash differs"
        )
    if record.get("status") != broker_result.get("http_status"):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal provider status differs"
        )
    try:
        request_fingerprint = canonical_request_fingerprint(
            str(broker_request.get("method") or "POST"),
            str(broker_request.get("url") or ""),
            base64.b64decode(str(broker_request["body_b64"]), validate=True),
        )
    except Exception as exc:  # noqa: BLE001 - exact broker request boundary
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal request fingerprint cannot be derived"
        ) from exc
    if record.get("request_fingerprint") != request_fingerprint:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal request fingerprint differs"
        )
    for field in ("provider_record", "routing_provider_record"):
        signed_record = broker_result.get(field)
        if signed_record is not None and signed_record != dict(provider_record):
            raise ProtectedRoutingProviderTerminalError(
                "routing protected terminal provider record differs"
            )
    attempt = broker_result.get("transport_attempt")
    if not isinstance(attempt, Mapping):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal transport attempt is missing"
        )
    _validate_transport_identity(
        attempt=attempt, request=broker_request, raw_response_body=response_body
    )
    if record.get("transport_attempt_hash") != attempt.get("attempt_hash"):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal signed attempt hash differs"
        )
    if record.get("request_hash") != attempt.get("request_hash"):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal signed request hash differs"
        )
    try:
        derived = dict(
            compiler.project_result(
                prepared=prepared_call,
                broker_request=broker_request,
                broker_result=broker_result,
                core_request_fingerprint=authorization.core_request_fingerprint,
            )
        )
    except (RoutingProviderBindingError, TypeError, ValueError) as exc:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal compiler validation failed"
        ) from exc
    if derived.get("billing_state") != "known":
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal billing is uncertain"
        )
    allowed_projection = {
        "outcome", "evidence_hash", "credit_microunits", "latency_ms",
        "billing_state", "binding_id", "provider_id", "tool_id",
        "request_fingerprint",
    }
    if set(derived) != allowed_projection:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal compiler projection is incomplete"
        )
    if derived["evidence_hash"] == record["record_hash"]:
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal evidence projection is not distinct"
        )
    if (
        derived["binding_id"] != prepared_call.binding.binding_id
        or derived["provider_id"] != prepared_call.binding.provider_id
        or derived["tool_id"] != prepared_call.binding.tool_id
        or derived["request_fingerprint"] != authorization.core_request_fingerprint
        or type(derived["credit_microunits"]) is not int
        or type(derived["latency_ms"]) is not int
        or derived["credit_microunits"] > prepared_call.credit_ceiling_microunits
        or derived["latency_ms"] > prepared_call.timeout_ms
    ):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal compiler identity differs"
        )
    identity = {
        "binding_id": prepared_call.binding.binding_id,
        "tool_id": prepared_call.binding.tool_id,
        "binding_version": prepared_call.binding.adapter_version,
        "source_lineage_id": prepared_call.binding.source_lineage_id,
        "unit_ref": prepared_call.unit_ref,
        "request_fingerprint": authorization.core_request_fingerprint,
        "outcome": derived["outcome"],
        "evidence_hash": derived["evidence_hash"],
        "credit_microunits": derived["credit_microunits"],
        "latency_ms": derived["latency_ms"],
        "execution_mode": ReceiptExecutionMode.MEASURED_LAB.value,
    }
    provider_receipt = ProviderReceipt(
        receipt_ref="provider_receipt:" + sha256_json(identity).split(":", 1)[1][:16],
        **identity,
    )
    if validate_provider_receipt(provider_receipt):
        raise ProtectedRoutingProviderTerminalError(
            "routing protected terminal provider receipt is invalid"
        )
    provider_record_hash = _hash(record.get("record_hash"), "provider record hash")
    output = {
        "schema_version": ROUTING_PROVIDER_TERMINAL_RESULT_SCHEMA_V2,
        "operation": ROUTING_PROVIDER_TERMINAL_OPERATION_V2,
        "terminal_status": "authenticated_response",
        "authorization_hash": authorization_proof["authorization_hash"],
        "authorization_proof_hash": authorization_proof["authorization_proof_hash"],
        "binding": prepared_call.binding.to_dict(),
        "unit_ref": prepared_call.unit_ref,
        "request_fingerprint": authorization.core_request_fingerprint,
        "provider_record_hash": provider_record_hash,
        "provider_body_hash": record_body_hash,
        "transport_attempt_hash": attempt["attempt_hash"],
        "projection": derived,
        "provider_receipt": provider_receipt.to_dict(),
    }
    return output


__all__ = [
    "ROUTING_PROVIDER_TERMINAL_OPERATION_V2",
    "ROUTING_PROVIDER_TERMINAL_RESULT_SCHEMA_V2",
    "ROUTING_PROVIDER_TERMINAL_PURPOSE_V2",
    "ROUTING_PROVIDER_DISPATCH_OPERATION_V2",
    "ROUTING_PROVIDER_DISPATCH_REQUEST_SCHEMA_V2",
    "ROUTING_PROVIDER_DISPATCH_PURPOSE_V2",
    "ROUTING_BUDGET_RESERVATION_SCHEMA_V3",
    "ROUTING_BUDGET_RESERVATION_RESULT_SCHEMA_V3",
    "ROUTING_BUDGET_RESERVATION_PROOF_SCHEMA_V3",
    "ROUTING_BUDGET_RESERVATION_PURPOSE_V3",
    "ProtectedRoutingProviderTerminalError",
    "build_routing_budget_reservation_v3",
    "validate_routing_budget_reservation_v3",
    "validate_routing_budget_reservation_result_v3",
    "routing_budget_reservation_proof_v3",
    "prepared_routing_provider_call_from_mapping",
    "execute_protected_routing_provider_terminal_v2",
]
