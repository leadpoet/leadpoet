"""Fail-closed runtime wiring for measured routing experiments.

This module intentionally has no HTTP client and never reads a provider
credential. A live reviewed runner uses the typed protected scoring dispatch
operation; the default configuration is replay/fixture only.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import os
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from gateway.research_lab.provider_evidence_proxy import PROXY_URL_ENV
from gateway.research_lab.routing_experiment_attestation import (
    ROUTING_EXPERIMENT_ATTESTATION_OPERATION_V2,
    ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2,
    execute_routing_experiment_attestation_v2,
    validate_routing_experiment_attestation_input_v2,
)
from gateway.research_lab.routing_execution_authorization import (
    ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
    ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
    RoutingProviderCallAuthorizationV2,
    build_routing_provider_authorization_request_v2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
)
from gateway.tee.execution_job_manager_v2 import PARENT_RECEIPT_GRAPHS_FIELD
from gateway.research_lab.routing_experiment_store import (
    SupabaseRoutingDecisionReceiptRepository,
    SupabaseRoutingExperimentStore,
    SupabaseRoutingProviderReceiptRepository,
    RoutingExperimentExecutionClaim,
    RoutingExecutionRequestLease,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.routing_admission import RoutingAdmissionBundleV2
from gateway.research_lab.routing_provider_terminal_protected import (
    ROUTING_BUDGET_RESERVATION_PROOF_SCHEMA_V3,
    ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
    ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
    ROUTING_PROVIDER_DISPATCH_REQUEST_SCHEMA_V2,
    ROUTING_PROVIDER_TERMINAL_OPERATION_V2,
    ROUTING_PROVIDER_TERMINAL_PURPOSE_V2,
    build_routing_budget_reservation_v3,
    validate_routing_budget_reservation_result_v3,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
)
from gateway.research_lab.routing_provider_bindings import (
    PreparedRoutingProviderCall,
    PreparedRoutingProviderWorkflow,
    ReviewedDeeplineActionCompiler,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    ProviderOutcome,
    ProviderReceipt,
    ProviderReceiptStore,
    ReceiptExecutionMode,
    RoutingCallAuthorization,
    RoutingDecisionReceiptStore,
    RoutingExperimentArtifactAuthority,
    RoutingExperimentError,
    RoutingExperimentV2Adapter,
    RoutingExperimentV2Evaluation,
    RoutingExperimentV2Spec,
    evaluate_routing_experiment_v2,
    provider_receipt_key,
)
from leadpoet_canonical.attested_v2 import validate_signed_execution_receipt


ROUTING_EXPERIMENT_OPERATION_V2 = "routing_experiment_v2"
ROUTING_EXPERIMENT_PURPOSE_V2 = "research_lab.routing_experiment.v2"
ROUTING_EXPERIMENT_PROVIDER_PURPOSE_V2 = "research_lab.routing_provider_evidence.v2"


class RoutingExperimentRuntimeError(RuntimeError):
    """The runtime is not allowed to execute the requested routing run."""


def _routing_provider_authorization_parent_hashes(
    *,
    model_binding_observation: VerifiedRoutingModelBindingRequirements,
    protected_release_receipt: Mapping[str, Any],
) -> list[str]:
    """Return the SQL-bound authorization ancestry in semantic order."""

    protected_hash = str(protected_release_receipt.get("receipt_hash") or "")
    observation_hash = str(
        model_binding_observation.observation_receipt_hash or ""
    )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", protected_hash) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", observation_hash
    ):
        raise RoutingExperimentRuntimeError(
            "routing provider authorization ancestry is invalid"
        )
    if protected_hash == observation_hash:
        raise RoutingExperimentRuntimeError(
            "routing provider authorization ancestry contains duplicates"
        )
    return [protected_hash, observation_hash]


def _validate_routing_provider_authorization_parent_graphs(
    *,
    parent_receipt_graphs: Sequence[Mapping[str, Any]],
    expected_hashes: Sequence[str],
    expected_receipts: Sequence[Mapping[str, Any]],
) -> None:
    """Require both exact parent receipts before submitting the TEE job."""

    if (
        not isinstance(parent_receipt_graphs, Sequence)
        or isinstance(parent_receipt_graphs, (str, bytes, bytearray))
        or not parent_receipt_graphs
        or any(not isinstance(item, Mapping) for item in parent_receipt_graphs)
    ):
        raise RoutingExperimentRuntimeError(
            "routing provider authorization ancestry is unavailable"
        )
    expected_by_hash = {
        expected_hash: dict(receipt)
        for expected_hash, receipt in zip(expected_hashes, expected_receipts)
    }
    observed_hashes: list[str] = []
    for graph in parent_receipt_graphs:
        receipts = graph.get("receipts")
        if (
            not isinstance(receipts, Sequence)
            or isinstance(receipts, (str, bytes, bytearray))
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization ancestry is unavailable"
            )
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise RoutingExperimentRuntimeError(
                    "routing provider authorization ancestry is invalid"
                )
            receipt_hash = str(receipt.get("receipt_hash") or "")
            if receipt_hash in expected_by_hash:
                if dict(receipt) != expected_by_hash[receipt_hash]:
                    raise RoutingExperimentRuntimeError(
                        "routing provider authorization ancestry differs"
                    )
                observed_hashes.append(receipt_hash)
    if observed_hashes != list(expected_hashes):
        raise RoutingExperimentRuntimeError(
            "routing provider authorization ancestry differs"
        )


class RoutingExperimentTerminalRecoveryError(RoutingExperimentRuntimeError):
    """A durable recovery fact requires a new immutable experiment."""


class RoutingExperimentDeferredRecoveryError(RoutingExperimentRuntimeError):
    """Durable cleanup could not be confirmed; the queue lease must expire."""


class RoutingProviderDispatchExecutor:
    """Private marker for the fixed scoring-TEE dispatch executor.

    The reviewed runner must never receive a provider broker or an arbitrary
    callable.  The concrete operation executor is assembled by the reviewed
    product composition and carries a private token that cannot be supplied
    by a generic ``.execute`` object.
    """

    __slots__ = ()


_ROUTING_DISPATCH_EXECUTOR_TOKEN = object()


def _require_reviewed_direct_prepared_call(
    prepared: Any,
) -> PreparedRoutingProviderCall:
    """Keep the released V3 persistence/dispatch boundary single-action.

    ``PreparedRoutingProviderWorkflow`` is intentionally recognized here,
    rather than falling through to an attribute error.  A composite route
    needs a separately released protected aggregate receipt and append-only
    reconciliation contract; until then it must stop before reservation or
    dispatch.  Direct ``PreparedRoutingProviderCall`` instances retain the
    existing path unchanged.
    """

    if isinstance(prepared, PreparedRoutingProviderWorkflow):
        raise RoutingExperimentRuntimeError(
            "routing composite workflow dispatch is unavailable: "
            "protected aggregate receipt schema is not released"
        )
    if not isinstance(prepared, PreparedRoutingProviderCall):
        raise RoutingExperimentRuntimeError(
            "routing provider prepared call is not a reviewed direct action"
        )
    return prepared


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RoutingExperimentRuntimeError(f"{name} must be boolean")


@dataclass(frozen=True)
class RoutingExperimentRuntimeConfig:
    """Explicit off-by-default gate for the independent routing worker."""

    enabled: bool = False
    live_execution_enabled: bool = False
    worker_lease_seconds: int = 300
    evidence_proxy_url: str = ""
    attested_authority_mode: str = ""

    @classmethod
    def from_env(cls) -> "RoutingExperimentRuntimeConfig":
        lease = int(os.getenv("RESEARCH_LAB_ROUTING_EXPERIMENT_LEASE_SECONDS", "300"))
        if lease < 30 or lease > 3600:
            raise RoutingExperimentRuntimeError(
                "RESEARCH_LAB_ROUTING_EXPERIMENT_LEASE_SECONDS must be 30..3600"
            )
        return cls(
            enabled=_env_bool("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED"),
            live_execution_enabled=_env_bool(
                "RESEARCH_LAB_ROUTING_EXPERIMENT_LIVE_ENABLED"
            ),
            worker_lease_seconds=lease,
            evidence_proxy_url=str(os.getenv(PROXY_URL_ENV, "") or "").strip(),
            attested_authority_mode=str(
                os.getenv("RESEARCH_LAB_ROUTING_EXPERIMENT_AUTHORITY", "") or ""
            ).strip(),
        )

    def assert_live_enabled(self) -> None:
        if not self.enabled:
            raise RoutingExperimentRuntimeError("routing experiment worker is disabled")
        if not self.live_execution_enabled:
            raise RoutingExperimentRuntimeError("routing experiment live execution is disabled")
        if not self.evidence_proxy_url:
            raise RoutingExperimentRuntimeError("routing experiment evidence proxy is required")
        if self.attested_authority_mode != "attested":
            raise RoutingExperimentRuntimeError(
                "routing experiment attested authority is not configured"
            )


class AttestedRoutingExperimentExecutionAuthority(Protocol):
    """The protected workflow assertion required before a provider call."""

    def authorize(
        self,
        *,
        operation: str,
        purpose: str,
        authorization: RoutingCallAuthorization,
        binding: ProviderBindingIdentity,
    ) -> Mapping[str, Any]: ...


class FailClosedRoutingExperimentExecutionAuthority:
    """Default until a protected TEE operation is bound end-to-end."""

    def authorize(
        self,
        *,
        operation: str,
        purpose: str,
        authorization: RoutingCallAuthorization,
        binding: ProviderBindingIdentity,
    ) -> Mapping[str, Any]:
        del operation, purpose, authorization, binding
        raise RoutingExperimentRuntimeError(
            "routing experiment attested execution authority is unavailable"
        )


class AttestedScoringV2RoutingProviderCallAuthority:
    """Authorize one exact provider call through the distinct protected operation.

    The returned proof carries the complete grant, deterministic result, and
    signed execution receipt.  The coordinator broker verifies that proof
    statelessly, so a restart does not depend on this authority instance.
    """

    def __init__(
        self,
        *,
        executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    ) -> None:
        self._executor = executor

    def validate_composition(self) -> None:
        """Reject a host callback in place of the reviewed TEE job executor."""

        if (
            not isinstance(self._executor, RoutingProviderDispatchExecutor)
            or getattr(self._executor, "_routing_dispatch_executor_token", None)
            is not _ROUTING_DISPATCH_EXECUTOR_TOKEN
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization executor is not product-composed"
            )

    def authorize(
        self,
        authorization: RoutingProviderCallAuthorizationV2,
        *,
        artifact_lineage: VerifiedRoutingArtifactLineage,
        model_binding_observation: VerifiedRoutingModelBindingRequirements,
        execution_envelope: RoutingExperimentExecutionEnvelopeV2,
        admission_bundle: RoutingAdmissionBundleV2,
        prepared_call: Any,
        protected_release_receipt: Mapping[str, Any],
        parent_receipt_graphs: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if self._executor is None:
            raise RoutingExperimentRuntimeError(
                "routing provider authorization operation is not released"
            )
        authorization_payload = authorization.to_dict()
        core_payload = build_routing_provider_authorization_request_v2(
            authorization=authorization,
            artifact_lineage=artifact_lineage,
            model_binding_observation=model_binding_observation,
            execution_envelope=execution_envelope,
            admission_bundle=admission_bundle,
            prepared_call=prepared_call,
            protected_release_receipt=protected_release_receipt,
        )
        expected_parent_hashes = _routing_provider_authorization_parent_hashes(
            model_binding_observation=model_binding_observation,
            protected_release_receipt=protected_release_receipt,
        )
        _validate_routing_provider_authorization_parent_graphs(
            parent_receipt_graphs=parent_receipt_graphs,
            expected_hashes=expected_parent_hashes,
            expected_receipts=(
                protected_release_receipt,
                model_binding_observation.signed_receipt,
            ),
        )
        payload = {
            **core_payload,
            "parent_receipt_hashes": expected_parent_hashes,
            PARENT_RECEIPT_GRAPHS_FIELD: [
                dict(item) for item in parent_receipt_graphs
            ],
        }
        expected_input_root = sha256_json(payload)
        response = self._executor(
            {
                "operation": ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
                "purpose": ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
                "payload": payload,
                "parent_receipt_hashes": expected_parent_hashes,
            }
        )
        result = response.get("result") if isinstance(response, Mapping) else None
        receipt = (
            response.get("execution_receipt") or response.get("receipt")
            if isinstance(response, Mapping)
            else None
        )
        if (
            not isinstance(response, Mapping)
            or response.get("status") != "succeeded"
            or response.get("operation") != ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2
            or response.get("purpose") != ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2
            or not isinstance(result, Mapping)
            or not isinstance(receipt, Mapping)
            or receipt.get("role") != "gateway_scoring"
            or receipt.get("purpose") != ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2
            or receipt.get("status") != "succeeded"
            or receipt.get("input_root") != expected_input_root
            or receipt.get("parent_receipt_hashes") != expected_parent_hashes
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(receipt.get("receipt_hash") or "")
            )
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization TEE receipt is invalid"
            )
        authorization_job_id = str(receipt.get("job_id") or "")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", authorization_job_id
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization execution job identity is invalid"
            )
        try:
            expected = execute_routing_provider_call_authorization_v2(
                authorization_payload,
                authorization_job_id=authorization_job_id,
            )
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider authorization result is invalid"
            ) from exc
        if dict(result) != expected:
            raise RoutingExperimentRuntimeError(
                "routing provider authorization result is not exact"
            )
        if (
            receipt.get("job_id") != expected["authorization_job_id"]
            or receipt.get("output_root") != expected["output_root"]
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization receipt job identity differs"
            )
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider authorization TEE signature is invalid"
            ) from exc
        proof_hash = str(receipt["receipt_hash"])
        proof = {
            "authorization_hash": authorization.authorization_hash(),
            "authorization_request_hash": expected_input_root,
            "authorization_proof_hash": proof_hash,
            "request_body_hash": authorization.request_body_hash,
            "action_id": authorization.action_id,
            "credit_cap_microunits": authorization.credit_cap_microunits,
            "timeout_ms": authorization.timeout_ms,
            # The broker receives the exact signed result and its input
            # document as well as the compact routing fields.  The latter
            # are needed for URL/body checks; the former prevent a process
            # local registry from becoming the only source of authority.
            "authorization": authorization_payload,
            "authorization_result": expected,
            "authorization_receipt": dict(receipt),
        }
        return proof

    def validate_broker_request(
        self,
        proof: Mapping[str, Any],
        broker_request: Mapping[str, Any],
    ) -> None:
        try:
            from gateway.tee.provider_broker_v2 import (
                validate_routing_authorization_proof_v2,
            )

            validate_routing_authorization_proof_v2(proof, broker_request)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider broker proof is invalid"
            ) from exc


class AttestedScoringV2RoutingProviderTerminalAuthority:
    """Submit one broker terminal to the protected scorer and verify its receipt.

    The broker result is the only accepted source for the signed provider
    record and response body.  This class does not project billing, create a
    terminal signature, or trust the legacy ``routing_terminal`` field.  The
    protected scorer performs the provider-record/compiler validation; this
    wrapper verifies the ordinary ``ExecutionJobManagerV2`` receipt that
    encloses that result.
    """

    def __init__(
        self,
        *,
        executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
        protected_release_receipt: Mapping[str, Any] | None,
    ) -> None:
        self._executor = executor
        self._protected_release_receipt = (
            dict(protected_release_receipt)
            if isinstance(protected_release_receipt, Mapping)
            else None
        )

    def execute(
        self,
        *,
        authorization_proof: Mapping[str, Any],
        prepared_call: Any,
        broker_request: Mapping[str, Any],
        broker_result: Mapping[str, Any],
        parent_receipt_graphs: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Run the protected operation and return only its result and receipt."""

        if self._executor is None:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authority is unavailable"
            )
        release = self._protected_release_receipt
        if release is None:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal protected release is unavailable"
            )
        try:
            validate_signed_execution_receipt(release)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal protected release is invalid"
            ) from exc
        if (
            release.get("role") != "gateway_scoring"
            or release.get("purpose") != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
            or release.get("status") != "succeeded"
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal protected release scope differs"
            )
        if not isinstance(authorization_proof, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization proof is unavailable"
            )
        proof_hash = str(authorization_proof.get("authorization_proof_hash") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", proof_hash):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization proof hash is invalid"
            )
        authorization_receipt = authorization_proof.get("authorization_receipt")
        if not isinstance(authorization_receipt, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization receipt is unavailable"
            )
        try:
            validate_signed_execution_receipt(authorization_receipt)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization receipt is invalid"
            ) from exc
        if (
            authorization_receipt.get("role") != "gateway_scoring"
            or authorization_receipt.get("purpose")
            != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
            or authorization_receipt.get("status") != "succeeded"
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization receipt scope differs"
            )
        if (
            not isinstance(broker_request, Mapping)
            or not isinstance(broker_result, Mapping)
            or broker_result.get("terminal_status") != "authenticated_response"
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal broker result is not authenticated"
            )
        routing_provider_record = broker_result.get("routing_provider_record")
        provider_record = broker_result.get("provider_record")
        if (
            routing_provider_record is not None
            and provider_record is not None
            and routing_provider_record != provider_record
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal provider record substitution detected"
            )
        if routing_provider_record is not None:
            provider_record = routing_provider_record
        if not isinstance(provider_record, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal signed record is unavailable from broker result"
            )
        body_b64 = broker_result.get("body_b64")
        try:
            raw_body = base64.b64decode(str(body_b64 or ""), validate=True)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal response body is unavailable from broker result"
            ) from exc
        if not raw_body or len(raw_body) > 8 * 1024 * 1024:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal response body is invalid"
            )
        if (
            not isinstance(parent_receipt_graphs, Sequence)
            or isinstance(parent_receipt_graphs, (str, bytes, bytearray))
            or not parent_receipt_graphs
            or any(not isinstance(item, Mapping) for item in parent_receipt_graphs)
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization ancestry is unavailable"
            )
        authorization_receipt_hash = str(
            authorization_receipt.get("receipt_hash") or ""
        )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", authorization_receipt_hash):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization receipt hash is invalid"
            )
        graph_receipts = {
            str(receipt.get("receipt_hash") or "")
            for graph in parent_receipt_graphs
            for receipt in (graph.get("receipts") or ())
            if isinstance(receipt, Mapping)
        }
        if authorization_receipt_hash not in graph_receipts:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal authorization ancestry differs"
            )
        try:
            prepared_projection = asdict(prepared_call)
            prepared_projection["binding"] = prepared_call.binding.to_dict()
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal prepared call is invalid"
            ) from exc
        payload: dict[str, Any] = {
            "schema_version": "leadpoet.routing_provider_terminal_request.v2",
            "authorization_proof": dict(authorization_proof),
            "prepared_call": prepared_projection,
            "broker_request": dict(broker_request),
            "broker_result": dict(broker_result),
            "provider_record": dict(provider_record),
            "raw_response_body_b64": base64.b64encode(raw_body).decode("ascii"),
            PARENT_RECEIPT_GRAPHS_FIELD: [dict(item) for item in parent_receipt_graphs],
        }
        # The terminal job is a new manager job.  Its identifier cannot reuse
        # the broker authorization job because the latter is already bound by
        # the signed request; deriving a second identifier avoids a payload /
        # manifest hash cycle while retaining the authorization receipt parent.
        terminal_job_id = (
            "routing-terminal:"
            + sha256_json(
                {
                    "schema_version": "leadpoet.routing_provider_terminal_job.v2",
                    "authorization_proof_hash": proof_hash,
                    "authorization_receipt_hash": authorization_receipt_hash,
                }
            ).split(":", 1)[1][:32]
        )
        response = self._executor(
            {
                "operation": ROUTING_PROVIDER_TERMINAL_OPERATION_V2,
                "purpose": ROUTING_PROVIDER_TERMINAL_PURPOSE_V2,
                "payload": payload,
                "job_id": terminal_job_id,
                "parent_receipt_hashes": [authorization_receipt_hash],
            }
        )
        if not isinstance(response, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal TEE response is invalid"
            )
        result = response.get("result")
        receipt = response.get("execution_receipt")
        alternate_receipt = response.get("receipt")
        if (
            receipt is not None
            and alternate_receipt is not None
            and receipt != alternate_receipt
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal TEE receipt substitution detected"
            )
        if receipt is None:
            receipt = alternate_receipt
        expected_job_id = terminal_job_id
        if (
            response.get("status") != "succeeded"
            or response.get("operation") != ROUTING_PROVIDER_TERMINAL_OPERATION_V2
            or response.get("purpose") != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
            or not isinstance(result, Mapping)
            or not isinstance(receipt, Mapping)
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal TEE response scope is invalid"
            )
        if (
            receipt.get("role") != "gateway_scoring"
            or receipt.get("purpose") != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2
            or receipt.get("status") != "succeeded"
            or receipt.get("failure_code") not in (None, "")
            or receipt.get("job_id") != expected_job_id
            or receipt.get("input_root") != sha256_json(payload)
            or receipt.get("output_root") != sha256_json(dict(result))
            or receipt.get("parent_receipt_hashes") != [authorization_receipt_hash]
            or receipt.get("enclave_pubkey") != release.get("enclave_pubkey")
            or any(
                receipt.get(name) != release.get(name)
                for name in (
                    "commit_sha",
                    "pcr0",
                    "build_manifest_hash",
                    "dependency_lock_hash",
                    "config_hash",
                    "boot_identity_hash",
                )
            )
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider terminal standard receipt differs"
            )
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider terminal standard receipt signature is invalid"
            ) from exc
        return {"result": dict(result), "execution_receipt": dict(receipt)}


class AttestedScoringV2RoutingProviderDispatchAuthority:
    """Invoke the released scoring-side routing dispatch operation.

    The host sends the exact compiler projection and signed authorization
    proof to a typed TEE job RPC.  It never sends, receives, or constructs a
    provider broker result.  The scoring enclave obtains that result through
    its existing attested coordinator path and returns only the bounded
    terminal projection plus the standard signed job receipt.
    """

    def __init__(
        self,
        *,
        executor: RoutingProviderDispatchExecutor | None,
        protected_release_receipt: Mapping[str, Any] | None,
    ) -> None:
        if executor is not None:
            if not isinstance(executor, RoutingProviderDispatchExecutor):
                raise TypeError(
                    "routing provider dispatch executor must be the reviewed fixed-operation executor"
                )
            if getattr(executor, "_routing_dispatch_executor_token", None) is not _ROUTING_DISPATCH_EXECUTOR_TOKEN:
                raise TypeError(
                    "routing provider dispatch executor is not product-composed"
                )
        self._executor = executor
        self._protected_release_receipt = (
            dict(protected_release_receipt)
            if isinstance(protected_release_receipt, Mapping)
            else None
        )

    def validate_composition(self) -> None:
        """Require the fixed-operation executor and signed release ancestor."""

        if (
            not isinstance(self._executor, RoutingProviderDispatchExecutor)
            or getattr(self._executor, "_routing_dispatch_executor_token", None)
            is not _ROUTING_DISPATCH_EXECUTOR_TOKEN
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch executor is not product-composed"
            )
        release = self._protected_release_receipt
        if not isinstance(release, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch protected release is unavailable"
            )
        try:
            validate_signed_execution_receipt(release)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch protected release is invalid"
            ) from exc
        if (
            release.get("role") != "gateway_scoring"
            or release.get("purpose") != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
            or release.get("status") != "succeeded"
            or release.get("failure_code") not in (None, "")
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch protected release scope differs"
            )

    def execute(
        self,
        *,
        authorization_proof: Mapping[str, Any],
        prepared_call: Any,
        broker_request: Mapping[str, Any],
        budget_reservation: Mapping[str, Any],
        parent_receipt_graphs: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if self._executor is None:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authority is unavailable"
            )
        release = self._protected_release_receipt
        if release is None:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch protected release is unavailable"
            )
        try:
            validate_signed_execution_receipt(release)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch protected release is invalid"
            ) from exc
        if (
            release.get("role") != "gateway_scoring"
            or release.get("purpose") != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
            or release.get("status") != "succeeded"
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch protected release scope differs"
            )
        if not isinstance(authorization_proof, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authorization proof is unavailable"
            )
        authorization_receipt = authorization_proof.get("authorization_receipt")
        if not isinstance(authorization_receipt, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authorization receipt is unavailable"
            )
        try:
            validate_signed_execution_receipt(authorization_receipt)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authorization receipt is invalid"
            ) from exc
        authorization_receipt_hash = str(
            authorization_receipt.get("receipt_hash") or ""
        )
        if (
            authorization_receipt.get("role") != "gateway_scoring"
            or authorization_receipt.get("purpose")
            != ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2
            or authorization_receipt.get("status") != "succeeded"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", authorization_receipt_hash)
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authorization receipt scope differs"
            )
        if (
            not isinstance(parent_receipt_graphs, Sequence)
            or isinstance(parent_receipt_graphs, (str, bytes, bytearray))
            or not parent_receipt_graphs
            or any(not isinstance(item, Mapping) for item in parent_receipt_graphs)
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authorization ancestry is unavailable"
            )
        if not any(
            receipt == dict(authorization_receipt)
            for graph in parent_receipt_graphs
            for receipt in graph.get("receipts") or ()
            if isinstance(receipt, Mapping)
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authorization ancestry differs"
            )
        try:
            prepared_projection = asdict(prepared_call)
            prepared_projection["binding"] = prepared_call.binding.to_dict()
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch prepared call is invalid"
            ) from exc
        if not isinstance(broker_request, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch broker request is invalid"
            )
        if not isinstance(budget_reservation, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch budget reservation is invalid"
            )
        payload = {
            "schema_version": ROUTING_PROVIDER_DISPATCH_REQUEST_SCHEMA_V2,
            "authorization_proof": dict(authorization_proof),
            "prepared_call": prepared_projection,
            "broker_request": dict(broker_request),
            "budget_reservation": dict(budget_reservation),
            PARENT_RECEIPT_GRAPHS_FIELD: [
                dict(item) for item in parent_receipt_graphs
            ],
        }
        dispatch_job_id = routing_provider_dispatch_job_id_v2(
            authorization_proof
        )
        if broker_request.get("job_id") != dispatch_job_id:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch broker job identity differs"
            )
        response = self._executor(
            {
                "operation": ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
                "purpose": ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
                "payload": payload,
                "job_id": dispatch_job_id,
                "parent_receipt_hashes": [authorization_receipt_hash],
            }
        )
        if not isinstance(response, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch TEE response is invalid"
            )
        result = response.get("result")
        receipt = response.get("execution_receipt") or response.get("receipt")
        if (
            response.get("status") != "succeeded"
            or response.get("operation") != ROUTING_PROVIDER_DISPATCH_OPERATION_V2
            or response.get("purpose") != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
            or not isinstance(result, Mapping)
            or not isinstance(receipt, Mapping)
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch TEE response scope is invalid"
            )
        if (
            receipt.get("role") != "gateway_scoring"
            or receipt.get("purpose") != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
            or receipt.get("status") != "succeeded"
            or receipt.get("job_id") != dispatch_job_id
            or receipt.get("input_root") != sha256_json(payload)
            or receipt.get("output_root") != sha256_json(dict(result))
            or receipt.get("parent_receipt_hashes") != [authorization_receipt_hash]
            or receipt.get("enclave_pubkey") != release.get("enclave_pubkey")
            or any(
                receipt.get(name) != release.get(name)
                for name in (
                    "commit_sha",
                    "pcr0",
                    "build_manifest_hash",
                    "dependency_lock_hash",
                    "config_hash",
                    "boot_identity_hash",
                )
            )
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch standard receipt differs"
            )
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch standard receipt signature is invalid"
            ) from exc
        _validate_redacted_routing_dispatch_result(result)
        return {"result": dict(result), "execution_receipt": dict(receipt)}


_ROUTING_DISPATCH_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "operation",
        "terminal_status",
        "authorization_hash",
        "authorization_proof_hash",
        "binding",
        "unit_ref",
        "request_fingerprint",
        "provider_record_hash",
        "provider_body_hash",
        "transport_attempt_hash",
        "projection",
        "provider_receipt",
        "budget_reservation",
    }
)
_ROUTING_DISPATCH_PROJECTION_FIELDS = frozenset(
    {
        "outcome",
        "evidence_hash",
        "credit_microunits",
        "latency_ms",
        "billing_state",
        "binding_id",
        "provider_id",
        "tool_id",
        "request_fingerprint",
    }
)
_ROUTING_DISPATCH_RAW_FIELDS = frozenset(
    {
        "body_b64",
        "provider_record",
        "routing_provider_record",
        "transport_attempt",
        "raw_response_body_b64",
        "response_body",
        "response_body_b64",
        "provider_output",
    }
)


def _validate_redacted_routing_dispatch_result(value: Any) -> None:
    """Reject raw provider data before it can cross the host boundary."""

    if not isinstance(value, Mapping) or set(value) != _ROUTING_DISPATCH_RESULT_FIELDS:
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch result is not the redacted schema"
        )
    if any(field in value for field in _ROUTING_DISPATCH_RAW_FIELDS):
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch result contains raw provider data"
        )
    if value.get("operation") != ROUTING_PROVIDER_TERMINAL_OPERATION_V2:
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch result operation differs"
        )
    projection = value.get("projection")
    if not isinstance(projection, Mapping) or set(projection) != _ROUTING_DISPATCH_PROJECTION_FIELDS:
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch projection is not redacted"
        )
    if any(field in projection for field in _ROUTING_DISPATCH_RAW_FIELDS):
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch projection contains raw provider data"
        )
    provider_receipt = value.get("provider_receipt")
    if not isinstance(provider_receipt, Mapping):
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch provider receipt is invalid"
        )
    try:
        receipt = ProviderReceipt.from_mapping(provider_receipt)
    except Exception as exc:
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch provider receipt is invalid"
        ) from exc
    if validate_provider_receipt(receipt):
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch provider receipt is invalid"
        )
    budget_reservation = value.get("budget_reservation")
    expected_budget_fields = {
        "schema_version",
        "reservation_id",
        "event_key",
        "experiment_hash",
        "binding_id",
        "claim_key",
        "claim_generation",
        "credit_microunits",
        "lease_expires_at",
        "response_hash",
        "transport_attempt_hash",
    }
    if (
        not isinstance(budget_reservation, Mapping)
        or set(budget_reservation) != expected_budget_fields
        or budget_reservation.get("schema_version")
        != ROUTING_BUDGET_RESERVATION_PROOF_SCHEMA_V3
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(budget_reservation.get("event_key") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(budget_reservation.get("experiment_hash") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(budget_reservation.get("claim_key") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(budget_reservation.get("response_hash") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(budget_reservation.get("transport_attempt_hash") or ""),
        )
    ):
        raise RoutingExperimentRuntimeError(
            "routing provider dispatch budget reservation proof is invalid"
        )


class AttestedScoringV2RoutingEvaluationAuthority:
    """Host bridge for the separate protected routing-evaluation operation.

    The deployed TEE must add the exact operation/purpose to its protected
    manifest and durable purpose allowlist before this bridge can succeed.
    Until then the bridge fails closed; it cannot substitute an existing
    scoring purpose or a host-built promotion receipt.
    """

    def __init__(
        self,
        *,
        executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._executor = executor

    def attest(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        validate_routing_experiment_attestation_input_v2(payload)
        if self._executor is None:
            raise RoutingExperimentRuntimeError(
                "routing evaluation TEE operation is not released"
            )
        response = self._executor(
            {
                "operation": ROUTING_EXPERIMENT_ATTESTATION_OPERATION_V2,
                "purpose": ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2,
                "payload": dict(payload),
            }
        )
        if not isinstance(response, Mapping) or response.get("status") != "succeeded":
            raise RoutingExperimentRuntimeError("routing evaluation TEE operation failed")
        result = response.get("result")
        receipt = response.get("execution_receipt") or response.get("receipt")
        if not isinstance(result, Mapping) or not isinstance(receipt, Mapping):
            raise RoutingExperimentRuntimeError("routing evaluation TEE response is malformed")
        expected = execute_routing_experiment_attestation_v2(payload)
        if dict(result) != expected:
            raise RoutingExperimentRuntimeError("routing evaluation TEE result is not exact")
        if (
            response.get("operation") != ROUTING_EXPERIMENT_ATTESTATION_OPERATION_V2
            or response.get("purpose") != ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2
            or receipt.get("role") != "gateway_scoring"
            or receipt.get("purpose") != ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2
            or receipt.get("status") != "succeeded"
            or receipt.get("input_root") != expected["input_root"]
            or receipt.get("output_root") != expected["output_root"]
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(receipt.get("receipt_hash") or "")
            )
        ):
            raise RoutingExperimentRuntimeError("routing evaluation TEE receipt is invalid")
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing evaluation TEE signature is invalid"
            ) from exc
        return {
            "result": dict(expected),
            # Preserve the complete signed public receipt. The store and SQL
            # authority bind its commit, PCR0, build manifest, boot identity,
            # input root, and output root. A hash-only projection is not an
            # authoritative promotion receipt.
            "receipt": dict(receipt),
        }


class KmsRoutingExperimentArtifactAuthority:
    """Adapter over the existing private artifact KMS verifier."""

    def __init__(self, verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None) -> None:
        self._verifier = verifier

    def verify(self, *, artifact: Any, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            if self._verifier is None:
                from research_lab.eval import verify_private_artifact_manifest_signature

                result = verify_private_artifact_manifest_signature(manifest)
            else:
                result = self._verifier(manifest)
        except Exception as exc:  # noqa: BLE001 - do not expose verifier internals
            raise RoutingExperimentRuntimeError(
                "routing experiment artifact signature verification failed"
            ) from exc
        if not isinstance(result, Mapping) or result.get("verified") is not True:
            raise RoutingExperimentRuntimeError(
                "routing experiment artifact signature verification was rejected"
            )
        signature_ref = str(manifest.get("signature_ref") or "")
        key_id = str(result.get("key_id") or "")
        if (
            result.get("manifest_hash") != artifact.manifest_hash
            or not signature_ref.startswith("s3://")
            or result.get("signature_ref") != signature_ref
            or not key_id
            or result.get("signing_algorithm") != "ECDSA_SHA_256"
            or result.get("consumer_contract_binding_mode") != "semantic_v1_required"
        ):
            raise RoutingExperimentRuntimeError(
                "routing experiment artifact signature binding is incomplete"
            )
        return {
            "verified": True,
            "model_artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "commit_sha": artifact.commit_sha,
            "signature_ref": signature_ref,
            "key_id": key_id,
            "signing_algorithm": "ECDSA_SHA_256",
            "consumer_contract_binding_mode": "semantic_v1_required",
        }


class ProviderBrokerRoutingExecutor(Protocol):
    """Redacted bridge implemented inside the protected broker workflow."""

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _RoutingExecutionBinding:
    experiment_hash: str
    experiment_id: str
    claim: RoutingExperimentExecutionClaim
    deadline_monotonic: float
    deadline_supplier: Callable[[], float] | None = None


@dataclass(frozen=True)
class _PendingRoutingProviderAttempt:
    reservation_id: str
    claim: RoutingExperimentExecutionClaim
    billing_state: str
    authoritative_billed_credit_microunits: int | None


class ProviderBrokerV2RoutingExecutor:
    """Use an injected ``ProviderBrokerV2`` request and redaction projection.

    The request factory runs only in the attested coordinator integration. It
    supplies the broker's measured URL/body/credential lease contract.  The
    result projector returns only outcome, evidence hash, measured cost, and
    latency; raw broker response data never crosses into this Lab runtime.
    """

    def __init__(
        self,
        *,
        broker: Any,
        broker_request_factory: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        result_projector: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        store: Any | None = None,
    ) -> None:
        if isinstance(store, SupabaseRoutingExperimentStore):
            raise RoutingExperimentRuntimeError(
                "legacy provider broker executor is incompatible with V3 durable store"
            )
        self._broker = broker
        self._broker_request_factory = broker_request_factory
        self._result_projector = result_projector
        self._v3_durable_store = isinstance(store, SupabaseRoutingExperimentStore)

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._v3_durable_store:
            raise RoutingExperimentRuntimeError(
                "legacy provider broker executor is incompatible with V3 durable store"
            )
        broker_request = self._broker_request_factory(dict(request))
        if not isinstance(broker_request, Mapping):
            raise RoutingExperimentRuntimeError("provider broker request is malformed")
        response = self._broker.execute(dict(broker_request))
        projected = self._result_projector(dict(request), response)
        if not isinstance(projected, Mapping):
            raise RoutingExperimentRuntimeError("provider broker result projection is malformed")
        return dict(projected)


def _reservation_id(authorization: RoutingCallAuthorization, binding: ProviderBindingIdentity) -> str:
    return "routing-reservation:" + sha256_json(
        {
            "experiment_id": authorization.experiment_id,
            "variant_id": authorization.variant_id,
            "unit_ref": authorization.unit_ref,
            "tool_id": authorization.tool_id,
            "attempt": authorization.attempt,
            "request_fingerprint": authorization.request_fingerprint,
            "binding_id": binding.binding_id,
        }
    ).split(":", 1)[1][:32]


def _budget_event_key(kind: str, reservation_id: str, *parts: str) -> str:
    return sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
            "kind": kind,
            "reservation_id": reservation_id,
            "parts": list(parts),
        }
    )


class ProviderBrokerRoutingRunner:
    """Authorization-first measured runner with conservative settlement."""

    def __init__(
        self,
        *,
        config: RoutingExperimentRuntimeConfig,
        store: SupabaseRoutingExperimentStore,
        execution_authority: AttestedRoutingExperimentExecutionAuthority,
        broker_executor: ProviderBrokerRoutingExecutor,
        execution: _RoutingExecutionBinding | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.execution_authority = execution_authority
        self.broker_executor = broker_executor
        self._execution = execution
        self._pending_attempts: dict[str, _PendingRoutingProviderAttempt] = {}

    def __call__(
        self,
        binding: ProviderBindingIdentity,
        unit_ref: str,
        request_fingerprint: str,
        authorization: RoutingCallAuthorization,
    ) -> Mapping[str, Any]:
        if isinstance(self.store, SupabaseRoutingExperimentStore):
            # This runner still emits the V2 ``terminal_proof`` document and
            # cannot satisfy migration 157's protected V3 append contract.
            # Reject before authorization, reservation, dispatch, or any
            # provider call.  In-memory fixture/replay stores remain supported.
            raise RoutingExperimentRuntimeError(
                "legacy provider broker runner is incompatible with V3 durable store"
            )
        self.config.assert_live_enabled()
        if authorization.execution_mode != ReceiptExecutionMode.MEASURED_LAB.value:
            raise RoutingExperimentRuntimeError("routing broker requires measured_lab authorization")
        execution = self._execution
        if execution is None:
            raise RoutingExperimentRuntimeError("routing runner is not bound to an experiment")
        experiment_hash = execution.experiment_hash
        claim = execution.claim
        if (
            authorization.experiment_id != execution.experiment_id
            or authorization.tool_id != binding.tool_id
            or authorization.unit_ref != unit_ref
            or authorization.request_fingerprint != request_fingerprint
            or authorization.remaining_credit_microunits < 0
            or authorization.timeout_ceiling_ms < 1
        ):
            raise RoutingExperimentRuntimeError("routing call authorization does not bind the request")
        deadline = (
            execution.deadline_supplier()
            if execution.deadline_supplier is not None
            else execution.deadline_monotonic
        )
        remaining_lease_ms = int((deadline - time.monotonic()) * 1000)
        # This is the bounded-lease proof.  We do not start a provider call
        # unless its full authorization timeout fits before the claim fence.
        # Thus all receipt, decision, and terminal-event writes retain a
        # fixed safety margin without a mutable shared claim capability.
        if remaining_lease_ms <= 0:
            raise RoutingExperimentRuntimeError("routing execution claim lease is exhausted")
        if authorization.timeout_ceiling_ms > remaining_lease_ms:
            raise RoutingExperimentRuntimeError(
                "routing call timeout exceeds remaining claim lease"
            )
        attestation = self.execution_authority.authorize(
            operation=ROUTING_EXPERIMENT_OPERATION_V2,
            purpose=ROUTING_EXPERIMENT_PROVIDER_PURPOSE_V2,
            authorization=authorization,
            binding=binding,
        )
        if (
            not isinstance(attestation, Mapping)
            or attestation.get("attested") is not True
            or attestation.get("operation") != ROUTING_EXPERIMENT_OPERATION_V2
            or attestation.get("purpose") != ROUTING_EXPERIMENT_PROVIDER_PURPOSE_V2
            or attestation.get("binding_id") != binding.binding_id
        ):
            raise RoutingExperimentRuntimeError("routing execution attestation is invalid")
        reservation_id = _reservation_id(authorization, binding)
        reserve_doc = {
            "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
            "operation": ROUTING_EXPERIMENT_OPERATION_V2,
            "binding_id": binding.binding_id,
            "request_fingerprint": request_fingerprint,
        }
        reserve_event_key = _budget_event_key("reserve", reservation_id, request_fingerprint)
        try:
            reserve_result = self.store.reserve_budget(
                event_key=reserve_event_key,
                reservation_id=reservation_id,
                experiment_hash=experiment_hash,
                binding_id=binding.binding_id,
                claim=claim,
                credit_microunits=authorization.remaining_credit_microunits,
                lease_seconds=self.config.worker_lease_seconds,
                event_doc=reserve_doc,
            )
            try:
                validate_routing_budget_reservation_result_v3(
                    reserve_result,
                    reservation={
                        "reservation_id": reservation_id,
                        "event_key": reserve_event_key,
                        "experiment_hash": experiment_hash,
                        "binding_id": binding.binding_id,
                        "claim_key": claim.claim_key,
                        "claim_generation": claim.claim_generation,
                        "credit_microunits": authorization.remaining_credit_microunits,
                    },
                )
            except Exception as exc:
                raise RoutingExperimentRuntimeError(
                    "routing provider durable budget reservation differs"
                ) from exc
        except Exception as exc:
            # An expired or closed deterministic reservation may represent an
            # interrupted provider call.  Conservatively preserve it as
            # uncertain before core emits a new retry fingerprint. A failed
            # recovery never opens a provider path.
            try:
                result = self.store.mark_budget_uncertain(
                    event_key=_budget_event_key(
                        "uncertain", reservation_id, request_fingerprint
                    ),
                    reservation_id=reservation_id,
                    claim=claim,
                    event_doc={
                        "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
                        "request_fingerprint": request_fingerprint,
                        "billing_state": "uncertain",
                    },
                )
                if (
                    not isinstance(result, Mapping)
                    or result.get("uncertain") is not True
                    or result.get("credit_microunits")
                    not in (None, authorization.remaining_credit_microunits)
                ):
                    raise RoutingExperimentRuntimeError(
                        "routing provider budget uncertainty result is not confirmed"
                    )
            except Exception:
                raise RoutingExperimentDeferredRecoveryError(
                    "routing provider budget recovery could not be confirmed"
                ) from exc
            raise
        route_family = "deepline" if binding.provider_id == "deepline" else "provider_broker_v2"
        broker_request = {
            "schema_version": "leadpoet.research_lab.routing_broker_request.v2",
            "operation": ROUTING_EXPERIMENT_OPERATION_V2,
            "purpose": ROUTING_EXPERIMENT_PROVIDER_PURPOSE_V2,
            "provider_route": route_family,
            "binding_id": binding.binding_id,
            "provider_id": binding.provider_id,
            "tool_id": binding.tool_id,
            "unit_ref": unit_ref,
            "request_fingerprint": request_fingerprint,
            "timeout_ms": authorization.timeout_ceiling_ms,
            "credit_cap_microunits": authorization.remaining_credit_microunits,
            "idempotency_key": request_fingerprint,
        }
        # Persist the dispatch boundary before invoking the broker.  If this
        # write cannot be confirmed, the provider path does not start and the
        # reservation remains conservatively uncertain rather than becoming a
        # reusable budget slot after a process crash.
        try:
            self.store.append_event(
                experiment_hash=experiment_hash,
                event_type="provider_dispatch_started",
                event_doc={
                    "schema_version": "leadpoet.research_lab.routing_provider_dispatch.v2",
                    "reservation_id": reservation_id,
                    "binding_id": binding.binding_id,
                    "request_fingerprint": request_fingerprint,
                },
                claim=claim,
            )
        except Exception:
            try:
                self.store.mark_budget_uncertain(
                    event_key=_budget_event_key(
                        "uncertain", reservation_id, request_fingerprint
                    ),
                    reservation_id=reservation_id,
                    claim=claim,
                    event_doc={
                        "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
                        "request_fingerprint": request_fingerprint,
                        "billing_state": "uncertain",
                        "recovery_reason": "dispatch_marker_persistence_failed",
                    },
                )
            except Exception:
                pass
            raise
        try:
            result = self.broker_executor(broker_request)
            receipt = self._receipt_from_result(
                binding=binding,
                unit_ref=unit_ref,
                request_fingerprint=request_fingerprint,
                authorization=authorization,
                result=result,
            )
            key = provider_receipt_key(
                tool_id=receipt.tool_id,
                binding_version=receipt.binding_version,
                request_fingerprint=receipt.request_fingerprint,
            )
            self.store.append_provider_attempt(
                experiment_hash=experiment_hash,
                key=key,
                receipt=receipt,
                variant_id=authorization.variant_id,
                claim=claim,
                billing_state=str(result["billing_state"]),
                authoritative_billed_credit_microunits=(
                    int(result["credit_microunits"])
                    if result["billing_state"] == "known"
                    else None
                ),
            )
            if result.get("billing_state") == "known":
                self.store.settle_budget(
                    event_key=_budget_event_key("settle", reservation_id, key),
                    reservation_id=reservation_id,
                    attempt_key=key,
                    claim=claim,
                    event_doc={
                        "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
                        "attempt_key": key,
                        "billing_state": "known",
                    },
                )
            else:
                self.store.mark_budget_uncertain(
                    event_key=_budget_event_key("uncertain", reservation_id, key),
                    reservation_id=reservation_id,
                    claim=claim,
                    event_doc={
                        "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
                        "attempt_key": key,
                        "billing_state": "uncertain",
                    },
                )
            return receipt.to_dict()
        except Exception:
            # A timeout or process boundary error may still be billed.  Keep
            # the reservation as uncertain before allowing the pure runner to
            # record its typed adapter-failure receipt and retry identity.
            self.store.mark_budget_uncertain(
                event_key=_budget_event_key("uncertain", reservation_id, request_fingerprint),
                reservation_id=reservation_id,
                claim=claim,
                event_doc={
                    "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
                    "request_fingerprint": request_fingerprint,
                    "billing_state": "uncertain",
                },
            )
            raise

    def for_execution(
        self,
        experiment_hash: str,
        experiment_id: str,
        claim: RoutingExperimentExecutionClaim,
        *,
        deadline_supplier: Callable[[], float] | None = None,
    ) -> "ProviderBrokerRoutingRunner":
        normalized = str(experiment_hash or "").strip().lower()
        if not normalized.startswith("sha256:") or len(normalized) != 71:
            raise RoutingExperimentRuntimeError("routing experiment hash is invalid")
        if claim.experiment_hash != normalized:
            raise RoutingExperimentRuntimeError("routing runner claim belongs to another experiment")
        if not str(experiment_id or "").strip():
            raise RoutingExperimentRuntimeError("routing experiment id is invalid")
        return ProviderBrokerRoutingRunner(
            config=self.config,
            store=self.store,
            execution_authority=self.execution_authority,
            broker_executor=self.broker_executor,
            execution=_RoutingExecutionBinding(
                experiment_hash=normalized,
                experiment_id=str(experiment_id),
                claim=claim,
                deadline_monotonic=(
                    time.monotonic() + max(1, self.config.worker_lease_seconds - 15)
                ),
                deadline_supplier=deadline_supplier,
            ),
        )

    def _receipt_from_result(
        self,
        *,
        binding: ProviderBindingIdentity,
        unit_ref: str,
        request_fingerprint: str,
        authorization: RoutingCallAuthorization,
        result: Mapping[str, Any],
    ) -> ProviderReceipt:
        allowed = {
            "outcome",
            "evidence_hash",
            "credit_microunits",
            "latency_ms",
            "billing_state",
            "binding_id",
            "provider_id",
            "tool_id",
            "request_fingerprint",
        }
        if set(result) != allowed:
            raise RoutingExperimentRuntimeError("routing broker result fields are invalid")
        outcome = str(result.get("outcome") or "")
        if outcome not in {item.value for item in ProviderOutcome}:
            raise RoutingExperimentRuntimeError("routing broker outcome is invalid")
        billing_state = str(result.get("billing_state") or "")
        if billing_state not in {"known", "uncertain"}:
            raise RoutingExperimentRuntimeError("routing broker billing state is invalid")
        credit = result.get("credit_microunits")
        latency = result.get("latency_ms")
        if type(credit) is not int or credit < 0 or type(latency) is not int or latency < 0:
            raise RoutingExperimentRuntimeError("routing broker cost or latency is invalid")
        if (
            result.get("binding_id") != binding.binding_id
            or result.get("provider_id") != binding.provider_id
            or result.get("tool_id") != binding.tool_id
            or result.get("request_fingerprint") != request_fingerprint
        ):
            raise RoutingExperimentRuntimeError("routing broker result identity is invalid")
        if credit > authorization.remaining_credit_microunits:
            raise RoutingExperimentRuntimeError("routing broker result exceeds authorized credit")
        if latency > authorization.timeout_ceiling_ms:
            raise RoutingExperimentRuntimeError("routing broker result exceeds authorized timeout")
        if billing_state == "uncertain" and credit != 0:
            raise RoutingExperimentRuntimeError("routing broker uncertain charge must be zero")
        identity = {
            "binding_id": binding.binding_id,
            "tool_id": binding.tool_id,
            "binding_version": binding.adapter_version,
            "source_lineage_id": binding.source_lineage_id,
            "unit_ref": unit_ref,
            "request_fingerprint": request_fingerprint,
            "outcome": outcome,
            "evidence_hash": result.get("evidence_hash"),
            "credit_microunits": credit,
            "latency_ms": latency,
            "execution_mode": ReceiptExecutionMode.MEASURED_LAB.value,
        }
        return ProviderReceipt(
            receipt_ref="provider_receipt:" + sha256_json(identity).split(":", 1)[1][:16],
            **identity,
        )


class ReviewedProviderBrokerRoutingRunner:
    """Production runner for signed inputs and protected provider dispatch.

    This path accepts no broker, request factory, result callback, endpoint, or
    raw unit data. The signed catalog and unit dataset compile the request. The
    separate protected authorization and dispatch operations bind and execute
    it before a redacted provider receipt is persisted.
    """

    def __init__(
        self,
        *,
        config: RoutingExperimentRuntimeConfig,
        store: SupabaseRoutingExperimentStore,
        artifact_lineage: VerifiedRoutingArtifactLineage,
        compiler: ReviewedDeeplineActionCompiler,
        model_binding_requirements: VerifiedRoutingModelBindingRequirements,
        authorization_authority: AttestedScoringV2RoutingProviderCallAuthority,
        dispatch_authority: AttestedScoringV2RoutingProviderDispatchAuthority,
        execution_envelope: RoutingExperimentExecutionEnvelopeV2 | None = None,
        admission_bundle: RoutingAdmissionBundleV2 | None = None,
        protected_release_receipt: Mapping[str, Any] | None = None,
        authorization_parent_receipt_graphs: Sequence[Mapping[str, Any]] = (),
        dispatch_parent_receipt_graphs: Sequence[Mapping[str, Any]] = (),
        admission_validator: Callable[[RoutingAdmissionBundleV2, Mapping[str, Any]], None]
        | None = None,
        execution: _RoutingExecutionBinding | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.artifact_lineage = artifact_lineage
        self.compiler = compiler
        self.model_binding_requirements = model_binding_requirements
        if not isinstance(
            authorization_authority, AttestedScoringV2RoutingProviderCallAuthority
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider call authorization authority is invalid"
            )
        self.authorization_authority = authorization_authority
        if not isinstance(
            dispatch_authority, AttestedScoringV2RoutingProviderDispatchAuthority
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authority is invalid"
            )
        self.dispatch_authority = dispatch_authority
        self.execution_envelope = execution_envelope
        self.admission_bundle = admission_bundle
        self.protected_release_receipt = (
            dict(protected_release_receipt)
            if isinstance(protected_release_receipt, Mapping)
            else None
        )
        self.admission_validator = admission_validator
        self.authorization_parent_receipt_graphs = tuple(
            dict(item) for item in authorization_parent_receipt_graphs
        )
        self.dispatch_parent_receipt_graphs = tuple(
            dict(item) for item in dispatch_parent_receipt_graphs
        )
        self._execution = execution

    def validate_composition(self) -> None:
        """Assert that a factory returned a fully constructed live runner."""

        if not isinstance(
            self.dispatch_authority, AttestedScoringV2RoutingProviderDispatchAuthority
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authority is invalid"
            )
        if not isinstance(
            self.authorization_authority,
            AttestedScoringV2RoutingProviderCallAuthority,
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization authority is invalid"
            )
        self.authorization_authority.validate_composition()
        self.dispatch_authority.validate_composition()
        if hasattr(self, "broker") or hasattr(self, "broker_executor"):
            raise RoutingExperimentRuntimeError(
                "routing reviewed runner contains a direct broker"
            )
        if not self.authorization_parent_receipt_graphs:
            raise RoutingExperimentRuntimeError(
                "routing provider authorization ancestry is unavailable"
            )
        if not self.dispatch_parent_receipt_graphs:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch ancestry is unavailable"
            )

    def for_execution(
        self,
        experiment_hash: str,
        experiment_id: str,
        claim: RoutingExperimentExecutionClaim,
        *,
        deadline_supplier: Callable[[], float] | None = None,
    ) -> "ReviewedProviderBrokerRoutingRunner":
        self.validate_composition()
        if not callable(deadline_supplier):
            raise RoutingExperimentRuntimeError(
                "routing reviewed runner requires an authoritative claim deadline"
            )
        initial_deadline = deadline_supplier()
        if (
            not isinstance(initial_deadline, (int, float))
            or isinstance(initial_deadline, bool)
            or initial_deadline <= time.monotonic()
        ):
            raise RoutingExperimentRuntimeError(
                "routing authoritative claim deadline is exhausted"
            )
        normalized = str(experiment_hash or "").strip().lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
            raise RoutingExperimentRuntimeError("routing experiment hash is invalid")
        if claim.experiment_hash != normalized:
            raise RoutingExperimentRuntimeError(
                "routing runner claim belongs to another experiment"
            )
        return ReviewedProviderBrokerRoutingRunner(
            config=self.config,
            store=self.store,
            artifact_lineage=self.artifact_lineage,
            compiler=self.compiler,
            model_binding_requirements=self.model_binding_requirements,
            authorization_authority=self.authorization_authority,
            dispatch_authority=self.dispatch_authority,
            execution_envelope=self.execution_envelope,
            admission_bundle=self.admission_bundle,
            protected_release_receipt=self.protected_release_receipt,
            authorization_parent_receipt_graphs=(
                self.authorization_parent_receipt_graphs
            ),
            dispatch_parent_receipt_graphs=self.dispatch_parent_receipt_graphs,
            admission_validator=self.admission_validator,
            execution=_RoutingExecutionBinding(
                experiment_hash=normalized,
                experiment_id=str(experiment_id),
                claim=claim,
                deadline_monotonic=float(initial_deadline),
                deadline_supplier=deadline_supplier,
            ),
        )

    def __call__(
        self,
        binding: ProviderBindingIdentity,
        unit_ref: str,
        request_fingerprint: str,
        authorization: RoutingCallAuthorization,
    ) -> Mapping[str, Any]:
        self.config.assert_live_enabled()
        execution = self._execution
        if execution is None:
            raise RoutingExperimentRuntimeError("routing runner is not bound to an experiment")
        if (
            authorization.execution_mode != ReceiptExecutionMode.MEASURED_LAB.value
            or authorization.experiment_id != execution.experiment_id
            or authorization.tool_id != binding.tool_id
            or authorization.unit_ref != unit_ref
            or authorization.request_fingerprint != request_fingerprint
        ):
            raise RoutingExperimentRuntimeError(
                "routing call authorization does not bind the request"
            )
        admission = self.admission_bundle
        if admission is None:
            raise RoutingExperimentRuntimeError(
                "routing provider admission bundle is unavailable"
            )
        if (
            self.execution_envelope is None
            or self.execution_envelope.envelope_hash() != admission.envelope_hash
            or self.execution_envelope.experiment_hash
            != execution.experiment_hash
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider execution envelope is unavailable"
            )
        if (
            admission.experiment_hash != execution.experiment_hash
            or admission.experiment_id != execution.experiment_id
            or admission.purpose != ROUTING_EXPERIMENT_PROVIDER_PURPOSE_V2
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider admission bundle scope differs"
            )
        if self.admission_validator is None:
            raise RoutingExperimentRuntimeError(
                "routing provider admission authority validator is unavailable"
            )
        if self.protected_release_receipt is None:
            raise RoutingExperimentRuntimeError(
                "routing provider protected release receipt is unavailable"
            )
        try:
            self.admission_validator(admission, self.protected_release_receipt)
            validate_signed_execution_receipt(self.protected_release_receipt)
        except Exception as exc:
            raise RoutingExperimentRuntimeError(
                "routing provider admission authority is invalid"
            ) from exc
        if any(
            self.protected_release_receipt.get(receipt_name)
            != getattr(admission, admission_name)
            for receipt_name, admission_name in (
                ("commit_sha", "protected_commit_sha"),
                ("pcr0", "protected_pcr0"),
                ("build_manifest_hash", "protected_build_manifest_hash"),
                ("dependency_lock_hash", "protected_dependency_lock_hash"),
                ("config_hash", "protected_config_hash"),
                ("boot_identity_hash", "protected_boot_identity_hash"),
                ("enclave_pubkey", "protected_enclave_pubkey"),
            )
        ) or admission.protected_receipt_hash != self.protected_release_receipt.get(
            "receipt_hash"
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider protected release identity differs"
            )
        deadline = (
            execution.deadline_supplier()
            if execution.deadline_supplier is not None
            else execution.deadline_monotonic
        )
        remaining_lease_ms = int((deadline - time.monotonic()) * 1000)
        if authorization.timeout_ceiling_ms > remaining_lease_ms or remaining_lease_ms <= 0:
            raise RoutingExperimentRuntimeError(
                "routing call timeout exceeds remaining claim lease"
            )
        requirement_hash = self.model_binding_requirements.resolve(
            binding=binding,
            artifact_lineage_hash=self.artifact_lineage.identity_hash(),
        )
        prepared = self.compiler.prepare(
            binding=binding,
            unit_ref=unit_ref,
            authorization_credit_microunits=authorization.remaining_credit_microunits,
            authorization_timeout_ms=authorization.timeout_ceiling_ms,
            expected_model_binding_requirements_hash=requirement_hash,
            phase=authorization.phase,
            execution_mode=authorization.execution_mode,
        )
        # Stop composite routes before reservation/dispatch.  The released
        # V3 persistence and protected operation are direct-action only.
        prepared = _require_reviewed_direct_prepared_call(prepared)
        claim_commitment = execution.claim.claim_fence
        exact_authorization = RoutingProviderCallAuthorizationV2(
            admission_job_id=admission.job_id,
            experiment_hash=execution.experiment_hash,
            experiment_id=execution.experiment_id,
            purpose=ROUTING_EXPERIMENT_PROVIDER_PURPOSE_V2,
            envelope_hash=admission.envelope_hash,
            admission_bundle_hash=admission.identity_hash(),
            protected_release_hash=admission.protected_release_hash,
            protected_boot_identity_hash=admission.protected_boot_identity_hash,
            variant_id=authorization.variant_id,
            stage=authorization.stage,
            artifact_lineage_hash=self.artifact_lineage.identity_hash(),
            pointer_document_hash=self.artifact_lineage.pointer_document_hash,
            model_artifact_hash=self.artifact_lineage.model_artifact_hash,
            manifest_hash=self.artifact_lineage.manifest_hash,
            image_digest=self.artifact_lineage.image_digest,
            commit_sha=self.artifact_lineage.commit_sha,
            build_id=self.artifact_lineage.build_id,
            routing_contract_hash=self.artifact_lineage.routing_contract_hash,
            routing_catalog_hash=self.artifact_lineage.routing_catalog_hash,
            routing_policy_hash=self.artifact_lineage.routing_policy_hash,
            feature_schema_hash=self.artifact_lineage.feature_schema_hash,
            verifier_contract_hash=self.artifact_lineage.verifier_contract_hash,
            binding=binding,
            transport_id=prepared.transport_id,
            binding_catalog_manifest_hash=prepared.binding_catalog_manifest_hash,
            binding_catalog_version=prepared.binding_catalog_version,
            action_id=prepared.action_id,
            unit_ref=unit_ref,
            unit_input_hash=prepared.unit_input_hash,
            unit_dataset_manifest_hash=prepared.unit_dataset_manifest_hash,
            unit_set_hash=prepared.unit_set_hash,
            model_binding_observation_receipt_hash=(
                admission.model_binding_observation_receipt_hash
            ),
            attempt=authorization.attempt,
            core_request_fingerprint=request_fingerprint,
            request_body_hash=prepared.request_body_hash,
            retry_policy_hash=prepared.retry_policy_hash,
            credit_cap_microunits=prepared.credit_ceiling_microunits,
            timeout_ms=prepared.timeout_ms,
            claim_key=execution.claim.claim_key,
            claim_generation=execution.claim.claim_generation,
            claim_fence_hash=claim_commitment,
        )
        proof = self.authorization_authority.authorize(
            exact_authorization,
            artifact_lineage=self.artifact_lineage,
            model_binding_observation=self.model_binding_requirements,
            execution_envelope=self.execution_envelope,
            admission_bundle=admission,
            prepared_call=prepared,
            protected_release_receipt=self.protected_release_receipt,
            parent_receipt_graphs=self.authorization_parent_receipt_graphs,
        )
        authorization_receipt = proof.get("authorization_receipt")
        if not isinstance(authorization_receipt, Mapping):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization receipt is unavailable"
            )
        authorization_result = proof.get("authorization_result")
        authorization_job_id = str(authorization_receipt.get("job_id") or "")
        if (
            not isinstance(authorization_result, Mapping)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", authorization_job_id
            )
            or authorization_result.get("authorization_job_id") != authorization_job_id
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider authorization execution job identity is unavailable"
            )
        broker_request = self.compiler.broker_request(
            prepared=prepared,
            experiment_hash=execution.experiment_hash,
            dispatch_job_id=routing_provider_dispatch_job_id_v2(proof),
            variant_id=authorization.variant_id,
            attempt_number=authorization.attempt,
            core_request_fingerprint=request_fingerprint,
            authorization_hash=exact_authorization.authorization_hash(),
            authorization_proof_hash=str(proof["authorization_proof_hash"]),
        )
        # The compiler owns the transport shape, while the protected scoring
        # dispatch operation owns the complete proof and validates it again
        # before crossing to the coordinator.
        broker_request = dict(broker_request)
        broker_request["routing_authorization"] = dict(proof)
        if broker_request.get("routing_authorization") != proof:
            raise RoutingExperimentRuntimeError(
                "routing provider proof differs from compiled request"
            )
        budget_reservation = build_routing_budget_reservation_v3(
            authorization=exact_authorization,
            prepared_call=prepared,
            lease_seconds=self.config.worker_lease_seconds,
        )
        reservation_id = str(budget_reservation["reservation_id"])
        reserve_doc = dict(budget_reservation["event_doc"])
        # Do not reserve or dispatch a provider call unless the durable store
        # has the protected dispatch contract. The host never receives a raw
        # broker result and cannot construct a terminal proof locally.
        if self.dispatch_authority is None:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authority is unavailable"
            )
        if not callable(
            getattr(self.store, "append_protected_provider_attempt", None)
        ):
            raise RoutingExperimentRuntimeError(
                "routing provider protected terminal persistence is unavailable"
            )
        if not self.dispatch_parent_receipt_graphs:
            raise RoutingExperimentRuntimeError(
                "routing provider dispatch authorization ancestry is unavailable"
            )
        try:
            reserve_result = self.store.reserve_budget(
                event_key=str(budget_reservation["event_key"]),
                reservation_id=reservation_id,
                experiment_hash=execution.experiment_hash,
                binding_id=binding.binding_id,
                claim=execution.claim,
                credit_microunits=prepared.credit_ceiling_microunits,
                lease_seconds=self.config.worker_lease_seconds,
                event_doc=reserve_doc,
            )
            try:
                validate_routing_budget_reservation_result_v3(
                    reserve_result,
                    reservation=budget_reservation,
                )
            except Exception as exc:
                raise RoutingExperimentRuntimeError(
                    "routing provider durable budget reservation differs"
                ) from exc
        except Exception as exc:
            self._mark_budget_uncertain_or_defer(
                reservation_id=reservation_id,
                claim=execution.claim,
                event_doc={
                    **reserve_doc,
                    "billing_state": "uncertain",
                    "recovery_reason": "reservation_failed_or_invalid",
                },
                original=exc,
                expected_credit_microunits=prepared.credit_ceiling_microunits,
            )
            raise
        try:
            self.store.append_event(
                experiment_hash=execution.experiment_hash,
                event_type="provider_dispatch_started",
                event_doc={
                    **reserve_doc,
                    "reservation_id": reservation_id,
                },
                claim=execution.claim,
            )
            dispatch_response = self.dispatch_authority.execute(
                authorization_proof=proof,
                prepared_call=prepared,
                broker_request=broker_request,
                budget_reservation=budget_reservation,
                parent_receipt_graphs=self.dispatch_parent_receipt_graphs,
            )
            terminal_result = dispatch_response.get("result")
            if not isinstance(terminal_result, Mapping):
                raise RoutingExperimentRuntimeError(
                    "routing provider dispatch protected result is invalid"
                )
            provider_receipt_document = terminal_result.get("provider_receipt")
            if not isinstance(provider_receipt_document, Mapping):
                raise RoutingExperimentRuntimeError(
                    "routing provider dispatch provider receipt is unavailable"
                )
            try:
                receipt = ProviderReceipt.from_mapping(provider_receipt_document)
            except Exception as exc:
                raise RoutingExperimentRuntimeError(
                    "routing provider dispatch provider receipt is invalid"
                ) from exc
            receipt_errors = validate_provider_receipt(receipt)
            if receipt_errors:
                raise RoutingExperimentRuntimeError(
                    "routing provider dispatch provider receipt is invalid"
                )
            projected = {
                "outcome": receipt.outcome,
                "evidence_hash": receipt.evidence_hash,
                "credit_microunits": receipt.credit_microunits,
                "latency_ms": receipt.latency_ms,
                "billing_state": "known",
                "binding_id": binding.binding_id,
                "provider_id": binding.provider_id,
                "tool_id": binding.tool_id,
                "request_fingerprint": request_fingerprint,
            }
            if (
                receipt.binding_id != binding.binding_id
                or receipt.tool_id != binding.tool_id
                or receipt.unit_ref != unit_ref
                or receipt.request_fingerprint != request_fingerprint
                or receipt.execution_mode != ReceiptExecutionMode.MEASURED_LAB.value
                or terminal_result.get("projection") != projected
            ):
                raise RoutingExperimentRuntimeError(
                    "routing provider dispatch provider receipt identity differs"
                )
            budget_proof = terminal_result.get("budget_reservation")
            if (
                not isinstance(budget_proof, Mapping)
                or budget_proof.get("reservation_id") != reservation_id
                or budget_proof.get("event_key")
                != budget_reservation["event_key"]
                or budget_proof.get("experiment_hash")
                != execution.experiment_hash
                or budget_proof.get("binding_id") != binding.binding_id
                or budget_proof.get("claim_key") != execution.claim.claim_key
                or budget_proof.get("claim_generation")
                != execution.claim.claim_generation
                or budget_proof.get("credit_microunits")
                != prepared.credit_ceiling_microunits
            ):
                raise RoutingExperimentRuntimeError(
                    "routing provider protected budget reservation differs"
                )
            key = provider_receipt_key(
                tool_id=receipt.tool_id,
                binding_version=receipt.binding_version,
                request_fingerprint=receipt.request_fingerprint,
            )
            self.store.append_protected_provider_attempt(
                experiment_hash=execution.experiment_hash,
                key=key,
                receipt=receipt,
                variant_id=authorization.variant_id,
                reservation_id=reservation_id,
                action_id=prepared.action_id,
                authorization=exact_authorization,
                authorization_proof_hash=str(proof["authorization_proof_hash"]),
                authorization_request_hash=str(proof["authorization_request_hash"]),
                authorization_receipt=authorization_receipt,
                terminal_result=terminal_result,
                terminal_execution_receipt=dispatch_response.get(
                    "execution_receipt"
                ),
                protected_release_receipt=self.protected_release_receipt,
                admission_bundle=admission,
                claim=execution.claim,
                billing_state=str(projected["billing_state"]),
                authoritative_billed_credit_microunits=(
                    int(projected["credit_microunits"])
                    if projected["billing_state"] == "known"
                    else None
                ),
            )
            if projected["billing_state"] != "known":
                self.store.mark_budget_uncertain(
                    event_key=_budget_event_key("uncertain", reservation_id, key),
                    reservation_id=reservation_id,
                    claim=execution.claim,
                    event_doc={**reserve_doc, "attempt_key": key, "billing_state": "uncertain"},
                )
            else:
                self.store.settle_budget(
                    event_key=_budget_event_key("settle", reservation_id, key),
                    reservation_id=reservation_id,
                    attempt_key=key,
                    claim=execution.claim,
                    event_doc={**reserve_doc, "attempt_key": key, "billing_state": "known"},
                )
            return receipt.to_dict()
        except Exception as exc:
            self._mark_budget_uncertain_or_defer(
                reservation_id=reservation_id,
                claim=execution.claim,
                event_doc={
                    **reserve_doc,
                    "billing_state": "uncertain",
                    "recovery_reason": "dispatch_or_terminal_failed",
                },
                original=exc,
                event_key=_budget_event_key(
                    "uncertain", reservation_id, exact_authorization.authorization_hash()
                ),
                expected_credit_microunits=prepared.credit_ceiling_microunits,
            )
            raise

    def _mark_budget_uncertain_or_defer(
        self,
        *,
        reservation_id: str,
        claim: RoutingExperimentExecutionClaim,
        event_doc: Mapping[str, Any],
        original: BaseException,
        event_key: str | None = None,
        expected_credit_microunits: int | None = None,
    ) -> None:
        """Confirm durable uncertainty before allowing queue terminalization."""

        try:
            result = self.store.mark_budget_uncertain(
                event_key=event_key
                or _budget_event_key(
                    "uncertain", reservation_id, str(event_doc.get("request_fingerprint") or "")
                ),
                reservation_id=reservation_id,
                claim=claim,
                event_doc=dict(event_doc),
            )
            if (
                not isinstance(result, Mapping)
                or result.get("uncertain") is not True
                or (
                    expected_credit_microunits is not None
                    and result.get("credit_microunits")
                    != expected_credit_microunits
                )
            ):
                raise RoutingExperimentRuntimeError(
                    "routing provider budget uncertainty result is not confirmed"
                )
        except Exception:
            raise RoutingExperimentDeferredRecoveryError(
                "routing provider budget recovery could not be confirmed"
            ) from original


class RoutingExperimentService:
    """Separate service; it does not enter scoring or production promotion."""

    def __init__(
        self,
        *,
        config: RoutingExperimentRuntimeConfig,
        store: SupabaseRoutingExperimentStore,
    ) -> None:
        self.config = config
        self.store = store

    def submit(
        self,
        spec: RoutingExperimentV2Spec,
        *,
        execution_envelope: RoutingExperimentExecutionEnvelopeV2 | None = None,
    ) -> Mapping[str, Any]:
        if not self.config.enabled:
            raise RoutingExperimentRuntimeError("routing experiment worker is disabled")
        if spec.allow_live_credit_spend:
            self.config.assert_live_enabled()
        return self.store.submit(spec, execution_envelope=execution_envelope)

    def claim_execution(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        worker_ref: str,
        lease: RoutingExecutionRequestLease | None = None,
    ) -> RoutingExperimentExecutionClaim:
        """Bind one run to its active queue lease before any provider work."""

        experiment_hash = spec.experiment_hash()
        if not isinstance(lease, RoutingExecutionRequestLease):
            raise RoutingExperimentRuntimeError(
                "an active execution queue lease is required before claiming"
            )
        if lease.experiment_hash != experiment_hash:
            raise RoutingExperimentRuntimeError(
                "execution queue lease belongs to another experiment"
            )
        if lease.worker_ref != worker_ref:
            raise RoutingExperimentRuntimeError(
                "execution queue lease belongs to another worker"
            )
        claim_key = sha256_json(
            {
                "schema_version": "leadpoet.research_lab.routing_claim_key.v3",
                "experiment_hash": experiment_hash,
                "request_hash": lease.request_hash,
                "lease_hash": lease.lease_hash,
                "lease_generation": lease.lease_generation,
                "worker_ref": worker_ref,
            }
        )
        result = self.store.claim_execution(
            experiment_hash=experiment_hash,
            request_hash=lease.request_hash,
            lease_hash=lease.lease_hash,
            lease_generation=lease.lease_generation,
            claim_key=claim_key,
            worker_ref=worker_ref,
            lease_seconds=self.config.worker_lease_seconds,
            claim_doc={
                "schema_version": "leadpoet.research_lab.routing_claim.v3",
                "request_hash": lease.request_hash,
                "lease_hash": lease.lease_hash,
                "lease_generation": lease.lease_generation,
                "worker_ref": worker_ref,
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentRuntimeError("routing claim response is malformed")
        recovered_stale_claim = result.get("recoverable") is True
        if recovered_stale_claim:
            stale_claim_key = str(result.get("claim_key") or "")
            stale_generation = result.get("claim_generation")
            if (
                not re.fullmatch(r"sha256:[0-9a-f]{64}", stale_claim_key)
                or type(stale_generation) is not int
                or stale_generation < 1
            ):
                raise RoutingExperimentRuntimeError("routing stale claim response is invalid")
            recovery_key = sha256_json(
                {
                    "schema_version": "leadpoet.research_lab.routing_claim_recovery_key.v3",
                    "experiment_hash": experiment_hash,
                    "stale_claim_key": stale_claim_key,
                    "stale_claim_generation": stale_generation,
                    "request_hash": lease.request_hash,
                    "lease_hash": lease.lease_hash,
                    "lease_generation": lease.lease_generation,
                    "worker_ref": worker_ref,
                }
            )
            # The recovery key is distinct from both the prospective and stale
            # claim keys, so the recovery RPC cannot collide with the existing
            # claimed row returned by SQL.
            self.store.recover_claim(
                experiment_hash=experiment_hash,
                recovery_key=recovery_key,
                worker_ref=worker_ref,
                recovery_doc={
                    "schema_version": "leadpoet.research_lab.routing_claim_recovery.v3",
                    "worker_ref": worker_ref,
                    "stale_claim_key": stale_claim_key,
                    "stale_claim_generation": stale_generation,
                },
            )
            # Recovery is terminal.  The provider boundary may have been
            # crossed before the stale worker disappeared, so the authority
            # retains every open reservation at its full uncertain ceiling.
            # Never issue a fresh claim for this experiment: a new immutable
            # experiment is required before any provider call can resume.
            raise RoutingExperimentTerminalRecoveryError(
                "routing experiment claim recovered; submit a new immutable experiment"
            )
        returned_claim_key = str(result.get("claim_key") or claim_key)
        if returned_claim_key != claim_key:
            raise RoutingExperimentRuntimeError("routing claim identity differs")
        if result.get("claimed") is not True:
            raise RoutingExperimentRuntimeError("routing experiment is already claimed")
        generation = result.get("claim_generation")
        if type(generation) is not int or generation < 1:
            raise RoutingExperimentRuntimeError("routing claim generation is invalid")
        returned_request_hash = str(result.get("request_hash") or lease.request_hash)
        returned_lease_hash = str(result.get("lease_hash") or lease.lease_hash)
        returned_lease_generation = result.get("lease_generation", lease.lease_generation)
        returned_lease_expiry = str(result.get("lease_expires_at") or "")
        if (
            returned_request_hash != lease.request_hash
            or returned_lease_hash != lease.lease_hash
            or returned_lease_generation != lease.lease_generation
        ):
            raise RoutingExperimentRuntimeError("routing claim queue identity differs")
        claim = RoutingExperimentExecutionClaim(
            experiment_hash=experiment_hash,
            claim_key=claim_key,
            claim_generation=generation,
            claim_fence_hash=sha256_json(
                {
                    "schema_version": "leadpoet.research_lab.routing_claim_fence.v3",
                    "experiment_hash": experiment_hash,
                    "claim_key": claim_key,
                    "claim_generation": generation,
                }
            ),
            request_hash=lease.request_hash,
            lease_hash=lease.lease_hash,
            lease_generation=lease.lease_generation,
            worker_ref=worker_ref,
            lease_expires_at=returned_lease_expiry,
        )
        if recovered_stale_claim:
            self._block_on_unresolved_recovered_budget_heads(
                experiment_hash=experiment_hash,
                claim=claim,
            )
        return claim

    def _block_on_unresolved_recovered_budget_heads(
        self,
        *,
        experiment_hash: str,
        claim: RoutingExperimentExecutionClaim,
    ) -> None:
        """Turn expired reservations into unknown charges and halt the run.

        There is no broker-issued, durable proof in this service that a
        process died before a request crossed the provider boundary.  We
        therefore never release an expired reservation to zero.  A later,
        independently authoritative broker outcome may settle it; until then
        a resumed run cannot consume more provider credit or produce a Lab
        reference.
        """

        unresolved = self.store.unresolved_budget_reservations(
            experiment_hash=experiment_hash,
            claim=claim,
        )
        if not unresolved:
            return
        for reservation in unresolved:
            if reservation.event_type != "reserve" or not reservation.lease_expired:
                continue
            self.store.mark_budget_uncertain(
                event_key=_budget_event_key(
                    "uncertain_recovery",
                    reservation.reservation_id,
                    claim.claim_key,
                    str(claim.claim_generation),
                ),
                reservation_id=reservation.reservation_id,
                claim=claim,
                event_doc={
                    "schema_version": "leadpoet.research_lab.routing_budget_event.v2",
                    "reservation_id": reservation.reservation_id,
                    "binding_id": reservation.binding_id,
                    "billing_state": "uncertain",
                    "recovery_reason": (
                        "expired_after_dispatch_marker"
                        if reservation.dispatch_started
                        else "expired_without_authoritative_no_call_proof"
                    ),
                },
            )
        raise RoutingExperimentTerminalRecoveryError(
            "routing experiment has unresolved provider budget; authoritative broker settlement is required"
        )

    def evaluate(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        gold_labels: Mapping[str, bool],
        adapters: Mapping[str, RoutingExperimentV2Adapter],
        runner: Callable[..., ProviderReceipt | Mapping[str, Any]],
        artifact_authority: RoutingExperimentArtifactAuthority | None,
        authoritative_billing_rollup: Callable[[ProviderReceiptStore], Mapping[str, Any]] | None = None,
        execution_envelope: RoutingExperimentExecutionEnvelopeV2 | None = None,
        worker_ref: str = "routing-experiment-service",
        claim: RoutingExperimentExecutionClaim | None = None,
        lease: RoutingExecutionRequestLease | None = None,
        claim_deadline_supplier: Callable[[], float] | None = None,
    ) -> RoutingExperimentV2Evaluation:
        if not self.config.enabled:
            raise RoutingExperimentRuntimeError("routing experiment worker is disabled")
        self.submit(spec, execution_envelope=execution_envelope)
        if claim is None:
            claim = self.claim_execution(
                spec=spec,
                worker_ref=worker_ref,
                lease=lease,
            )
        elif claim.experiment_hash != spec.experiment_hash():
            raise RoutingExperimentRuntimeError("routing evaluation claim belongs to another experiment")
        receipt_store: ProviderReceiptStore | None = ProviderReceiptStore(
            SupabaseRoutingProviderReceiptRepository(
                store=self.store,
                experiment_hash=spec.experiment_hash(),
                claim=claim,
            )
        )
        decision_store: RoutingDecisionReceiptStore | None = RoutingDecisionReceiptStore(
            SupabaseRoutingDecisionReceiptRepository(
                store=self.store,
                experiment_hash=spec.experiment_hash(),
                claim=claim,
            )
        )
        if spec.allow_live_credit_spend:
            self.config.assert_live_enabled()
            if not isinstance(runner, ReviewedProviderBrokerRoutingRunner):
                raise RoutingExperimentRuntimeError(
                    "routing live execution requires the reviewed provider runner"
                )
            try:
                runner.validate_composition()
            except Exception as exc:
                raise RoutingExperimentRuntimeError(
                    "routing live execution runner dispatch composition is invalid"
                ) from exc
            runner = runner.for_execution(
                spec.experiment_hash(),
                spec.experiment_id,
                claim,
                deadline_supplier=claim_deadline_supplier,
            )
        evaluation = evaluate_routing_experiment_v2(
            spec,
            gold_labels=gold_labels,
            runner=runner,
            adapters=adapters,
            receipt_store=receipt_store,
            decision_store=decision_store,
            authoritative_billing_rollup=authoritative_billing_rollup,
            artifact_authority=artifact_authority,
            require_isolation=spec.allow_live_credit_spend,
        )
        self.store.append_evaluation(spec=spec, evaluation=evaluation, claim=claim)
        self.store.append_event(
            experiment_hash=spec.experiment_hash(),
            event_type="run_completed",
            event_doc={
                "evaluation_receipt_id": evaluation.receipt_id,
                "selected_variant_id": evaluation.selected_variant_id or "unselected",
            },
            claim=claim,
        )
        return evaluation


__all__ = [
    "ROUTING_EXPERIMENT_OPERATION_V2",
    "ROUTING_EXPERIMENT_PURPOSE_V2",
    "ROUTING_EXPERIMENT_PROVIDER_PURPOSE_V2",
    "ROUTING_PROVIDER_DISPATCH_OPERATION_V2",
    "ROUTING_PROVIDER_DISPATCH_PURPOSE_V2",
    "RoutingExperimentRuntimeError",
    "RoutingExperimentTerminalRecoveryError",
    "RoutingExperimentDeferredRecoveryError",
    "RoutingProviderDispatchExecutor",
    "RoutingExperimentRuntimeConfig",
    "AttestedRoutingExperimentExecutionAuthority",
    "FailClosedRoutingExperimentExecutionAuthority",
    "AttestedScoringV2RoutingProviderCallAuthority",
    "AttestedScoringV2RoutingProviderDispatchAuthority",
    "AttestedScoringV2RoutingProviderTerminalAuthority",
    "AttestedScoringV2RoutingEvaluationAuthority",
    "KmsRoutingExperimentArtifactAuthority",
    "ProviderBrokerRoutingExecutor",
    "ProviderBrokerV2RoutingExecutor",
    "ProviderBrokerRoutingRunner",
    "RoutingExperimentService",
]
