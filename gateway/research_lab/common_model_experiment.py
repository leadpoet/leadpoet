"""PR93 control-plane loop for PR274's exact Model runner protocol.

The Lab owns admission, claims, protected dispatch, billing, persistence, and
evaluation. It does not compile a route or choose a provider. Every action is
emitted and validated by the exact immutable Model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Mapping, Protocol

from gateway.research_lab.routing_experiment_runtime import (
    ReviewedProviderBrokerRoutingRunner,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    build_routing_model_completion_contract_v1,
)
from research_lab.common_model_runner_host import (
    HostActionResult,
    ModelRunnerHostError,
)
from research_lab.model_runner_protocol import ExactModelRunnerRegistration
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    ProviderOutcome,
    ProviderReceipt,
    ReceiptExecutionMode,
    RoutingCallAuthorization,
    RoutingExperimentV2Spec,
)
from research_lab.canonical import sha256_json


class CommonModelExperimentError(RuntimeError):
    """The exact Model action chain cannot cross the Lab boundary safely."""


class CommonModelExperimentRecoveryError(CommonModelExperimentError):
    """A durable paid-call marker prevents unsafe replay after restart."""


@dataclass(frozen=True)
class ProtectedModelActionResult:
    """One host result and its optional protected provider receipt."""

    host_result: HostActionResult
    provider_receipt: ProviderReceipt | None = None
    replay_ref: Mapping[str, Any] | None = None
    model_provider_response_ingestion: Mapping[str, Any] | None = None


class ExactModelActionDispatcher(Protocol):
    """Narrow host actions; no broker, endpoint, credential, or execute hook."""

    def dispatch_provider_action(
        self,
        *,
        action: Mapping[str, Any],
        variant_id: str,
        unit_ref: str,
    ) -> ProtectedModelActionResult: ...

    def verify_company_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult: ...

    def verify_intent_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult: ...

    def verify_contact_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult: ...

    def replay_provider_action(
        self,
        *,
        action: Mapping[str, Any],
        variant_id: str,
        unit_ref: str,
        replay_ref: Mapping[str, Any],
    ) -> ProtectedModelActionResult: ...


class ReviewedModelVerificationAuthority(Protocol):
    """Exact reviewed verifier methods; no generic execution callback."""

    def verify_company(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult: ...

    def verify_intent(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult: ...

    def verify_contact(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult: ...


def _artifact_hash(variant: Any) -> str:
    artifact = variant.artifact
    return sha256_json(
        {
            "model_artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "commit_sha": artifact.commit_sha,
        }
    )


def _action_budget(
    action: Mapping[str, Any],
    *,
    normalization_identity: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        raise CommonModelExperimentError("Model provider action budget is missing")
    if action.get("action_type") == "normalize_icp":
        # V3 normalization owns these bounds directly because it is outside
        # the company-first orchestration step document.  Never substitute a
        # catalog, environment, or host default for either value.
        credit = arguments.get("credit_cap")
        timeout = arguments.get("timeout_seconds")
        call_cap = arguments.get("call_cap")
        if not isinstance(normalization_identity, Mapping):
            raise CommonModelExperimentError(
                "normalization budget differs from artifact-owned top-level bounds"
            )
        expected_call_cap = normalization_identity.get("call_cap")
        expected_credit = normalization_identity.get("credit_cap")
        expected_timeout = normalization_identity.get("timeout_seconds")
        if (
            "step" in arguments
            or type(call_cap) is not int
            or type(call_cap) is not type(expected_call_cap)
            or call_cap != expected_call_cap
            or type(credit) is not type(expected_credit)
            or credit != expected_credit
            or type(timeout) is not type(expected_timeout)
            or timeout != expected_timeout
        ):
            raise CommonModelExperimentError(
                "normalization budget differs from artifact-owned top-level bounds"
            )
    else:
        step = arguments.get("step")
        if not isinstance(step, Mapping):
            raise CommonModelExperimentError(
                "Model provider action budget is missing"
            )
        credit = step.get("credit_cap")
        timeout = step.get("timeout_seconds")
    if (
        isinstance(credit, bool)
        or not isinstance(credit, (int, float))
        or credit < 0
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise CommonModelExperimentError("Model provider action budget is invalid")
    return int(round(float(credit) * 1_000_000)), int(
        round(float(timeout) * 1_000)
    )


class ReviewedProtectedModelActionDispatcher:
    """Bind PR274 actions to PR93's V3 protected dispatch and verifiers."""

    def __init__(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        registrations: Mapping[str, ExactModelRunnerRegistration],
        runner: ReviewedProviderBrokerRoutingRunner,
        claim: Any,
        deadline_supplier: Any,
        verifier: ReviewedModelVerificationAuthority,
    ) -> None:
        if not isinstance(spec, RoutingExperimentV2Spec):
            raise CommonModelExperimentError("routing experiment spec is invalid")
        if not isinstance(runner, ReviewedProviderBrokerRoutingRunner):
            raise CommonModelExperimentError(
                "reviewed protected provider runner is required"
            )
        if not callable(deadline_supplier):
            raise CommonModelExperimentError("claim deadline is unavailable")
        for method in ("verify_company", "verify_intent", "verify_contact"):
            if not callable(getattr(verifier, method, None)):
                raise CommonModelExperimentError(
                    "reviewed Model verifier is unavailable"
                )
        self._spec = spec
        self._variants = {item.variant_id: item for item in spec.variants}
        if set(registrations) != set(self._variants):
            raise CommonModelExperimentError(
                "Model variant runner registrations are incomplete"
            )
        if any(
            not isinstance(item, ExactModelRunnerRegistration)
            for item in registrations.values()
        ):
            raise CommonModelExperimentError(
                "Model variant runner registration is invalid"
            )
        self._registrations = dict(registrations)
        self._bindings = {item.tool_id: item for item in spec.provider_bindings}
        if len(self._bindings) != len(spec.provider_bindings):
            raise CommonModelExperimentError(
                "provider tool binding is duplicated"
            )
        self._runner = runner.for_execution(
            spec.experiment_hash(),
            spec.experiment_id,
            claim,
            deadline_supplier=deadline_supplier,
        )
        self._verifier = verifier

    def dispatch_provider_action(
        self,
        *,
        action: Mapping[str, Any],
        variant_id: str,
        unit_ref: str,
    ) -> ProtectedModelActionResult:
        variant = self._variants.get(variant_id)
        if variant is None:
            raise CommonModelExperimentError("Model action variant is unknown")
        tool_id = str(action.get("tool_id") or "")
        binding = self._bindings.get(tool_id)
        if not isinstance(binding, ProviderBindingIdentity):
            raise CommonModelExperimentError(
                "Model selected an unavailable provider binding"
            )
        _validate_variant_provider_binding(
            registration=self._registrations[variant_id],
            action=action,
            provider_binding=binding,
            allowed_binding_ids=variant.binding_ids,
        )
        binding_contract = str(action.get("binding_contract_sha256") or "")
        if binding.execution_contract_hash != "sha256:" + binding_contract:
            raise CommonModelExperimentError(
                "Model action binding contract differs"
            )
        normalization_identity = None
        if action.get("action_type") == "normalize_icp":
            registration = self._registrations[variant_id]
            registration.protocol.validate_normalization_action(
                action,
                host_capability_manifest=(
                    registration.host_capability_manifest
                ),
            )
            normalization_identity = (
                registration.protocol_generation.champion_execution[
                    "normalization_action"
                ]
            )
        credit, timeout_ms = _action_budget(
            action,
            normalization_identity=normalization_identity,
        )
        provider_ceiling = self._spec.credit_budget.provider_credit_ceilings.get(
            binding.binding_id
        )
        if provider_ceiling is None or credit > provider_ceiling:
            raise CommonModelExperimentError(
                "Model action exceeds the reviewed provider budget"
            )
        request_hash = str(action.get("request_fingerprint_sha256") or "")
        if len(request_hash) != 64:
            raise CommonModelExperimentError(
                "Model action request fingerprint is invalid"
            )
        authorization = RoutingCallAuthorization(
            experiment_id=self._spec.experiment_id,
            variant_id=variant_id,
            artifact_key=_artifact_hash(variant),
            stage=str(action.get("stage") or ""),
            unit_ref=unit_ref,
            tool_id=tool_id,
            attempt=int(action.get("sequence") or 0),
            request_fingerprint="sha256:" + request_hash,
            remaining_credit_microunits=min(
                credit,
                self._spec.credit_budget.total_credit_microunits,
            ),
            timeout_ceiling_ms=timeout_ms,
            execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
            phase="initial",
        )
        protected = self._runner.dispatch_model_action(
            binding=binding,
            unit_ref=unit_ref,
            request_fingerprint=authorization.request_fingerprint,
            authorization=authorization,
            action=action,
        )
        initial = self._protected_action_result(
            action=action,
            binding=binding,
            protected=protected,
        )
        if not isinstance(initial.replay_ref, Mapping):
            raise CommonModelExperimentError(
                "protected Model action has no durable replay reference"
            )
        # The protected operation persists the exact provider attempt before
        # returning.  Reopen it immediately through the read-only replay path
        # so completion custody is derived only from durable attempt state,
        # never from an uncommitted response in this process.
        replayed = self._runner.replay_model_action(
            binding=binding,
            unit_ref=unit_ref,
            action=action,
            replay_ref=initial.replay_ref,
        )
        durable = self._protected_action_result(
            action=action,
            binding=binding,
            protected=replayed,
        )
        if durable != initial:
            raise CommonModelExperimentError(
                "durable provider attempt differs from dispatch result"
            )
        return durable

    def replay_provider_action(
        self,
        *,
        action: Mapping[str, Any],
        variant_id: str,
        unit_ref: str,
        replay_ref: Mapping[str, Any],
    ) -> ProtectedModelActionResult:
        variant = self._variants.get(variant_id)
        binding = self._bindings.get(str(action.get("tool_id") or ""))
        if variant is None or not isinstance(binding, ProviderBindingIdentity):
            raise CommonModelExperimentError(
                "protected Model replay identity is unknown"
            )
        _validate_variant_provider_binding(
            registration=self._registrations[variant_id],
            action=action,
            provider_binding=binding,
            allowed_binding_ids=variant.binding_ids,
        )
        protected = self._runner.replay_model_action(
            binding=binding,
            unit_ref=unit_ref,
            action=action,
            replay_ref=replay_ref,
        )
        return self._protected_action_result(
            action=action,
            binding=binding,
            protected=protected,
        )

    @staticmethod
    def _protected_action_result(
        *,
        action: Mapping[str, Any],
        binding: ProviderBindingIdentity,
        protected: Mapping[str, Any],
    ) -> ProtectedModelActionResult:
        if not isinstance(protected, Mapping) or set(protected) != {
            "provider_receipt",
            "model_provider_response",
            "model_provider_response_sha256",
            "model_completion_contract_hash",
            "protected_dispatch_job_id",
            "terminal_receipt_hash",
        }:
            raise CommonModelExperimentError(
                "protected provider response is malformed"
            )
        receipt = ProviderReceipt.from_mapping(protected["provider_receipt"])
        response = protected["model_provider_response"]
        if not isinstance(response, Mapping):
            raise CommonModelExperimentError(
                "protected provider response is malformed"
            )
        response_size = len(
            json.dumps(
                response,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        max_response = action.get("max_response_bytes")
        if (
            type(max_response) is not int
            or response_size > max_response
        ):
            raise CommonModelExperimentError(
                "protected provider response exceeds the Model bound"
            )
        if (
            protected["model_provider_response_sha256"]
            != sha256_json(response)
            or protected["model_completion_contract_hash"]
            != sha256_json(build_routing_model_completion_contract_v1(action))
        ):
            raise CommonModelExperimentError(
                "protected provider response commitments differ"
            )
        if receipt.outcome == ProviderOutcome.ADAPTER_FAILURE.value:
            outcome = "failed"
            provider_response = None
        elif receipt.outcome == ProviderOutcome.VERIFIED.value:
            outcome = "succeeded"
            provider_response = {
                **dict(response),
            }
        else:
            outcome = "empty"
            provider_response = {
                **dict(response),
            }
        if receipt.call_count is None:
            raise CommonModelExperimentError(
                "measured provider receipt call count is missing"
            )
        return ProtectedModelActionResult(
            host_result=HostActionResult(
                outcome=outcome,
                reason_code="protected_provider_" + receipt.outcome,
                provider_response=provider_response,
                calls=receipt.call_count,
                cost_credits=receipt.credit_microunits / 1_000_000,
                latency_ms=receipt.latency_ms,
                provider_receipt_ref=receipt.receipt_ref,
                provider_identity_sha256=hashlib.sha256(
                    binding.provider_id.encode("utf-8")
                ).hexdigest(),
            ),
            provider_receipt=receipt,
            replay_ref={
                "schema_version": (
                    "leadpoet.research_lab.protected_model_replay_ref.v1"
                ),
                "protected_dispatch_job_id": protected[
                    "protected_dispatch_job_id"
                ],
                "terminal_receipt_hash": protected[
                    "terminal_receipt_hash"
                ],
                "model_provider_response_sha256": protected[
                    "model_provider_response_sha256"
                ],
                "model_completion_contract_hash": protected[
                    "model_completion_contract_hash"
                ],
            },
        )

    def verify_company_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult:
        return self._verifier.verify_company(action=action, unit_ref=unit_ref)

    def verify_intent_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult:
        return self._verifier.verify_intent(action=action, unit_ref=unit_ref)

    def verify_contact_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult:
        return self._verifier.verify_contact(action=action, unit_ref=unit_ref)


class ModelTransitionRepository(Protocol):
    """Append-only recovery seam keyed by the Model idempotency key."""

    def resolve_run_protocol_generation(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        artifact_key: str,
    ) -> str: ...

    def load_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        idempotency_key: str,
    ) -> Mapping[str, Any] | None: ...

    def append_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        action: Mapping[str, Any],
        continuation: Mapping[str, Any],
        completion: Mapping[str, Any],
        provider_receipt: Mapping[str, Any] | None,
        protocol_generation_sha256: str,
        replay_ref: Mapping[str, Any] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class ExactModelUnitResult:
    variant_id: str
    unit_ref: str
    start_request: Mapping[str, Any]
    terminal_result: Mapping[str, Any]
    provider_receipts: tuple[ProviderReceipt, ...]
    replayed_transition_count: int
    target_verified_qualified_count: int
    protocol_generation_sha256: str


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _generation_bound_continuation_sha256(
    continuation: Mapping[str, Any],
    *,
    protocol_generation_sha256: str,
) -> str:
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        str(protocol_generation_sha256 or ""),
    ):
        raise CommonModelExperimentError(
            "Model protocol generation hash is invalid"
        )
    return _sha256({
        "schema_version": (
            "leadpoet.research_lab.generation_bound_continuation.v1"
        ),
        "protocol_generation_sha256": protocol_generation_sha256,
        "continuation": dict(continuation),
    })


class FencedModelTransitionRepository:
    """Store only redacted hashes under one active V3 claim.

    A process-local repository may return a complete cached transition. This
    production repository never persists provider records. A prior durable
    marker carries only the signed protected-job replay commitments.
    """

    def __init__(self, *, store: Any, claim: Any) -> None:
        if not callable(getattr(store, "append_event", None)) or not callable(
            getattr(store, "load_model_transition_marker", None)
        ) or not callable(
            getattr(store, "load_exact_model_run_registration", None)
        ):
            raise CommonModelExperimentError(
                "durable Model transition store is unavailable"
            )
        self._store = store
        self._claim = claim

    def resolve_run_protocol_generation(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        artifact_key: str,
    ) -> str:
        """Resolve the exact durable run tuple before marker or OCI access."""

        try:
            registration = self._store.load_exact_model_run_registration(
                experiment_hash=experiment_hash
            )
        except Exception as exc:
            raise CommonModelExperimentRecoveryError(
                "exact Model run registration could not be resolved"
            ) from exc
        if not isinstance(registration, Mapping):
            raise CommonModelExperimentRecoveryError(
                "exact Model run registration is missing"
            )
        artifact_keys = registration.get("artifact_keys")
        generations = registration.get("protocol_generations")
        generation = (
            generations.get(variant_id)
            if isinstance(generations, Mapping)
            else None
        )
        if (
            not isinstance(artifact_keys, Mapping)
            or artifact_keys.get(variant_id) != artifact_key
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(generation or "")
            )
        ):
            raise CommonModelExperimentRecoveryError(
                "exact Model run artifact or generation differs"
            )
        return str(generation)

    def load_model_transition(self, **identity: Any) -> Mapping[str, Any] | None:
        return self._store.load_model_transition_marker(**identity)

    def append_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        action: Mapping[str, Any],
        continuation: Mapping[str, Any],
        completion: Mapping[str, Any],
        provider_receipt: Mapping[str, Any] | None,
        protocol_generation_sha256: str,
        replay_ref: Mapping[str, Any] | None = None,
    ) -> None:
        provider_response = completion.get("provider_response")
        if (provider_receipt is None) != (replay_ref is None):
            raise CommonModelExperimentError(
                "protected Model replay and provider receipt must be paired"
            )
        if replay_ref is None:
            replay_values = {
                "protected_dispatch_job_id": None,
                "terminal_receipt_hash": None,
                "model_completion_contract_hash": None,
            }
        else:
            expected_replay_fields = {
                "schema_version",
                "protected_dispatch_job_id",
                "terminal_receipt_hash",
                "model_provider_response_sha256",
                "model_completion_contract_hash",
            }
            if (
                not isinstance(replay_ref, Mapping)
                or set(replay_ref) != expected_replay_fields
                or replay_ref.get("schema_version")
                != "leadpoet.research_lab.protected_model_replay_ref.v1"
                or replay_ref.get("model_provider_response_sha256")
                != _sha256(provider_response)
            ):
                raise CommonModelExperimentError(
                    "protected Model replay reference differs"
                )
            replay_values = {
                "protected_dispatch_job_id": replay_ref[
                    "protected_dispatch_job_id"
                ],
                "terminal_receipt_hash": replay_ref[
                    "terminal_receipt_hash"
                ],
                "model_completion_contract_hash": replay_ref[
                    "model_completion_contract_hash"
                ],
            }
        result = self._store.append_event(
            experiment_hash=experiment_hash,
            event_type="model_transition_completed",
            event_doc={
                "event_schema_version": (
                    "leadpoet.research_lab.model_transition.v1"
                ),
                "variant_id": variant_id,
                "unit_ref": unit_ref,
                "idempotency_key": action.get("idempotency_key"),
                "action_sha256": action.get("action_sha256"),
                "continuation_sha256": _generation_bound_continuation_sha256(
                    continuation,
                    protocol_generation_sha256=protocol_generation_sha256,
                ),
                "completion_sha256": completion.get("completion_sha256"),
                "provider_response_sha256": _sha256(provider_response),
                "provider_receipt": (
                    None
                    if provider_receipt is None
                    else dict(provider_receipt)
                ),
                **replay_values,
            },
            claim=self._claim,
        )
        if not isinstance(result, Mapping):
            raise CommonModelExperimentError(
                "durable Model transition result is invalid"
            )


_PROVIDER_ACTION_TYPES = frozenset(
    {
        "normalize_icp",
        "execute_candidate_tool",
        "execute_intent_tool",
        "execute_contact_tool",
    }
)


def _validate_variant_provider_binding(
    *,
    registration: ExactModelRunnerRegistration,
    action: Mapping[str, Any],
    provider_binding: ProviderBindingIdentity,
    allowed_binding_ids: tuple[str, ...],
) -> None:
    """Require the active variant to authorize this exact provider tool.

    The global routing catalog is not sufficient: a challenger must not be
    able to select a host binding that was only admitted for another variant.
    This check runs before the protected dispatch runner is called.
    """

    if provider_binding.binding_id not in allowed_binding_ids:
        raise CommonModelExperimentError(
            "Model variant did not declare the selected provider binding"
        )
    manifest = registration.host_capability_manifest
    raw_bindings = manifest.get("bindings") if isinstance(manifest, Mapping) else None
    if not isinstance(raw_bindings, (list, tuple)):
        raise CommonModelExperimentError(
            "Model variant host capability bindings are invalid"
        )
    action_type = str(action.get("action_type") or "")
    tool_id = str(action.get("tool_id") or "").strip()
    requested_hash = str(action.get("binding_contract_sha256") or "")
    matches = [
        item
        for item in raw_bindings
        if isinstance(item, Mapping)
        and item.get("action_type") == action_type
        and str(item.get("tool_id") or "").strip() == tool_id
    ]
    if len(matches) != 1 or matches[0].get("available") is not True:
        raise CommonModelExperimentError(
            "Model variant did not authorize the selected provider binding"
        )
    manifest_hash = str(matches[0].get("binding_contract_sha256") or "")
    if requested_hash != manifest_hash or (
        provider_binding.execution_contract_hash != "sha256:" + requested_hash
    ):
        raise CommonModelExperimentError(
            "Model variant provider binding contract differs"
        )


def _completion_from_transition(
    transition: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    protocol_generation_sha256: str,
) -> tuple[Mapping[str, Any], ProviderReceipt | None]:
    if set(transition) != {
        "action",
        "continuation",
        "completion",
        "provider_receipt",
        "protocol_generation_sha256",
    }:
        raise CommonModelExperimentError("stored Model transition is malformed")
    if transition.get("protocol_generation_sha256") != (
        protocol_generation_sha256
    ):
        raise CommonModelExperimentError(
            "stored Model transition protocol generation differs"
        )
    if transition.get("action") != dict(action):
        raise CommonModelExperimentError("stored Model transition action differs")
    completion = transition.get("completion")
    if not isinstance(completion, Mapping):
        raise CommonModelExperimentError("stored Model completion is malformed")
    raw_receipt = transition.get("provider_receipt")
    receipt = None
    if raw_receipt is not None:
        if not isinstance(raw_receipt, Mapping):
            raise CommonModelExperimentError(
                "stored protected provider receipt is malformed"
            )
        receipt = ProviderReceipt.from_mapping(raw_receipt)
    return dict(completion), receipt


def _protected_replay_ref_from_marker(
    marker: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    continuation: Mapping[str, Any],
    protocol_generation_sha256: str,
) -> Mapping[str, Any] | None:
    """Validate one SQL marker before any protected replay or verifier work."""

    generation_bound = _generation_bound_continuation_sha256(
        continuation,
        protocol_generation_sha256=protocol_generation_sha256,
    )
    stored_continuation_hash = marker.get("continuation_sha256")
    if (
        marker.get("action_sha256") != action.get("action_sha256")
        or marker.get("idempotency_key") != action.get("idempotency_key")
        or stored_continuation_hash != generation_bound
    ):
        raise CommonModelExperimentError(
            "durable Model transition identity differs"
        )
    raw_receipt = marker.get("provider_receipt")
    replay_fields = {
        "protected_dispatch_job_id": marker.get(
            "protected_dispatch_job_id"
        ),
        "terminal_receipt_hash": marker.get("terminal_receipt_hash"),
        "model_completion_contract_hash": marker.get(
            "model_completion_contract_hash"
        ),
    }
    if raw_receipt is None:
        if any(value is not None for value in replay_fields.values()):
            raise CommonModelExperimentError(
                "durable verifier transition has provider replay state"
            )
        return None
    if not isinstance(raw_receipt, Mapping) or any(
        value is None for value in replay_fields.values()
    ):
        raise CommonModelExperimentError(
            "durable provider transition replay state is incomplete"
        )
    return {
        "schema_version": (
            "leadpoet.research_lab.protected_model_replay_ref.v1"
        ),
        **replay_fields,
        "model_provider_response_sha256": marker.get(
            "provider_response_sha256"
        ),
    }


def _bind_durable_provider_result(
    *,
    protocol: Any,
    action: Mapping[str, Any],
    protected: ProtectedModelActionResult,
) -> HostActionResult:
    """Build custody only after protected dispatch has passed readback."""

    host_result = protected.host_result
    receipt = protected.provider_receipt
    if receipt is None or not isinstance(host_result, HostActionResult):
        raise CommonModelExperimentError(
            "durable provider result is incomplete"
        )
    ingestion = _validated_durable_provider_ingestion(
        protocol=protocol,
        action=action,
        protected=protected,
    )
    try:
        binding = protocol.build_provider_receipt_binding(
            action, host_result
        )
    except ModelRunnerHostError as exc:
        raise CommonModelExperimentError(
            "artifact provider receipt custody differs from durable attempt"
        ) from exc
    if (
        not isinstance(binding, Mapping)
        or binding.get("provider_receipt_ref") != receipt.receipt_ref
        or binding.get("provider_receipt_ref")
        != host_result.provider_receipt_ref
        or binding.get("provider_identity_sha256")
        != host_result.provider_identity_sha256
        or (
            ingestion is not None
            and binding.get("provider_response_sha256")
            != ingestion.get("parsed_response_sha256")
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(binding.get("receipt_sha256") or "")
        )
        or (
            host_result.provider_receipt_sha256 is not None
            and host_result.provider_receipt_sha256
            != binding.get("receipt_sha256")
        )
    ):
        raise CommonModelExperimentError(
            "artifact provider receipt custody differs from durable attempt"
        )
    return replace(
        host_result,
        provider_receipt_sha256=str(binding["receipt_sha256"]),
        provider_identity_sha256=str(binding["provider_identity_sha256"]),
    )


def _validated_durable_provider_ingestion(
    *,
    protocol: Any,
    action: Mapping[str, Any],
    protected: ProtectedModelActionResult,
) -> Mapping[str, Any] | None:
    """Reingest raw custody bytes; never reinterpret model-owned content."""

    generation = getattr(protocol, "protocol_generation", None)
    requires_ingestion = bool(
        getattr(generation, "supports_provider_response_ingestion", False)
    )
    host_response = protected.host_result.provider_response
    persisted = protected.model_provider_response_ingestion
    if not requires_ingestion:
        if persisted is not None:
            raise CommonModelExperimentError(
                "legacy provider result carries unsupported ingestion custody"
            )
        return None
    if host_response is None:
        if persisted is not None:
            raise CommonModelExperimentError(
                "empty provider result carries ingestion custody"
            )
        return None
    if not isinstance(host_response, Mapping) or not isinstance(
        persisted, Mapping
    ):
        raise CommonModelExperimentError(
            "durable provider response ingestion is incomplete"
        )
    ingest = getattr(protocol, "ingest_provider_response", None)
    if not callable(ingest):
        raise CommonModelExperimentError(
            "artifact provider response ingestor is unavailable"
        )
    try:
        replayed = ingest(action, host_response)
    except Exception as exc:
        raise CommonModelExperimentError(
            "artifact provider response ingestion failed"
        ) from exc
    if not isinstance(replayed, Mapping) or dict(replayed) != dict(persisted):
        raise CommonModelExperimentError(
            "durable provider response ingestion differs from replay"
        )
    return dict(persisted)


def _validate_completion_provider_ingestion(
    *,
    protected: ProtectedModelActionResult,
    completion: Mapping[str, Any],
) -> None:
    """Join artifact completion to its separately custodied ingestion receipt."""

    ingestion = protected.model_provider_response_ingestion
    if ingestion is None:
        return
    parsed_response = ingestion.get("parsed_response")
    parsed_sha256 = str(
        ingestion.get("parsed_response_sha256") or ""
    ).removeprefix("sha256:")
    completion_sha256 = str(
        completion.get("provider_response_sha256") or ""
    ).removeprefix("sha256:")
    if (
        not isinstance(parsed_response, Mapping)
        or completion.get("provider_response") != parsed_response
        or completion_sha256 != parsed_sha256
        or _sha256(parsed_response).removeprefix("sha256:")
        != parsed_sha256
    ):
        raise CommonModelExperimentError(
            "Model completion differs from provider response ingestion"
        )


class ExactModelExperimentCoordinator:
    """Advance one registered PR274 artifact with durable Lab transitions."""

    def __init__(
        self,
        *,
        experiment_hash: str,
        registration: ExactModelRunnerRegistration,
        dispatcher: ExactModelActionDispatcher,
        transitions: ModelTransitionRepository,
        max_actions: int = 10_000,
    ) -> None:
        if not isinstance(registration, ExactModelRunnerRegistration):
            raise CommonModelExperimentError(
                "exact Model registration is required"
            )
        if isinstance(max_actions, bool) or not 1 <= max_actions <= 10_000:
            raise CommonModelExperimentError("Model action limit is invalid")
        self._experiment_hash = str(experiment_hash)
        self._registration = registration
        self._dispatcher = dispatcher
        self._transitions = transitions
        self._max_actions = max_actions

    def run_unit(
        self,
        *,
        variant_id: str,
        unit_ref: str,
        model_input: Mapping[str, Any],
        execution_mode: str,
        target_count: int,
        evaluated_on: str,
    ) -> ExactModelUnitResult:
        pinned_generation_sha256 = (
            self._transitions.resolve_run_protocol_generation(
                experiment_hash=self._experiment_hash,
                variant_id=variant_id,
                artifact_key=self._registration.key,
            )
        )
        generation = self._registration.protocol_generation
        generation_sha256 = generation.protocol_generation_sha256
        if generation_sha256 != pinned_generation_sha256:
            raise CommonModelExperimentRecoveryError(
                "active Model protocol generation differs from run registration"
            )
        self._registration.preflight(execution_mode=execution_mode)
        protocol = self._registration.protocol
        start = protocol.build_start(
            input=model_input,
            execution_mode=execution_mode,
            target_count=target_count,
            evaluated_on=evaluated_on,
            host_capability_manifest=(
                self._registration.host_capability_manifest
            ),
        )
        state = protocol.advance(start, continuation=None, completion=None)
        state = protocol.validate_result(state, start_request=start)
        if generation.supports_raw_icp and execution_mode != "intent_refresh":
            if model_input.get("kind") != "raw_icp":
                raise CommonModelExperimentError(
                    "v3 full execution requires an artifact-owned raw ICP"
                )
            # Normalization is conditional and entirely model-owned.  An
            # explicit raw ICP may normalize deterministically and advance
            # directly to acquisition, while an ambiguous ICP emits the
            # artifact-declared provider action.  Validate that action when it
            # exists, but never require the host to predict which path the
            # artifact will select.
            initial_action = state.get("action")
            if (
                isinstance(initial_action, Mapping)
                and initial_action.get("action_type") == "normalize_icp"
            ):
                protocol.validate_normalization_action(
                    initial_action,
                    host_capability_manifest=(
                        self._registration.host_capability_manifest
                    ),
                )
        receipts: list[ProviderReceipt] = []
        replayed = 0
        action_count = 0
        while state.get("status") == "action_required":
            if action_count >= self._max_actions:
                raise CommonModelExperimentError("Model action limit was reached")
            action = state.get("action")
            continuation = state.get("continuation")
            if not isinstance(action, Mapping) or not isinstance(
                continuation, Mapping
            ):
                raise CommonModelExperimentError("Model action state is malformed")
            idempotency_key = str(action.get("idempotency_key") or "")
            if len(idempotency_key) != 64:
                raise CommonModelExperimentError(
                    "Model action idempotency key is invalid"
                )
            stored = self._transitions.load_model_transition(
                experiment_hash=self._experiment_hash,
                variant_id=variant_id,
                unit_ref=unit_ref,
                idempotency_key=idempotency_key,
            )
            if stored is not None:
                if "action" in stored:
                    completion, receipt = _completion_from_transition(
                        stored,
                        action=action,
                        protocol_generation_sha256=generation_sha256,
                    )
                    if stored.get("continuation") != dict(continuation):
                        raise CommonModelExperimentError(
                            "stored Model continuation differs"
                        )
                else:
                    replay_ref = _protected_replay_ref_from_marker(
                        stored,
                        action=action,
                        continuation=continuation,
                        protocol_generation_sha256=generation_sha256,
                    )
                    action_type = str(action.get("action_type") or "")
                    receipt = None
                    if action_type in _PROVIDER_ACTION_TYPES:
                        if replay_ref is None:
                            raise CommonModelExperimentError(
                                "durable provider transition replay is missing"
                            )
                        protected = self._dispatcher.replay_provider_action(
                            action=action,
                            variant_id=variant_id,
                            unit_ref=unit_ref,
                            replay_ref=replay_ref,
                        )
                        if not isinstance(
                            protected, ProtectedModelActionResult
                        ) or protected.provider_receipt is None:
                            raise CommonModelExperimentError(
                                "protected provider replay result is invalid"
                            )
                        host_result = protected.host_result
                        receipt = protected.provider_receipt
                    elif action_type == "verify_company":
                        host_result = self._dispatcher.verify_company_action(
                            action=action, unit_ref=unit_ref
                        )
                    elif action_type == "verify_intent":
                        host_result = self._dispatcher.verify_intent_action(
                            action=action, unit_ref=unit_ref
                        )
                    elif action_type == "verify_contact":
                        host_result = self._dispatcher.verify_contact_action(
                            action=action, unit_ref=unit_ref
                        )
                    else:
                        raise CommonModelExperimentError(
                            "durable Model transition action is unsupported"
                        )
                    if not isinstance(host_result, HostActionResult):
                        raise CommonModelExperimentError(
                            "replayed host action result is invalid"
                        )
                    if action_type in _PROVIDER_ACTION_TYPES and (
                        generation.supports_provider_receipt_binding
                    ):
                        host_result = _bind_durable_provider_result(
                            protocol=protocol,
                            action=action,
                            protected=protected,
                        )
                    completion = protocol.build_completion(action, host_result)
                    if action_type in _PROVIDER_ACTION_TYPES:
                        _validate_completion_provider_ingestion(
                            protected=protected,
                            completion=completion,
                        )
                    if (
                        completion.get("completion_sha256")
                        != stored.get("completion_sha256")
                        or _sha256(completion.get("provider_response"))
                        != stored.get("provider_response_sha256")
                        or (
                            None
                            if receipt is None
                            else receipt.to_dict()
                        )
                        != stored.get("provider_receipt")
                    ):
                        raise CommonModelExperimentError(
                            "replayed Model completion differs from durable marker"
                        )
                replayed += 1
            else:
                action_type = str(action.get("action_type") or "")
                receipt = None
                if action_type in _PROVIDER_ACTION_TYPES:
                    protected = self._dispatcher.dispatch_provider_action(
                        action=action,
                        variant_id=variant_id,
                        unit_ref=unit_ref,
                    )
                    if not isinstance(protected, ProtectedModelActionResult):
                        raise CommonModelExperimentError(
                            "protected provider result is invalid"
                        )
                    host_result = protected.host_result
                    receipt = protected.provider_receipt
                    if receipt is None:
                        raise CommonModelExperimentError(
                            "protected provider receipt is missing"
                        )
                elif action_type == "verify_company":
                    host_result = self._dispatcher.verify_company_action(
                        action=action, unit_ref=unit_ref
                    )
                elif action_type == "verify_intent":
                    host_result = self._dispatcher.verify_intent_action(
                        action=action, unit_ref=unit_ref
                    )
                elif action_type == "verify_contact":
                    host_result = self._dispatcher.verify_contact_action(
                        action=action, unit_ref=unit_ref
                    )
                else:
                    raise CommonModelExperimentError(
                        "Model selected an unsupported action type"
                    )
                if not isinstance(host_result, HostActionResult):
                    raise CommonModelExperimentError("host action result is invalid")
                if action_type in _PROVIDER_ACTION_TYPES and (
                    generation.supports_provider_receipt_binding
                ):
                    host_result = _bind_durable_provider_result(
                        protocol=protocol,
                        action=action,
                        protected=protected,
                    )
                completion = protocol.build_completion(action, host_result)
                if action_type in _PROVIDER_ACTION_TYPES:
                    _validate_completion_provider_ingestion(
                        protected=protected,
                        completion=completion,
                    )
                self._transitions.append_model_transition(
                    experiment_hash=self._experiment_hash,
                    variant_id=variant_id,
                    unit_ref=unit_ref,
                    action=action,
                    continuation=continuation,
                    completion=completion,
                    provider_receipt=(
                        None if receipt is None else receipt.to_dict()
                    ),
                    protocol_generation_sha256=generation_sha256,
                    replay_ref=(
                        protected.replay_ref
                        if action_type in _PROVIDER_ACTION_TYPES
                        else None
                    ),
                )
            if receipt is not None:
                receipts.append(receipt)
            state = protocol.advance(
                start,
                continuation=continuation,
                completion=completion,
            )
            state = protocol.validate_result(state, start_request=start)
            action_count += 1
        if state.get("status") != "completed":
            raise CommonModelExperimentError("Model terminal state is invalid")
        return ExactModelUnitResult(
            variant_id=variant_id,
            unit_ref=unit_ref,
            start_request=dict(start),
            terminal_result=dict(state),
            provider_receipts=tuple(receipts),
            replayed_transition_count=replayed,
            target_verified_qualified_count=target_count,
            protocol_generation_sha256=generation_sha256,
        )


__all__ = [
    "CommonModelExperimentError",
    "CommonModelExperimentRecoveryError",
    "ExactModelActionDispatcher",
    "ExactModelExperimentCoordinator",
    "ExactModelUnitResult",
    "FencedModelTransitionRepository",
    "ModelTransitionRepository",
    "ProtectedModelActionResult",
    "ReviewedModelVerificationAuthority",
    "ReviewedProtectedModelActionDispatcher",
]
