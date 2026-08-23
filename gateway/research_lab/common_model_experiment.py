"""PR93 control-plane loop for PR274's exact Model runner protocol.

The Lab owns admission, claims, protected dispatch, billing, persistence, and
evaluation. It does not compile a route or choose a provider. Every action is
emitted and validated by the exact immutable Model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from research_lab.common_model_runner_host import HostActionResult
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


MODEL_TRANSITION_SCHEMA_VERSION = (
    "leadpoet.research_lab.model_transition.v2"
)
_ARTIFACT_KEY_RE = re.compile(
    r"[0-9a-f]{40}:"
    r"(?:sha256:)?[0-9a-f]{64}:"
    r"(?:sha256:)?[0-9a-f]{64}"
)


def _require_artifact_key(value: Any) -> str:
    artifact_key = str(value or "").strip().lower()
    if not _ARTIFACT_KEY_RE.fullmatch(artifact_key):
        raise CommonModelExperimentError(
            "exact Model transition artifact identity is invalid"
        )
    return artifact_key


@dataclass(frozen=True)
class ProtectedModelActionResult:
    """One host result and its optional protected provider receipt."""

    host_result: HostActionResult
    provider_receipt: ProviderReceipt | None = None
    replay_ref: Mapping[str, Any] | None = None


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


def _action_budget(action: Mapping[str, Any]) -> tuple[int, int]:
    arguments = action.get("arguments")
    step = arguments.get("step") if isinstance(arguments, Mapping) else None
    if not isinstance(step, Mapping):
        raise CommonModelExperimentError("Model provider action budget is missing")
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
        credit, timeout_ms = _action_budget(action)
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
        return self._protected_action_result(
            action=action,
            protected=protected,
        )

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
            protected=protected,
        )

    @staticmethod
    def _protected_action_result(
        *,
        action: Mapping[str, Any],
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
    """Append-only recovery seam keyed by logical and artifact identity."""

    def load_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        idempotency_key: str,
        artifact_key: str,
    ) -> Mapping[str, Any] | None: ...

    def append_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        artifact_key: str,
        action: Mapping[str, Any],
        continuation: Mapping[str, Any],
        completion: Mapping[str, Any],
        provider_receipt: Mapping[str, Any] | None,
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


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class FencedModelTransitionRepository:
    """Store only redacted hashes under one active V3 claim.

    A process-local repository may return a complete cached transition. This
    production repository never persists provider records. A prior durable
    marker carries only the signed protected-job replay commitments.
    """

    def __init__(self, *, store: Any, claim: Any) -> None:
        if not callable(getattr(store, "append_event", None)) or not callable(
            getattr(store, "load_model_transition_marker", None)
        ):
            raise CommonModelExperimentError(
                "durable Model transition store is unavailable"
            )
        self._store = store
        self._claim = claim

    def load_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        idempotency_key: str,
        artifact_key: str,
    ) -> Mapping[str, Any] | None:
        expected_artifact_key = _require_artifact_key(artifact_key)
        marker = self._store.load_model_transition_marker(
            experiment_hash=experiment_hash,
            variant_id=variant_id,
            unit_ref=unit_ref,
            idempotency_key=idempotency_key,
            artifact_key=expected_artifact_key,
        )
        if marker is None:
            return None
        if (
            not isinstance(marker, Mapping)
            or marker.get("event_schema_version")
            != MODEL_TRANSITION_SCHEMA_VERSION
            or marker.get("artifact_key") != expected_artifact_key
        ):
            raise CommonModelExperimentRecoveryError(
                "durable Model transition artifact identity differs"
            )
        return dict(marker)

    def append_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        artifact_key: str,
        action: Mapping[str, Any],
        continuation: Mapping[str, Any],
        completion: Mapping[str, Any],
        provider_receipt: Mapping[str, Any] | None,
        replay_ref: Mapping[str, Any] | None = None,
    ) -> None:
        expected_artifact_key = _require_artifact_key(artifact_key)
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
                    MODEL_TRANSITION_SCHEMA_VERSION
                ),
                "variant_id": variant_id,
                "unit_ref": unit_ref,
                "artifact_key": expected_artifact_key,
                "idempotency_key": action.get("idempotency_key"),
                "action_sha256": action.get("action_sha256"),
                "continuation_sha256": _sha256(continuation),
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
) -> tuple[Mapping[str, Any], ProviderReceipt | None]:
    if set(transition) != {"action", "continuation", "completion", "provider_receipt"}:
        raise CommonModelExperimentError("stored Model transition is malformed")
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
) -> Mapping[str, Any] | None:
    """Validate one SQL marker before any protected replay or verifier work."""

    if (
        marker.get("action_sha256") != action.get("action_sha256")
        or marker.get("idempotency_key") != action.get("idempotency_key")
        or marker.get("continuation_sha256") != _sha256(continuation)
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
        self._artifact_key = _require_artifact_key(registration.key)
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
        self._registration.preflight()
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
                artifact_key=self._artifact_key,
            )
            if stored is not None:
                if "action" in stored:
                    completion, receipt = _completion_from_transition(
                        stored, action=action
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
                    completion = protocol.build_completion(
                        action, host_result
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
                completion = protocol.build_completion(action, host_result)
                self._transitions.append_model_transition(
                    experiment_hash=self._experiment_hash,
                    variant_id=variant_id,
                    unit_ref=unit_ref,
                    artifact_key=self._artifact_key,
                    action=action,
                    continuation=continuation,
                    completion=completion,
                    provider_receipt=(
                        None if receipt is None else receipt.to_dict()
                    ),
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
