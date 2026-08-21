"""Consumer-neutral host loop for the model-owned continuation protocol.

This module owns no routing, policy, parsing, or qualification behavior.  It
dispatches only the exact action emitted by the immutable model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence


HOST_EXECUTION_RECEIPT_SCHEMA_VERSION = "model-runner-host-receipt:v1"
_ACTION_TYPES = frozenset({
    "execute_candidate_tool",
    "verify_company",
    "execute_intent_tool",
    "verify_intent",
    "execute_contact_tool",
    "verify_contact",
})
_OUTCOMES = frozenset({
    "succeeded",
    "empty",
    "unavailable",
    "timeout",
    "failed",
})


class ModelRunnerHostError(RuntimeError):
    """A model action cannot be dispatched without changing its contract."""


@dataclass(frozen=True)
class HostActionResult:
    """One host call result before model-owned parsing and qualification."""

    outcome: str
    reason_code: str
    provider_response: Mapping[str, Any] | None
    calls: int
    cost_credits: float
    latency_ms: float
    provider_request_id: str | None = None


@dataclass(frozen=True)
class HostActionBinding:
    """Credentialed execution for one exact model-owned action and tool ID."""

    action_type: str
    tool_id: str
    binding_contract_sha256: str
    dispatch: Callable[[Mapping[str, Any]], HostActionResult]


class ModelRunnerProtocol(Protocol):
    """Artifact transport used by either an in-process or OCI model runner."""

    def advance(
        self,
        start_request: Mapping[str, Any],
        *,
        continuation: Mapping[str, Any] | None,
        completion: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...

    def build_completion(
        self,
        action: Mapping[str, Any],
        result: HostActionResult,
    ) -> Mapping[str, Any]: ...


PersistTransition = Callable[..., None]
LoadCompletion = Callable[[str], Optional[Mapping[str, Any]]]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding_key(value: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(value.get("action_type") or ""),
        str(value.get("tool_id") or "").strip().casefold(),
    )


def _available_manifest_bindings(
    start_request: Mapping[str, Any],
) -> dict[tuple[str, str], str]:
    manifest = start_request.get("host_capability_manifest")
    if not isinstance(manifest, Mapping):
        raise ModelRunnerHostError("start request has no host capability manifest")
    raw_bindings = manifest.get("bindings")
    if not isinstance(raw_bindings, Sequence) or isinstance(
        raw_bindings,
        (str, bytes, bytearray),
    ):
        raise ModelRunnerHostError("host capability bindings are invalid")
    available: dict[tuple[str, str], str] = {}
    for item in raw_bindings:
        if not isinstance(item, Mapping):
            raise ModelRunnerHostError("host capability binding is invalid")
        if item.get("available") is not True:
            continue
        key = _binding_key(item)
        contract_hash = str(item.get("binding_contract_sha256") or "")
        if key in available:
            raise ModelRunnerHostError("host capability binding is duplicated")
        available[key] = contract_hash
    if not available:
        raise ModelRunnerHostError("host capability manifest has no bindings")
    return available


def _binding_index(
    bindings: Sequence[HostActionBinding],
) -> dict[tuple[str, str], HostActionBinding]:
    indexed: dict[tuple[str, str], HostActionBinding] = {}
    for binding in bindings:
        key = (binding.action_type, binding.tool_id.strip().casefold())
        if key in indexed:
            raise ModelRunnerHostError("host action binding is duplicated")
        if binding.action_type not in _ACTION_TYPES:
            raise ModelRunnerHostError("host action binding type is invalid")
        if not callable(binding.dispatch):
            raise ModelRunnerHostError("host action binding is not dispatchable")
        indexed[key] = binding
    return indexed


def _host_receipt(
    *,
    consumer_id: str,
    action: Mapping[str, Any],
    completion: Mapping[str, Any],
    provider_request_id: str | None,
    replayed: bool,
) -> dict[str, Any]:
    request_id_hash = (
        _canonical_sha256(provider_request_id)
        if provider_request_id
        else None
    )
    payload = {
        "schema_version": HOST_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "consumer_id": consumer_id,
        "action_sha256": str(action.get("action_sha256") or ""),
        "idempotency_key": str(action.get("idempotency_key") or ""),
        "binding_contract_sha256": str(
            action.get("binding_contract_sha256") or ""
        ),
        "provider_request_id_sha256": request_id_hash,
        "provider_response_sha256": str(
            completion.get("provider_response_sha256") or ""
        ),
        "completion_sha256": str(completion.get("completion_sha256") or ""),
        "outcome": str(completion.get("outcome") or ""),
        "calls": completion.get("calls"),
        "cost_credits": completion.get("cost_credits"),
        "latency_ms": completion.get("latency_ms"),
        "replayed": replayed,
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


class CommonModelRunnerHost:
    """Run the exact artifact continuation until it reaches a terminal result."""

    def __init__(
        self,
        *,
        consumer_id: str,
        protocol: ModelRunnerProtocol,
        bindings: Sequence[HostActionBinding],
        persist_transition: PersistTransition,
        load_completion: LoadCompletion | None = None,
        max_actions: int = 10_000,
    ) -> None:
        if not consumer_id or len(consumer_id) > 80:
            raise ModelRunnerHostError("consumer_id is invalid")
        if not callable(persist_transition):
            raise ModelRunnerHostError("durable transition sink is required")
        if isinstance(max_actions, bool) or not 1 <= max_actions <= 10_000:
            raise ModelRunnerHostError("max_actions must be between 1 and 10000")
        self._consumer_id = consumer_id
        self._protocol = protocol
        self._bindings = _binding_index(bindings)
        self._persist_transition = persist_transition
        self._load_completion = load_completion
        self._max_actions = max_actions

    def run(
        self,
        start_request: Mapping[str, Any],
        *,
        continuation: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        expected = _available_manifest_bindings(start_request)
        actual = {
            key: binding.binding_contract_sha256
            for key, binding in self._bindings.items()
        }
        if actual != expected:
            raise ModelRunnerHostError(
                "executable bindings differ from the capability manifest"
            )

        state = self._protocol.advance(
            start_request,
            continuation=continuation,
            completion=None,
        )
        completed_actions = 0
        while state.get("status") == "action_required":
            if completed_actions >= self._max_actions:
                raise ModelRunnerHostError("model action limit was reached")
            action = state.get("action")
            next_continuation = state.get("continuation")
            if not isinstance(action, Mapping) or not isinstance(
                next_continuation,
                Mapping,
            ):
                raise ModelRunnerHostError("model action response is invalid")
            action_type = str(action.get("action_type") or "")
            tool_id = str(action.get("tool_id") or "").strip().casefold()
            binding = self._bindings.get((action_type, tool_id))
            if binding is None:
                raise ModelRunnerHostError("model selected an unbound host action")
            if (
                str(action.get("binding_contract_sha256") or "")
                != binding.binding_contract_sha256
            ):
                raise ModelRunnerHostError("model action binding hash differs")
            idempotency_key = str(action.get("idempotency_key") or "")
            if not idempotency_key:
                raise ModelRunnerHostError("model action has no idempotency key")

            cached = (
                self._load_completion(idempotency_key)
                if self._load_completion is not None
                else None
            )
            if cached is None:
                host_result = binding.dispatch(action)
                if not isinstance(host_result, HostActionResult):
                    raise ModelRunnerHostError(
                        "host binding must return HostActionResult"
                    )
                if host_result.outcome not in _OUTCOMES:
                    raise ModelRunnerHostError("host binding outcome is invalid")
                completion = self._protocol.build_completion(
                    action,
                    host_result,
                )
                provider_request_id = host_result.provider_request_id
                replayed = False
            else:
                completion = cached
                provider_request_id = None
                replayed = True

            advanced = self._protocol.advance(
                start_request,
                continuation=next_continuation,
                completion=completion,
            )
            advanced_continuation = advanced.get("continuation")
            if not isinstance(advanced_continuation, Mapping):
                raise ModelRunnerHostError("model continuation is missing")
            receipt = _host_receipt(
                consumer_id=self._consumer_id,
                action=action,
                completion=completion,
                provider_request_id=provider_request_id,
                replayed=replayed,
            )
            self._persist_transition(
                idempotency_key=idempotency_key,
                action=dict(action),
                completion=dict(completion),
                continuation=dict(advanced_continuation),
                host_receipt=receipt,
            )
            state = advanced
            completed_actions += 1

        if state.get("status") != "completed":
            raise ModelRunnerHostError("model runner returned an invalid status")
        if not isinstance(state.get("result"), Mapping) or not isinstance(
            state.get("model_receipt"),
            Mapping,
        ):
            raise ModelRunnerHostError("model runner terminal result is invalid")
        return state


__all__ = [
    "CommonModelRunnerHost",
    "HOST_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "HostActionBinding",
    "HostActionResult",
    "ModelRunnerHostError",
    "ModelRunnerProtocol",
]
