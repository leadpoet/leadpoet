"""Consumer-neutral host loop for the model-owned continuation protocol.

This module owns no routing, policy, parsing, or qualification behavior.  It
dispatches only the exact action emitted by the immutable model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence


HOST_EXECUTION_RECEIPT_SCHEMA_VERSION = "model-runner-host-receipt:v2"
_ACTION_TYPES = frozenset({
    "normalize_icp",
    "execute_candidate_tool",
    "verify_company",
    "execute_intent_tool",
    "verify_intent",
    "execute_contact_tool",
    "verify_contact",
})
_PROVIDER_ACTION_TYPES = frozenset({
    "normalize_icp",
    "execute_candidate_tool",
    "execute_intent_tool",
    "execute_contact_tool",
})
_OUTCOMES = frozenset({
    "succeeded",
    "empty",
    "unavailable",
    "timeout",
    "failed",
})
_PROVIDER_RECEIPT_RE = re.compile(r"provider_receipt:[0-9a-f]{16,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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
    provider_receipt_ref: str | None = None
    provider_receipt_sha256: str | None = None
    provider_identity_sha256: str | None = None


@dataclass(frozen=True)
class HostActionBinding:
    """Credentialed execution for one exact model-owned action and tool ID."""

    action_type: str
    tool_id: str
    binding_contract_sha256: str
    dispatch: Callable[[Mapping[str, Any]], HostActionResult]


@dataclass(frozen=True)
class ProviderReceiptCustodyRecord:
    """Sanitized hashes resolved from one server-owned durable receipt."""

    provider_receipt_ref: str
    provider_receipt_sha256: str
    provider_identity_sha256: str


class ProviderReceiptCustody(Protocol):
    """Host-owned lookup for already-persisted provider receipt bindings."""

    durable: bool

    def resolve_provider_receipt(
        self,
        provider_receipt_ref: str,
    ) -> ProviderReceiptCustodyRecord | None: ...


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

    def build_provider_receipt_binding(
        self,
        action: Mapping[str, Any],
        result: HostActionResult,
    ) -> Mapping[str, Any]: ...

    @property
    def artifact_provider_receipt_binding_required(self) -> bool: ...


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


def validate_host_action_bindings(
    start_request: Mapping[str, Any],
    bindings: Sequence[HostActionBinding],
) -> None:
    """Apply the exact binding equality check used by the host run loop."""

    expected = _available_manifest_bindings(start_request)
    actual = {
        key: binding.binding_contract_sha256
        for key, binding in _binding_index(bindings).items()
    }
    if actual != expected:
        raise ModelRunnerHostError(
            "executable bindings differ from the capability manifest"
        )


def _receipt_custody_values(value: Any) -> tuple[str, str, str]:
    if isinstance(value, Mapping):
        getter = value.get
    else:
        getter = lambda name: getattr(value, name, None)
    return (
        str(getter("provider_receipt_ref") or "").strip(),
        str(getter("provider_receipt_sha256") or "").strip(),
        str(getter("provider_identity_sha256") or "").strip(),
    )


def _require_durable_provider_receipt_custody(
    *,
    action: Mapping[str, Any],
    custody: ProviderReceiptCustody | None,
) -> None:
    if str(action.get("action_type") or "") not in _PROVIDER_ACTION_TYPES:
        return
    if (
        custody is None
        or getattr(custody, "durable", None) is not True
        or not callable(
            getattr(custody, "resolve_provider_receipt", None)
        )
    ):
        raise ModelRunnerHostError(
            "durable provider receipt custody is required"
        )


def _resolve_provider_receipt_custody(
    *,
    action: Mapping[str, Any],
    value: Any,
    custody: ProviderReceiptCustody | None,
) -> ProviderReceiptCustodyRecord | None:
    action_type = str(action.get("action_type") or "")
    (
        receipt_ref,
        receipt_sha256,
        provider_identity_sha256,
    ) = _receipt_custody_values(value)
    if action_type not in _PROVIDER_ACTION_TYPES:
        if receipt_ref or receipt_sha256 or provider_identity_sha256:
            raise ModelRunnerHostError(
                "verifier action cannot carry provider receipt custody"
            )
        return None
    if (
        not _PROVIDER_RECEIPT_RE.fullmatch(receipt_ref)
        or not _SHA256_RE.fullmatch(receipt_sha256)
        or not _SHA256_RE.fullmatch(provider_identity_sha256)
    ):
        raise ModelRunnerHostError(
            "provider action receipt custody fields are invalid"
        )
    _require_durable_provider_receipt_custody(
        action=action,
        custody=custody,
    )
    resolver = getattr(custody, "resolve_provider_receipt")
    try:
        record = resolver(receipt_ref)
    except Exception as exc:
        raise ModelRunnerHostError(
            "durable provider receipt custody lookup failed"
        ) from exc
    if not isinstance(record, ProviderReceiptCustodyRecord):
        raise ModelRunnerHostError(
            "durable provider receipt custody record is unavailable"
        )
    if (
        record.provider_receipt_ref != receipt_ref
        or record.provider_receipt_sha256 != receipt_sha256
        or record.provider_identity_sha256 != provider_identity_sha256
    ):
        raise ModelRunnerHostError(
            "provider receipt custody differs from the host result"
        )
    return record


def _bind_provider_receipt(
    *,
    action: Mapping[str, Any],
    host_result: HostActionResult,
    protocol: ModelRunnerProtocol,
) -> HostActionResult:
    if str(action.get("action_type") or "") not in _PROVIDER_ACTION_TYPES:
        return host_result
    if getattr(
        protocol,
        "artifact_provider_receipt_binding_required",
        True,
    ) is not True:
        # Exact v2 registrations drain under their original custody contract.
        # Only v3 declares the artifact-owned provider receipt member.
        return host_result
    try:
        binding = protocol.build_provider_receipt_binding(
            action, host_result
        )
    except Exception as exc:
        raise ModelRunnerHostError(
            "artifact provider receipt binding failed"
        ) from exc
    if not isinstance(binding, Mapping):
        raise ModelRunnerHostError(
            "artifact provider receipt binding is invalid"
        )
    receipt_ref = str(binding.get("provider_receipt_ref") or "")
    receipt_sha256 = str(binding.get("receipt_sha256") or "")
    provider_identity_sha256 = str(
        binding.get("provider_identity_sha256") or ""
    )
    if (
        receipt_ref != str(host_result.provider_receipt_ref or "")
        or not _SHA256_RE.fullmatch(receipt_sha256)
        or not _SHA256_RE.fullmatch(provider_identity_sha256)
        or (
            host_result.provider_receipt_sha256 is not None
            and host_result.provider_receipt_sha256 != receipt_sha256
        )
        or (
            host_result.provider_identity_sha256 is not None
            and host_result.provider_identity_sha256
            != provider_identity_sha256
        )
    ):
        raise ModelRunnerHostError(
            "artifact provider receipt binding differs from host result"
        )
    return replace(
        host_result,
        provider_receipt_sha256=receipt_sha256,
        provider_identity_sha256=provider_identity_sha256,
    )


def _validate_completion_receipt_custody(
    *,
    action: Mapping[str, Any],
    completion: Mapping[str, Any],
    custody_record: ProviderReceiptCustodyRecord | None,
) -> None:
    values = _receipt_custody_values(completion)
    if custody_record is None:
        if any(values):
            raise ModelRunnerHostError(
                "verifier completion cannot carry provider receipt custody"
            )
        return
    if values != (
        custody_record.provider_receipt_ref,
        custody_record.provider_receipt_sha256,
        custody_record.provider_identity_sha256,
    ):
        raise ModelRunnerHostError(
            "model completion changed provider receipt custody"
        )


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
        "provider_receipt_ref": str(
            completion.get("provider_receipt_ref") or ""
        ),
        "provider_receipt_sha256": str(
            completion.get("provider_receipt_sha256") or ""
        ),
        "provider_identity_sha256": str(
            completion.get("provider_identity_sha256") or ""
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
        provider_receipt_custody: ProviderReceiptCustody | None = None,
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
        self._provider_receipt_custody = provider_receipt_custody
        self._load_completion = load_completion
        self._max_actions = max_actions

    def run(
        self,
        start_request: Mapping[str, Any],
        *,
        continuation: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        validate_host_action_bindings(
            start_request,
            tuple(self._bindings.values()),
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

            _require_durable_provider_receipt_custody(
                action=action,
                custody=self._provider_receipt_custody,
            )

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
                host_result = _bind_provider_receipt(
                    action=action,
                    host_result=host_result,
                    protocol=self._protocol,
                )
                custody_record = _resolve_provider_receipt_custody(
                    action=action,
                    value=host_result,
                    custody=self._provider_receipt_custody,
                )
                completion = self._protocol.build_completion(
                    action,
                    host_result,
                )
                _validate_completion_receipt_custody(
                    action=action,
                    completion=completion,
                    custody_record=custody_record,
                )
                provider_request_id = host_result.provider_request_id
                replayed = False
            else:
                completion = cached
                custody_record = _resolve_provider_receipt_custody(
                    action=action,
                    value=completion,
                    custody=self._provider_receipt_custody,
                )
                _validate_completion_receipt_custody(
                    action=action,
                    completion=completion,
                    custody_record=custody_record,
                )
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
    "ProviderReceiptCustody",
    "ProviderReceiptCustodyRecord",
    "validate_host_action_bindings",
]
