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
_VERIFIER_ACTION_TYPES = frozenset({
    "verify_company",
    "verify_intent",
    "verify_contact",
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
    model_provider_response_ingestion: Mapping[str, Any] | None = None
    provider_action_receipt_sha256: str | None = None


@dataclass(frozen=True)
class HostCompiledProviderDispatch:
    """Exact credential-free provider dispatch compiled by the artifact."""

    canonical: str
    action_sha256: str
    action_type: str
    tool_id: str
    provider: str
    compiler_id: str
    compiler_contract_sha256: str
    request_sha256: str
    dispatch_sha256: str

    @classmethod
    def from_mapping(
        cls,
        action: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> "HostCompiledProviderDispatch":
        if not isinstance(value, Mapping):
            raise ModelRunnerHostError(
                "model artifact provider dispatch is invalid"
            )
        try:
            canonical = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            document = json.loads(canonical)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelRunnerHostError(
                "model artifact provider dispatch is not canonical JSON"
            ) from exc
        request = document.get("request")
        credential_binding = (
            request.get("credential_binding")
            if isinstance(request, Mapping)
            else None
        )
        required = {
            "schema_version",
            "action_sha256",
            "action_type",
            "tool_id",
            "compiler_id",
            "compiler_contract_sha256",
            "provider",
            "request",
            "request_sha256",
            "response_contract",
            "budgets",
            "idempotency_key",
            "dispatch_sha256",
        }
        if (
            not required.issubset(document)
            or document.get("schema_version")
            != "model-runner-provider-dispatch:v1"
            or document.get("action_sha256")
            != action.get("action_sha256")
            or document.get("action_type") != action.get("action_type")
            or document.get("tool_id") != action.get("tool_id")
            or document.get("idempotency_key")
            != f"model-action:{action.get('action_sha256')}"
            or not isinstance(document.get("provider"), str)
            or re.fullmatch(
                r"[a-z][a-z0-9_.-]{1,79}", document["provider"]
            )
            is None
            or not isinstance(document.get("compiler_id"), str)
            or not document["compiler_id"]
            or not isinstance(request, Mapping)
            or not isinstance(document.get("response_contract"), Mapping)
            or not isinstance(document.get("budgets"), Mapping)
            or not isinstance(credential_binding, Mapping)
            or credential_binding.get("persist") is not False
        ):
            raise ModelRunnerHostError(
                "model artifact provider dispatch identity differs"
            )
        compiler_contract_sha256 = str(
            document.get("compiler_contract_sha256") or ""
        )
        request_sha256 = str(document.get("request_sha256") or "")
        dispatch_sha256 = str(document.get("dispatch_sha256") or "")
        if any(
            _SHA256_RE.fullmatch(item) is None
            for item in (
                compiler_contract_sha256,
                request_sha256,
                dispatch_sha256,
            )
        ):
            raise ModelRunnerHostError(
                "model artifact provider dispatch hash is invalid"
            )
        if request_sha256 != _canonical_sha256(request):
            raise ModelRunnerHostError(
                "model artifact provider request hash differs"
            )
        dispatch_payload = dict(document)
        dispatch_payload.pop("dispatch_sha256")
        if dispatch_sha256 != _canonical_sha256(dispatch_payload):
            raise ModelRunnerHostError(
                "model artifact provider dispatch hash differs"
            )
        return cls(
            canonical=canonical,
            action_sha256=document["action_sha256"],
            action_type=document["action_type"],
            tool_id=document["tool_id"],
            provider=document["provider"],
            compiler_id=document["compiler_id"],
            compiler_contract_sha256=compiler_contract_sha256,
            request_sha256=request_sha256,
            dispatch_sha256=dispatch_sha256,
        )

    def to_mapping(self) -> dict[str, Any]:
        value = json.loads(self.canonical)
        if not isinstance(value, dict):  # pragma: no cover - constructor proof
            raise ModelRunnerHostError(
                "model artifact provider dispatch is invalid"
            )
        return value


@dataclass(frozen=True)
class HostActionBinding:
    """Credentialed execution for one exact model-owned action and tool ID."""

    action_type: str
    tool_id: str
    binding_contract_sha256: str
    dispatch: Callable[..., HostActionResult]


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

    def prepare_provider_request(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def ingest_provider_response(
        self,
        action: Mapping[str, Any],
        host_response: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def execute_verifier_action(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    @property
    def artifact_provider_receipt_binding_required(self) -> bool: ...


PersistTransition = Callable[..., None]
LoadCompletion = Callable[[str], Optional[Mapping[str, Any]]]


_PROVIDER_RESPONSE_INGESTION_FIELDS = frozenset({
    "schema_version",
    "action_sha256",
    "dispatch_sha256",
    "compiler_id",
    "compiler_contract_sha256",
    "request_sha256",
    "host_response_schema_version",
    "host_response_sha256",
    "provider",
    "parsed_response_schema_version",
    "parsed_response",
    "parsed_response_sha256",
    "ingestion_sha256",
})


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _requires_raw_provider_response_custody(
    protocol: ModelRunnerProtocol,
) -> bool:
    marker = getattr(protocol, "requires_raw_provider_response_custody", None)
    if marker is not None:
        if type(marker) is not bool:
            raise ModelRunnerHostError(
                "model artifact provider custody mode is invalid"
            )
        return marker
    # The deterministic cross-consumer replay protocol is intentionally
    # minimal.  Its complete prepare/ingest/verifier surface identifies the
    # current raw-response custody contract without inventing a local version.
    return all(
        callable(getattr(protocol, member, None))
        for member in (
            "prepare_provider_request",
            "ingest_provider_response",
            "execute_verifier_action",
        )
    )


def _validated_provider_response_ingestion(
    *,
    action: Mapping[str, Any],
    dispatch: HostCompiledProviderDispatch,
    host_response: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelRunnerHostError(
            "model artifact provider response ingestion is invalid"
        )
    try:
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        receipt = json.loads(canonical)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelRunnerHostError(
            "model artifact provider response ingestion is invalid"
        ) from exc
    parsed_response = receipt.get("parsed_response")
    if (
        set(receipt) != _PROVIDER_RESPONSE_INGESTION_FIELDS
        or receipt.get("schema_version")
        != "model-runner-provider-response-ingestion:v1"
        or receipt.get("action_sha256") != action.get("action_sha256")
        or receipt.get("dispatch_sha256") != dispatch.dispatch_sha256
        or receipt.get("compiler_id") != dispatch.compiler_id
        or receipt.get("compiler_contract_sha256")
        != dispatch.compiler_contract_sha256
        or receipt.get("request_sha256") != dispatch.request_sha256
        or receipt.get("host_response_schema_version")
        != "host-provider-response:v1"
        or receipt.get("host_response_sha256")
        != _canonical_sha256(host_response)
        or receipt.get("provider") != dispatch.provider
        or not isinstance(
            receipt.get("parsed_response_schema_version"), str
        )
        or not receipt["parsed_response_schema_version"]
        or not isinstance(parsed_response, Mapping)
        or parsed_response.get("schema_version")
        != receipt["parsed_response_schema_version"]
    ):
        raise ModelRunnerHostError(
            "model artifact provider response ingestion identity differs"
        )
    for field in (
        "host_response_sha256",
        "parsed_response_sha256",
        "ingestion_sha256",
    ):
        if _SHA256_RE.fullmatch(str(receipt.get(field) or "")) is None:
            raise ModelRunnerHostError(
                "model artifact provider response ingestion hash is invalid"
            )
    if receipt["parsed_response_sha256"] != _canonical_sha256(
        parsed_response
    ):
        raise ModelRunnerHostError(
            "model artifact parsed provider response hash differs"
        )
    ingestion_payload = dict(receipt)
    ingestion_payload.pop("ingestion_sha256")
    if receipt["ingestion_sha256"] != _canonical_sha256(ingestion_payload):
        raise ModelRunnerHostError(
            "model artifact provider response ingestion hash differs"
        )
    return receipt


def _validate_current_provider_result(
    *,
    action: Mapping[str, Any],
    dispatch: HostCompiledProviderDispatch,
    result: HostActionResult,
    protocol: ModelRunnerProtocol,
) -> HostActionResult:
    host_response = result.provider_response
    ingestion = result.model_provider_response_ingestion
    if (
        not isinstance(result.provider_action_receipt_sha256, str)
        or _SHA256_RE.fullmatch(
            result.provider_action_receipt_sha256
        )
        is None
    ):
        raise ModelRunnerHostError(
            "provider action durable custody receipt is invalid"
        )
    if host_response is None:
        if ingestion is not None:
            raise ModelRunnerHostError(
                "empty provider result carries ingestion custody"
            )
        return result
    if (
        not isinstance(host_response, Mapping)
        or set(host_response)
        != {"schema_version", "provider", "status_code", "body"}
        or host_response.get("schema_version")
        != "host-provider-response:v1"
        or host_response.get("provider") != dispatch.provider
        or isinstance(host_response.get("status_code"), bool)
        or not isinstance(host_response.get("status_code"), int)
        or not 100 <= host_response["status_code"] <= 599
        or not isinstance(host_response.get("body"), Mapping)
        or not isinstance(ingestion, Mapping)
    ):
        raise ModelRunnerHostError(
            "provider host response custody is invalid"
        )
    persisted = _validated_provider_response_ingestion(
        action=action,
        dispatch=dispatch,
        host_response=host_response,
        value=ingestion,
    )
    ingest = getattr(protocol, "ingest_provider_response", None)
    if not callable(ingest):
        raise ModelRunnerHostError(
            "model artifact provider response ingestor is unavailable"
        )
    try:
        replayed = ingest(action, host_response)
    except Exception as exc:
        raise ModelRunnerHostError(
            "model artifact provider response ingestion failed"
        ) from exc
    replayed_receipt = _validated_provider_response_ingestion(
        action=action,
        dispatch=dispatch,
        host_response=host_response,
        value=replayed,
    )
    if replayed_receipt != persisted:
        raise ModelRunnerHostError(
            "durable provider response ingestion differs from replay"
        )
    return replace(
        result,
        provider_response=dict(host_response),
        model_provider_response_ingestion=persisted,
    )


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


def _validate_current_completion(
    *,
    action: Mapping[str, Any],
    completion: Mapping[str, Any],
    host_result: HostActionResult,
) -> None:
    if not isinstance(completion, Mapping):
        raise ModelRunnerHostError("model completion is invalid")
    ingestion = host_result.model_provider_response_ingestion
    expected_response = (
        ingestion.get("parsed_response")
        if isinstance(ingestion, Mapping)
        else None
    )
    expected_response_sha256 = (
        ingestion.get("parsed_response_sha256")
        if isinstance(ingestion, Mapping)
        else _canonical_sha256(None)
    )
    if (
        completion.get("provider_response") != expected_response
        or completion.get("provider_response_sha256")
        != expected_response_sha256
        or str(completion.get("provider_receipt_ref") or "")
        != str(host_result.provider_receipt_ref or "")
        or str(completion.get("provider_identity_sha256") or "")
        != str(host_result.provider_identity_sha256 or "")
        or (
            host_result.provider_receipt_sha256 is not None
            and completion.get("provider_receipt_sha256")
            != host_result.provider_receipt_sha256
        )
    ):
        raise ModelRunnerHostError(
            "model completion differs from durable provider custody"
        )
    if str(action.get("action_type") or "") not in _PROVIDER_ACTION_TYPES:
        raise ModelRunnerHostError(
            "provider completion action identity differs"
        )


def _provider_action_custody_document(
    host_result: HostActionResult,
) -> dict[str, Any]:
    return {
        "schema_version": "model-runner-host-action-custody:v1",
        "host_response": (
            None
            if host_result.provider_response is None
            else dict(host_result.provider_response)
        ),
        "model_provider_response_ingestion": (
            None
            if host_result.model_provider_response_ingestion is None
            else dict(host_result.model_provider_response_ingestion)
        ),
        "provider_action_receipt_sha256": (
            host_result.provider_action_receipt_sha256
        ),
    }


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

    def _compile_provider_dispatch(
        self,
        action: Mapping[str, Any],
    ) -> HostCompiledProviderDispatch:
        prepare = getattr(self._protocol, "prepare_provider_request", None)
        if not callable(prepare):
            raise ModelRunnerHostError(
                "model artifact provider compiler is unavailable"
            )
        try:
            value = prepare(action)
        except Exception as exc:
            raise ModelRunnerHostError(
                "model artifact provider request preparation failed"
            ) from exc
        return HostCompiledProviderDispatch.from_mapping(action, value)

    def _build_current_provider_completion(
        self,
        *,
        action: Mapping[str, Any],
        binding: HostActionBinding,
        compiled_dispatch: HostCompiledProviderDispatch,
    ) -> tuple[Mapping[str, Any], HostActionResult]:
        host_result = binding.dispatch(action, compiled_dispatch)
        if not isinstance(host_result, HostActionResult):
            raise ModelRunnerHostError(
                "host binding must return HostActionResult"
            )
        if host_result.outcome not in _OUTCOMES:
            raise ModelRunnerHostError("host binding outcome is invalid")
        host_result = _validate_current_provider_result(
            action=action,
            dispatch=compiled_dispatch,
            result=host_result,
            protocol=self._protocol,
        )
        if callable(
            getattr(self._protocol, "build_provider_receipt_binding", None)
        ):
            host_result = _bind_provider_receipt(
                action=action,
                host_result=host_result,
                protocol=self._protocol,
            )
        if self._provider_receipt_custody is not None:
            _resolve_provider_receipt_custody(
                action=action,
                value=host_result,
                custody=self._provider_receipt_custody,
            )
        completion = self._protocol.build_completion(action, host_result)
        _validate_current_completion(
            action=action,
            completion=completion,
            host_result=host_result,
        )
        return completion, host_result

    def _reload_current_provider_completion(
        self,
        *,
        action: Mapping[str, Any],
        compiled_dispatch: HostCompiledProviderDispatch,
        cached: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], HostActionResult]:
        custody = cached.get("provider_action_custody")
        completion = cached.get("completion")
        if (
            not isinstance(custody, Mapping)
            or custody.get("schema_version")
            != "model-runner-host-action-custody:v1"
            or set(custody)
            != {
                "schema_version",
                "host_response",
                "model_provider_response_ingestion",
                "provider_action_receipt_sha256",
            }
            or not isinstance(completion, Mapping)
        ):
            raise ModelRunnerHostError(
                "durable provider action custody is incomplete"
            )
        host_result = HostActionResult(
            outcome=str(completion.get("outcome") or ""),
            reason_code=str(completion.get("reason_code") or ""),
            provider_response=custody.get("host_response"),
            calls=completion.get("calls"),
            cost_credits=completion.get("cost_credits"),
            latency_ms=completion.get("latency_ms"),
            provider_receipt_ref=completion.get("provider_receipt_ref"),
            provider_receipt_sha256=completion.get(
                "provider_receipt_sha256"
            ),
            provider_identity_sha256=completion.get(
                "provider_identity_sha256"
            ),
            model_provider_response_ingestion=custody.get(
                "model_provider_response_ingestion"
            ),
            provider_action_receipt_sha256=custody.get(
                "provider_action_receipt_sha256"
            ),
        )
        host_result = _validate_current_provider_result(
            action=action,
            dispatch=compiled_dispatch,
            result=host_result,
            protocol=self._protocol,
        )
        rebuilt = self._protocol.build_completion(action, host_result)
        if dict(rebuilt) != dict(completion):
            raise ModelRunnerHostError(
                "durable provider completion differs from replay"
            )
        _validate_current_completion(
            action=action,
            completion=completion,
            host_result=host_result,
        )
        return dict(completion), host_result

    def _build_current_verifier_completion(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        execute = getattr(self._protocol, "execute_verifier_action", None)
        if not callable(execute):
            raise ModelRunnerHostError(
                "model artifact verifier is unavailable"
            )
        try:
            execution = execute(action)
        except Exception as exc:
            raise ModelRunnerHostError(
                "model artifact verifier execution failed"
            ) from exc
        result = execution.get("result") if isinstance(execution, Mapping) else None
        if (
            not isinstance(execution, Mapping)
            or execution.get("action_sha256")
            != action.get("action_sha256")
            or execution.get("action_type") != action.get("action_type")
            or execution.get("calls") != 0
            or execution.get("cost_credits") not in (0, 0.0)
            or execution.get("provider_receipt_allowed") is not False
            or not isinstance(result, Mapping)
            or not isinstance(result.get("reason_code"), str)
        ):
            raise ModelRunnerHostError(
                "model artifact verifier execution identity differs"
            )
        host_result = HostActionResult(
            outcome="succeeded",
            reason_code=result["reason_code"],
            provider_response=dict(result),
            calls=0,
            cost_credits=0.0,
            latency_ms=0.0,
            model_provider_response_ingestion=None,
            provider_action_receipt_sha256=None,
        )
        completion = self._protocol.build_completion(action, host_result)
        if any(
            completion.get(field)
            for field in (
                "provider_receipt_ref",
                "provider_receipt_sha256",
                "provider_identity_sha256",
            )
        ):
            raise ModelRunnerHostError(
                "verifier completion carries provider custody"
            )
        return completion

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

            current_custody = _requires_raw_provider_response_custody(
                self._protocol
            )
            compiled_dispatch = (
                self._compile_provider_dispatch(action)
                if current_custody
                and action_type in _PROVIDER_ACTION_TYPES
                else None
            )
            if not current_custody:
                _require_durable_provider_receipt_custody(
                    action=action,
                    custody=self._provider_receipt_custody,
                )

            cached = (
                self._load_completion(idempotency_key)
                if self._load_completion is not None
                else None
            )
            provider_action_custody: Mapping[str, Any] | None = None
            if current_custody and action_type in _VERIFIER_ACTION_TYPES:
                if cached is not None:
                    if not isinstance(cached, Mapping):
                        raise ModelRunnerHostError(
                            "durable verifier completion is invalid"
                        )
                    completion = cached.get("completion", cached)
                    if not isinstance(completion, Mapping):
                        raise ModelRunnerHostError(
                            "durable verifier completion is invalid"
                        )
                    rebuilt = self._build_current_verifier_completion(action)
                    if dict(rebuilt) != dict(completion):
                        raise ModelRunnerHostError(
                            "durable verifier completion differs from replay"
                        )
                    replayed = True
                else:
                    completion = self._build_current_verifier_completion(
                        action
                    )
                    replayed = False
                provider_request_id = None
            elif current_custody and action_type in _PROVIDER_ACTION_TYPES:
                if compiled_dispatch is None:  # pragma: no cover - branch proof
                    raise ModelRunnerHostError(
                        "model artifact provider dispatch is unavailable"
                    )
                if cached is None:
                    completion, host_result = (
                        self._build_current_provider_completion(
                            action=action,
                            binding=binding,
                            compiled_dispatch=compiled_dispatch,
                        )
                    )
                    replayed = False
                else:
                    if not isinstance(cached, Mapping):
                        raise ModelRunnerHostError(
                            "durable provider completion is invalid"
                        )
                    completion, host_result = (
                        self._reload_current_provider_completion(
                            action=action,
                            compiled_dispatch=compiled_dispatch,
                            cached=cached,
                        )
                    )
                    replayed = True
                provider_request_id = host_result.provider_request_id
                provider_action_custody = (
                    _provider_action_custody_document(host_result)
                )
            elif cached is None:
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
            persisted_transition = {
                "idempotency_key": idempotency_key,
                "action": dict(action),
                "completion": dict(completion),
                "continuation": dict(advanced_continuation),
                "host_receipt": receipt,
            }
            if provider_action_custody is not None:
                persisted_transition["provider_action_custody"] = dict(
                    provider_action_custody
                )
            self._persist_transition(
                **persisted_transition,
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
    "HostCompiledProviderDispatch",
    "ModelRunnerHostError",
    "ModelRunnerProtocol",
    "ProviderReceiptCustody",
    "ProviderReceiptCustodyRecord",
    "validate_host_action_bindings",
]
