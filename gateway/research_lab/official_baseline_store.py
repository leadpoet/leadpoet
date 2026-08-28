"""Service-role RPC adapter for official-baseline provider attempt custody.

Only closed identity, accounting, and protected-job documents cross this
boundary. Provider responses and projected company data never enter Supabase.
"""

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Mapping

from gateway.db.client import get_write_client
from gateway.research_lab.official_baseline_model_runner import (
    OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_REPLAY_IDENTITY_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_REPLAY_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_RESERVATION_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION,
    OFFICIAL_BASELINE_RUN_REGISTRATION_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION,
    validate_official_baseline_provider_closure,
)
from research_lab.canonical import sha256_json


OFFICIAL_BASELINE_MIGRATION = (
    "scripts/164-research-lab-official-baseline-action-authority.sql"
)
OFFICIAL_BASELINE_RPC_REGISTER_RUN = "research_lab_official_baseline_register_run_v1"
OFFICIAL_BASELINE_RPC_RESERVE_ACTION = (
    "research_lab_official_baseline_reserve_action_v1"
)
OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_KNOWN = (
    "research_lab_official_baseline_record_terminal_known_v1"
)
OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_UNCERTAIN = (
    "research_lab_official_baseline_record_terminal_uncertain_v1"
)
OFFICIAL_BASELINE_RPC_LOAD_REPLAY = "research_lab_official_baseline_load_replay_v1"
OFFICIAL_BASELINE_RPC_CLOSE_UNIT = "research_lab_official_baseline_close_unit_v1"
OFFICIAL_BASELINE_RPC_LOAD_FRONTIER = "research_lab_official_baseline_load_frontier_v1"
OFFICIAL_BASELINE_RPCS = (
    OFFICIAL_BASELINE_RPC_REGISTER_RUN,
    OFFICIAL_BASELINE_RPC_RESERVE_ACTION,
    OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_KNOWN,
    OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_UNCERTAIN,
    OFFICIAL_BASELINE_RPC_LOAD_REPLAY,
    OFFICIAL_BASELINE_RPC_CLOSE_UNIT,
    OFFICIAL_BASELINE_RPC_LOAD_FRONTIER,
)

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_UNIT_REF_RE = re.compile(r"baseline_icp:[0-9a-f]{64}")
_RESERVATION_REF_RE = re.compile(r"baseline_reservation:[0-9a-f]{64}")
_PROVIDER_RECEIPT_REF_RE = re.compile(r"provider_receipt:[0-9a-f]{16}")
_SAFE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}")

_RUN_REGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "run_sha256",
        "benchmark_date",
        "rolling_window_hash",
        "model_artifact_hash",
        "manifest_hash",
        "release_selection_sha256",
        "artifact_key_sha256",
        "protocol_generation_sha256",
        "projection_identity_sha256",
        "authority_identity_sha256",
    }
)
_ACTION_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_key",
        "run_sha256",
        "unit_ref",
        "action_idempotency_sha256",
        "action_sha256",
        "action_sequence",
        "action_type",
        "tool_id",
        "binding_contract_sha256",
        "request_fingerprint_sha256",
        "request_body_sha256",
        "call_cap",
        "credit_cap_microunits",
        "timeout_ms",
        "protected_job_ref",
        "protected_request_sha256",
        "lease_holder_sha256",
        "expected_frontier_sha256",
    }
)
_TERMINAL_KNOWN_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_key",
        "reservation_ref",
        "lease_generation",
        "protected_job_ref",
        "protected_request_sha256",
        "protected_result_sha256",
        "protected_terminal_receipt_ref",
        "protected_terminal_receipt_sha256",
        "provider_request_ref",
        "provider_receipt_ref",
        "provider_receipt_sha256",
        "provider_identity_sha256",
        "model_provider_response_sha256",
        "outcome",
        "call_count",
        "cost_microunits",
        "latency_ms",
    }
)
_TERMINAL_UNCERTAIN_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_key",
        "reservation_ref",
        "lease_generation",
        "protected_job_ref",
        "protected_request_sha256",
        "provider_request_ref",
        "uncertainty_sha256",
    }
)
_REPLAY_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_key",
        "run_sha256",
        "unit_ref",
        "action_idempotency_sha256",
        "action_sha256",
        "request_fingerprint_sha256",
    }
)
_UNIT_COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "run_sha256",
        "unit_ref",
        "protocol_generation_sha256",
        "raw_input_sha256",
        "start_request_sha256",
        "terminal_result_sha256",
        "model_receipt_sha256",
        "projection_sha256",
    }
)
_REPLAY_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "attempt_key",
        "reservation_ref",
        "lease_generation",
        "lease_expires_at",
        "protected_job_ref",
        "protected_request_sha256",
        "protected_result_sha256",
        "protected_terminal_receipt_ref",
        "protected_terminal_receipt_sha256",
        "provider_request_ref",
        "provider_receipt_ref",
        "provider_receipt_sha256",
        "provider_identity_sha256",
        "model_provider_response_sha256",
        "outcome",
        "call_count",
        "cost_microunits",
        "latency_ms",
        "attempt_sha256",
    }
)


class OfficialBaselineStoreError(RuntimeError):
    """The official-baseline SQL authority rejected or malformed a record."""


def official_baseline_action_replay_identity(
    *,
    run_sha256: str,
    unit_ref: str,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact action identity shared by reserve, replay, and custody."""

    if not isinstance(action, Mapping):
        raise OfficialBaselineStoreError("official baseline model action is invalid")

    def action_hash(field: str) -> str:
        raw = str(action.get(field) or "").lower()
        value = raw if raw.startswith("sha256:") else "sha256:" + raw
        return _hash(value, f"official baseline {field}")

    identity = {
        "schema_version": OFFICIAL_BASELINE_ACTION_REPLAY_IDENTITY_SCHEMA_VERSION,
        "attempt_key": "",
        "run_sha256": _hash(run_sha256, "official baseline run_sha256"),
        "unit_ref": _unit_ref(unit_ref),
        "action_idempotency_sha256": action_hash("idempotency_key"),
        "action_sha256": action_hash("action_sha256"),
        "request_fingerprint_sha256": action_hash("request_fingerprint_sha256"),
    }
    attempt_body = {
        "schema_version": "leadpoet.research_lab.official_baseline_attempt_key.v1",
        **{
            key: value
            for key, value in identity.items()
            if key != "schema_version" and key != "attempt_key"
        },
    }
    identity["attempt_key"] = sha256_json(attempt_body)
    return _validate_replay_identity(identity)


def official_baseline_terminal_store_outcome(model_outcome: Any) -> str:
    """Map one known model outcome to migration-163 accounting.

    ``unavailable`` and ``timeout`` remain known protected results, but the
    SQL accounting vocabulary records them as ``failed``. Ambiguous provider
    consumption never uses this mapping; it enters ``terminal_uncertain``.
    """

    normalized = str(model_outcome or "")
    if normalized in {"succeeded", "empty", "failed"}:
        return normalized
    if normalized in {"unavailable", "timeout"}:
        return "failed"
    raise OfficialBaselineStoreError(
        "official baseline model terminal outcome is unsupported"
    )


def _closed_document(
    value: Any,
    *,
    fields: frozenset[str],
    schema_version: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OfficialBaselineStoreError(f"{label} is not closed")
    normalized = dict(value)
    if normalized.get("schema_version") != schema_version:
        raise OfficialBaselineStoreError(f"{label} schema differs")
    return normalized


def _hash(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if _HASH_RE.fullmatch(normalized) is None:
        raise OfficialBaselineStoreError(f"{label} is invalid")
    return normalized


def _unit_ref(value: Any, label: str = "unit_ref") -> str:
    normalized = str(value or "")
    if _UNIT_REF_RE.fullmatch(normalized) is None:
        raise OfficialBaselineStoreError(f"{label} is invalid")
    return normalized


def _safe_ref(value: Any, label: str) -> str:
    normalized = str(value or "")
    if _SAFE_REF_RE.fullmatch(normalized) is None:
        raise OfficialBaselineStoreError(f"{label} is invalid")
    return normalized


def _integer(value: Any, *, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise OfficialBaselineStoreError(f"{label} is invalid")
    return value


def _timestamp(value: Any, label: str) -> str:
    normalized = str(value or "")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OfficialBaselineStoreError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OfficialBaselineStoreError(f"{label} is invalid")
    return normalized


def _validate_run_registration(value: Any) -> dict[str, Any]:
    document = _closed_document(
        value,
        fields=_RUN_REGISTRATION_FIELDS,
        schema_version=OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION,
        label="official baseline run registration",
    )
    try:
        if (
            date.fromisoformat(str(document["benchmark_date"])).isoformat()
            != document["benchmark_date"]
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise OfficialBaselineStoreError(
            "official baseline benchmark date is invalid"
        ) from exc
    for field in _RUN_REGISTRATION_FIELDS - {"schema_version", "benchmark_date"}:
        _hash(document[field], f"official baseline {field}")
    return document


def _validate_action_authorization(value: Any) -> dict[str, Any]:
    document = _closed_document(
        value,
        fields=_ACTION_AUTHORIZATION_FIELDS,
        schema_version=OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION,
        label="official baseline action authorization",
    )
    for field in (
        "attempt_key",
        "run_sha256",
        "action_idempotency_sha256",
        "action_sha256",
        "binding_contract_sha256",
        "request_fingerprint_sha256",
        "request_body_sha256",
        "protected_request_sha256",
        "lease_holder_sha256",
        "expected_frontier_sha256",
    ):
        _hash(document[field], f"official baseline {field}")
    _unit_ref(document["unit_ref"])
    _safe_ref(document["tool_id"], "official baseline tool_id")
    _safe_ref(document["protected_job_ref"], "official baseline protected_job_ref")
    sequence = _integer(
        document["action_sequence"],
        minimum=0,
        maximum=9999,
        label="official baseline action_sequence",
    )
    call_cap = _integer(
        document["call_cap"],
        minimum=0,
        maximum=100_000,
        label="official baseline call_cap",
    )
    credit_cap = _integer(
        document["credit_cap_microunits"],
        minimum=0,
        maximum=100_000_000,
        label="official baseline credit_cap_microunits",
    )
    timeout_ms = _integer(
        document["timeout_ms"],
        minimum=0,
        maximum=900_000,
        label="official baseline timeout_ms",
    )
    action_type = document["action_type"]
    verifier = action_type in {"verify_company", "verify_intent", "verify_contact"}
    tool = action_type in {
        "normalize_icp",
        "execute_candidate_tool",
        "execute_intent_tool",
        "execute_contact_tool",
    }
    if not verifier and not tool:
        raise OfficialBaselineStoreError("official baseline action_type is invalid")
    if (
        verifier
        and (call_cap != 0 or credit_cap != 0 or timeout_ms != 0)
    ) or (tool and (call_cap < 1 or timeout_ms < 1)):
        raise OfficialBaselineStoreError(
            "official baseline action accounting is invalid"
        )
    assert sequence >= 0
    return document


def _validate_terminal_known(value: Any) -> dict[str, Any]:
    document = _closed_document(
        value,
        fields=_TERMINAL_KNOWN_FIELDS,
        schema_version=OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION,
        label="official baseline known terminal",
    )
    for field in (
        "attempt_key",
        "protected_request_sha256",
        "protected_result_sha256",
        "protected_terminal_receipt_sha256",
        "model_provider_response_sha256",
    ):
        _hash(document[field], f"official baseline {field}")
    if _RESERVATION_REF_RE.fullmatch(str(document["reservation_ref"] or "")) is None:
        raise OfficialBaselineStoreError("official baseline reservation_ref is invalid")
    _integer(
        document["lease_generation"],
        minimum=1,
        maximum=1,
        label="official baseline lease_generation",
    )
    _safe_ref(document["protected_job_ref"], "official baseline protected_job_ref")
    _safe_ref(
        document["protected_terminal_receipt_ref"],
        "official baseline protected_terminal_receipt_ref",
    )
    if document["outcome"] not in {"succeeded", "empty", "failed"}:
        raise OfficialBaselineStoreError("official baseline outcome is invalid")
    calls = _integer(
        document["call_count"],
        minimum=0,
        maximum=100_000,
        label="official baseline call_count",
    )
    cost = _integer(
        document["cost_microunits"],
        minimum=0,
        maximum=100_000_000,
        label="official baseline cost_microunits",
    )
    _integer(
        document["latency_ms"],
        minimum=0,
        maximum=900_000,
        label="official baseline latency_ms",
    )
    provider_fields = (
        "provider_request_ref",
        "provider_receipt_ref",
        "provider_receipt_sha256",
        "provider_identity_sha256",
    )
    provider_values = tuple(document[field] for field in provider_fields)
    if all(value is None for value in provider_values):
        if (
            document["outcome"] not in {"succeeded", "failed"}
            or calls != 0
            or cost != 0
        ):
            raise OfficialBaselineStoreError(
                "official baseline verifier accounting is invalid"
            )
    elif any(value is None for value in provider_values) or calls < 1:
        raise OfficialBaselineStoreError(
            "official baseline provider custody is incomplete"
        )
    else:
        _safe_ref(
            document["provider_request_ref"], "official baseline provider_request_ref"
        )
        if (
            _PROVIDER_RECEIPT_REF_RE.fullmatch(str(document["provider_receipt_ref"]))
            is None
        ):
            raise OfficialBaselineStoreError(
                "official baseline provider_receipt_ref is invalid"
            )
        _hash(
            document["provider_receipt_sha256"],
            "official baseline provider_receipt_sha256",
        )
        _hash(
            document["provider_identity_sha256"],
            "official baseline provider_identity_sha256",
        )
    return document


def _validate_terminal_uncertain(value: Any) -> dict[str, Any]:
    document = _closed_document(
        value,
        fields=_TERMINAL_UNCERTAIN_FIELDS,
        schema_version=OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION,
        label="official baseline uncertain terminal",
    )
    for field in (
        "attempt_key",
        "protected_request_sha256",
        "uncertainty_sha256",
    ):
        _hash(document[field], f"official baseline {field}")
    if _RESERVATION_REF_RE.fullmatch(str(document["reservation_ref"] or "")) is None:
        raise OfficialBaselineStoreError("official baseline reservation_ref is invalid")
    _integer(
        document["lease_generation"],
        minimum=1,
        maximum=1,
        label="official baseline lease_generation",
    )
    _safe_ref(document["protected_job_ref"], "official baseline protected_job_ref")
    if document["provider_request_ref"] is not None:
        _safe_ref(
            document["provider_request_ref"],
            "official baseline provider_request_ref",
        )
    return document


def _validate_replay_identity(value: Any) -> dict[str, Any]:
    document = _closed_document(
        value,
        fields=_REPLAY_IDENTITY_FIELDS,
        schema_version=OFFICIAL_BASELINE_ACTION_REPLAY_IDENTITY_SCHEMA_VERSION,
        label="official baseline replay identity",
    )
    for field in (
        "attempt_key",
        "run_sha256",
        "action_idempotency_sha256",
        "action_sha256",
        "request_fingerprint_sha256",
    ):
        _hash(document[field], f"official baseline {field}")
    _unit_ref(document["unit_ref"])
    return document


def _validate_unit_completion(value: Any) -> dict[str, Any]:
    document = _closed_document(
        value,
        fields=_UNIT_COMPLETION_FIELDS,
        schema_version=OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION,
        label="official baseline unit completion",
    )
    for field in _UNIT_COMPLETION_FIELDS - {"schema_version", "unit_ref"}:
        _hash(document[field], f"official baseline {field}")
    _unit_ref(document["unit_ref"])
    return document


def _validate_registration_result(
    value: Any, *, registration: Mapping[str, Any]
) -> dict[str, Any]:
    result = _closed_document(
        value,
        fields=frozenset(
            {"schema_version", "run_sha256", "registration_sha256", "idempotent"}
        ),
        schema_version=OFFICIAL_BASELINE_RUN_REGISTRATION_RESULT_SCHEMA_VERSION,
        label="official baseline run registration result",
    )
    if (
        result["run_sha256"] != registration["run_sha256"]
        or result["registration_sha256"] != sha256_json(dict(registration))
        or type(result["idempotent"]) is not bool
    ):
        raise OfficialBaselineStoreError(
            "official baseline run registration result differs"
        )
    return result


def _validate_reservation_result(
    value: Any, *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    result = _closed_document(
        value,
        fields=frozenset(
            {
                "schema_version",
                "disposition",
                "attempt_key",
                "reservation_ref",
                "lease_generation",
                "lease_expires_at",
                "protected_job_ref",
                "protected_request_sha256",
                "attempt_sha256",
            }
        ),
        schema_version=OFFICIAL_BASELINE_ACTION_RESERVATION_RESULT_SCHEMA_VERSION,
        label="official baseline action reservation result",
    )
    if result["disposition"] not in {
        "reserved_new",
        "reserved_existing",
        "inflight",
        "terminal_known",
        "terminal_uncertain",
    }:
        raise OfficialBaselineStoreError(
            "official baseline action reservation disposition is invalid"
        )
    attempt_key = authorization["attempt_key"]
    expected_ref = "baseline_reservation:" + attempt_key.removeprefix("sha256:")
    if (
        result["attempt_key"] != attempt_key
        or result["reservation_ref"] != expected_ref
        or result["protected_job_ref"] != authorization["protected_job_ref"]
        or result["protected_request_sha256"]
        != authorization["protected_request_sha256"]
    ):
        raise OfficialBaselineStoreError(
            "official baseline action reservation identity differs"
        )
    _integer(
        result["lease_generation"],
        minimum=1,
        maximum=1,
        label="official baseline lease_generation",
    )
    _timestamp(result["lease_expires_at"], "official baseline lease_expires_at")
    _hash(result["attempt_sha256"], "official baseline attempt_sha256")
    return result


def _validate_terminal_result(
    value: Any, *, terminal: Mapping[str, Any], expected_state: str
) -> dict[str, Any]:
    result = _closed_document(
        value,
        fields=frozenset(
            {"schema_version", "state", "attempt_key", "attempt_sha256", "idempotent"}
        ),
        schema_version=OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION,
        label="official baseline action terminal result",
    )
    if (
        result["state"] != expected_state
        or result["attempt_key"] != terminal["attempt_key"]
        or type(result["idempotent"]) is not bool
    ):
        raise OfficialBaselineStoreError(
            "official baseline action terminal result differs"
        )
    _hash(result["attempt_sha256"], "official baseline attempt_sha256")
    return result


def _validate_replay_result(
    value: Any, *, identity: Mapping[str, Any]
) -> dict[str, Any]:
    result = _closed_document(
        value,
        fields=_REPLAY_RESULT_FIELDS,
        schema_version=OFFICIAL_BASELINE_ACTION_REPLAY_RESULT_SCHEMA_VERSION,
        label="official baseline action replay result",
    )
    if result["attempt_key"] != identity["attempt_key"]:
        raise OfficialBaselineStoreError(
            "official baseline action replay attempt differs"
        )
    state = result["state"]
    nullable_fields = _REPLAY_RESULT_FIELDS - {
        "schema_version",
        "state",
        "attempt_key",
    }
    if state == "absent":
        if any(result[field] is not None for field in nullable_fields):
            raise OfficialBaselineStoreError(
                "official baseline absent replay contains custody"
            )
        return result
    if state not in {"reserved", "terminal_known", "terminal_uncertain"}:
        raise OfficialBaselineStoreError(
            "official baseline action replay state is invalid"
        )
    if _RESERVATION_REF_RE.fullmatch(str(result["reservation_ref"] or "")) is None:
        raise OfficialBaselineStoreError(
            "official baseline replay reservation_ref is invalid"
        )
    _integer(
        result["lease_generation"],
        minimum=1,
        maximum=1,
        label="official baseline replay lease_generation",
    )
    _timestamp(result["lease_expires_at"], "official baseline replay lease_expires_at")
    _safe_ref(result["protected_job_ref"], "official baseline replay protected_job_ref")
    _hash(
        result["protected_request_sha256"],
        "official baseline replay protected_request_sha256",
    )
    _hash(result["attempt_sha256"], "official baseline replay attempt_sha256")
    terminal_fields = (
        "protected_result_sha256",
        "protected_terminal_receipt_ref",
        "protected_terminal_receipt_sha256",
        "provider_request_ref",
        "provider_receipt_ref",
        "provider_receipt_sha256",
        "provider_identity_sha256",
        "model_provider_response_sha256",
        "outcome",
        "call_count",
        "cost_microunits",
        "latency_ms",
    )
    if state == "reserved":
        if any(result[field] is not None for field in terminal_fields):
            raise OfficialBaselineStoreError(
                "official baseline reserved replay contains terminal custody"
            )
        return result
    if state == "terminal_uncertain":
        if any(
            result[field] is not None
            for field in terminal_fields
            if field != "provider_request_ref"
        ):
            raise OfficialBaselineStoreError(
                "official baseline uncertain replay contains terminal result"
            )
        if result["provider_request_ref"] is not None:
            _safe_ref(
                result["provider_request_ref"],
                "official baseline replay provider_request_ref",
            )
        return result
    for field in (
        "protected_result_sha256",
        "protected_terminal_receipt_sha256",
        "model_provider_response_sha256",
    ):
        _hash(result[field], f"official baseline replay {field}")
    _safe_ref(
        result["protected_terminal_receipt_ref"],
        "official baseline replay protected_terminal_receipt_ref",
    )
    if result["outcome"] not in {"succeeded", "empty", "failed"}:
        raise OfficialBaselineStoreError("official baseline replay outcome is invalid")
    calls = _integer(
        result["call_count"],
        minimum=0,
        maximum=100_000,
        label="official baseline replay call_count",
    )
    cost = _integer(
        result["cost_microunits"],
        minimum=0,
        maximum=100_000_000,
        label="official baseline replay cost_microunits",
    )
    _integer(
        result["latency_ms"],
        minimum=0,
        maximum=900_000,
        label="official baseline replay latency_ms",
    )
    provider_fields = (
        "provider_request_ref",
        "provider_receipt_ref",
        "provider_receipt_sha256",
        "provider_identity_sha256",
    )
    values = tuple(result[field] for field in provider_fields)
    if all(value is None for value in values):
        if result["outcome"] not in {"succeeded", "failed"} or calls != 0 or cost != 0:
            raise OfficialBaselineStoreError(
                "official baseline verifier replay accounting is invalid"
            )
    elif any(value is None for value in values) or calls < 1:
        raise OfficialBaselineStoreError(
            "official baseline provider replay custody is incomplete"
        )
    else:
        _safe_ref(
            result["provider_request_ref"],
            "official baseline replay provider_request_ref",
        )
        if (
            _PROVIDER_RECEIPT_REF_RE.fullmatch(str(result["provider_receipt_ref"]))
            is None
        ):
            raise OfficialBaselineStoreError(
                "official baseline replay provider_receipt_ref is invalid"
            )
        _hash(
            result["provider_receipt_sha256"],
            "official baseline replay provider_receipt_sha256",
        )
        _hash(
            result["provider_identity_sha256"],
            "official baseline replay provider_identity_sha256",
        )
    return result


def _response_data(response: Any) -> Any:
    if isinstance(response, Mapping):
        return response.get("data")
    return getattr(response, "data", None)


class SupabaseOfficialBaselineAttemptStore:
    """Synchronous service-role adapter for the seven migration-163 RPCs."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        return self._client if self._client is not None else get_write_client()

    def _rpc(self, name: str, params: Mapping[str, Any]) -> Any:
        try:
            response = self.client.rpc(name, dict(params)).execute()
        except Exception as exc:  # noqa: BLE001 - fail closed at SQL authority
            raise OfficialBaselineStoreError(
                f"official baseline authority RPC failed:{name}:{type(exc).__name__}"
            ) from exc
        data = _response_data(response)
        if not isinstance(data, Mapping):
            raise OfficialBaselineStoreError(
                f"official baseline authority RPC response is invalid:{name}"
            )
        return dict(data)

    def register_run(self, *, registration: Mapping[str, Any]) -> Mapping[str, Any]:
        document = _validate_run_registration(registration)
        result = self._rpc(
            OFFICIAL_BASELINE_RPC_REGISTER_RUN,
            {"p_registration": document},
        )
        return _validate_registration_result(result, registration=document)

    def reserve_action(self, *, authorization: Mapping[str, Any]) -> Mapping[str, Any]:
        document = _validate_action_authorization(authorization)
        result = self._rpc(
            OFFICIAL_BASELINE_RPC_RESERVE_ACTION,
            {"p_authorization": document},
        )
        return _validate_reservation_result(result, authorization=document)

    def record_terminal_known(
        self, *, terminal: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        document = _validate_terminal_known(terminal)
        result = self._rpc(
            OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_KNOWN,
            {"p_terminal": document},
        )
        return _validate_terminal_result(
            result,
            terminal=document,
            expected_state="terminal_known",
        )

    def record_terminal_uncertain(
        self, *, uncertainty: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        document = _validate_terminal_uncertain(uncertainty)
        result = self._rpc(
            OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_UNCERTAIN,
            {"p_terminal": document},
        )
        return _validate_terminal_result(
            result,
            terminal=document,
            expected_state="terminal_uncertain",
        )

    def load_replay(self, *, identity: Mapping[str, Any]) -> Mapping[str, Any]:
        document = _validate_replay_identity(identity)
        result = self._rpc(
            OFFICIAL_BASELINE_RPC_LOAD_REPLAY,
            {"p_identity": document},
        )
        return _validate_replay_result(result, identity=document)

    def close_unit(self, *, closure: Mapping[str, Any]) -> Mapping[str, Any]:
        completion = _validate_unit_completion(closure)
        result = self._rpc(
            OFFICIAL_BASELINE_RPC_CLOSE_UNIT,
            {"p_completion": completion},
        )
        validate_official_baseline_provider_closure(
            result,
            expected_completion=completion,
        )
        return result

    def load_frontier(self, *, run_sha256: str, unit_ref: str) -> Mapping[str, Any]:
        normalized_run = _hash(run_sha256, "official baseline run_sha256")
        normalized_unit = _unit_ref(unit_ref)
        result = self._rpc(
            OFFICIAL_BASELINE_RPC_LOAD_FRONTIER,
            {"p_run_sha256": normalized_run, "p_unit_ref": normalized_unit},
        )
        completion = {
            "schema_version": OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION,
            **{
                field: result.get(field)
                for field in _UNIT_COMPLETION_FIELDS - {"schema_version"}
            },
        }
        if (
            completion["run_sha256"] != normalized_run
            or completion["unit_ref"] != normalized_unit
        ):
            raise OfficialBaselineStoreError(
                "official baseline frontier identity differs"
            )
        _validate_unit_completion(completion)
        validate_official_baseline_provider_closure(
            result,
            expected_completion=completion,
        )
        return result


__all__ = [
    "OFFICIAL_BASELINE_MIGRATION",
    "OFFICIAL_BASELINE_RPCS",
    "OFFICIAL_BASELINE_RPC_CLOSE_UNIT",
    "OFFICIAL_BASELINE_RPC_LOAD_FRONTIER",
    "OFFICIAL_BASELINE_RPC_LOAD_REPLAY",
    "OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_KNOWN",
    "OFFICIAL_BASELINE_RPC_RECORD_TERMINAL_UNCERTAIN",
    "OFFICIAL_BASELINE_RPC_REGISTER_RUN",
    "OFFICIAL_BASELINE_RPC_RESERVE_ACTION",
    "OfficialBaselineStoreError",
    "SupabaseOfficialBaselineAttemptStore",
    "official_baseline_action_replay_identity",
    "official_baseline_terminal_store_outcome",
]
