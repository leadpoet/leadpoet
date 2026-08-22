"""Supabase authority adapters for Research Lab routing experiments.

The pure routing contract owns hashes, model plans, and receipt validation.
This module is the only durable gateway seam: it calls the append-only SQL
authority added in migration 157.  It never sends provider credentials or raw
provider payloads to Supabase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Callable, Iterable, Mapping, Protocol

from gateway.db.client import get_write_client
from gateway.research_lab.routing_experiment_attestation import (
    build_routing_experiment_attestation_input_v2,
    execute_routing_experiment_attestation_v2,
)
from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
    validate_routing_execution_envelope_v2,
)
from gateway.research_lab.routing_admission import (
    RoutingAdmissionBundleV2,
    RoutingAdmissionError,
)
from gateway.research_lab.routing_provider_terminal import (
    RoutingProviderTerminalError,
    validate_routing_provider_terminal_v2,
)
from leadpoet_canonical.attested_v2 import validate_signed_execution_receipt
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderOutcome,
    ProviderReceipt,
    RoutingDecisionReceiptV2,
    RoutingExperimentError,
    RoutingExperimentPromotionAuthority,
    RoutingExperimentV2Evaluation,
    RoutingExperimentV2Spec,
    provider_receipt_key,
    validate_provider_receipt,
    validate_routing_decision_receipt,
)


class RoutingExperimentStoreError(RuntimeError):
    """The service-role routing authority rejected or could not read a record."""


class RoutingExperimentEvaluationAttestor(Protocol):
    """Returns a persisted scoring-enclave receipt for exact evidence roots."""

    def attest(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RoutingExperimentExecutionClaim:
    """Public SQL claim identity held by the independent routing worker."""

    experiment_hash: str
    claim_key: str
    claim_generation: int
    # A non-secret deterministic fence hash binds the claim identity. It is
    # safe to include in protected authorization receipts.
    claim_fence_hash: str
    request_hash: str = ""
    lease_hash: str = ""
    lease_generation: int = 0
    worker_ref: str = ""
    lease_expires_at: str = ""

    def __post_init__(self) -> None:
        _require_hash(self.experiment_hash, "claim experiment_hash")
        _require_hash(self.claim_key, "claim claim_key")
        if type(self.claim_generation) is not int or self.claim_generation < 1:
            raise RoutingExperimentStoreError("claim generation is invalid")
        if self.claim_fence_hash != routing_claim_fence_hash_v3(
            experiment_hash=self.experiment_hash,
            claim_key=self.claim_key,
            claim_generation=self.claim_generation,
        ):
            raise RoutingExperimentStoreError("claim fence hash differs")
        queue_fields = (
            self.request_hash,
            self.lease_hash,
            self.lease_generation,
            self.worker_ref,
        )
        if any(queue_fields):
            _require_hash(self.request_hash, "claim request_hash")
            _require_hash(self.lease_hash, "claim lease_hash")
            if type(self.lease_generation) is not int or self.lease_generation < 1:
                raise RoutingExperimentStoreError("claim lease generation is invalid")
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", str(self.worker_ref or "")
            ):
                raise RoutingExperimentStoreError("claim worker ref is invalid")
            try:
                lease_expiry = datetime.fromisoformat(
                    str(self.lease_expires_at or "").replace("Z", "+00:00")
                )
            except (TypeError, ValueError) as exc:
                raise RoutingExperimentStoreError(
                    "claim lease expiry is invalid"
                ) from exc
            if lease_expiry.tzinfo is None or lease_expiry.utcoffset() is None:
                raise RoutingExperimentStoreError("claim lease expiry is invalid")
        elif self.lease_expires_at:
            raise RoutingExperimentStoreError(
                "claim lease expiry requires queue identity"
            )

    @property
    def claim_fence(self) -> str:
        """The non-secret claim fence used by protected authorization."""

        return self.claim_fence_hash


@dataclass(frozen=True)
class RoutingExecutionRequestLease:
    """Durable queue ownership, separate from the execution claim capability.

    The queue lease contains only hashes and bounded identifiers. It is bound
    into the execution claim key before any experiment claim RPC.
    """

    request_hash: str
    experiment_hash: str
    lease_hash: str
    worker_ref: str
    lease_generation: int
    lease_expires_at: str

    def __post_init__(self) -> None:
        _require_hash(self.request_hash, "execution request hash")
        _require_hash(self.experiment_hash, "execution request experiment hash")
        _require_hash(self.lease_hash, "execution request lease hash")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", str(self.worker_ref or "")
        ):
            raise RoutingExperimentStoreError("execution request worker ref is invalid")
        if type(self.lease_generation) is not int or self.lease_generation < 1:
            raise RoutingExperimentStoreError("execution request lease generation is invalid")
        if not str(self.lease_expires_at or "").strip():
            raise RoutingExperimentStoreError("execution request lease expiry is invalid")


@dataclass(frozen=True)
class RoutingExperimentExpiredBudgetReservation:
    """A stale open reservation that must be closed conservatively on resume."""

    reservation_id: str
    binding_id: str
    credit_microunits: int
    dispatch_started: bool

    def __post_init__(self) -> None:
        for field_name, value in (
            ("reservation id", self.reservation_id),
            ("binding id", self.binding_id),
        ):
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", str(value or "")
            ):
                raise RoutingExperimentStoreError(f"expired budget {field_name} is invalid")
        if type(self.credit_microunits) is not int or self.credit_microunits < 0:
            raise RoutingExperimentStoreError("expired budget credit is invalid")
        if type(self.dispatch_started) is not bool:
            raise RoutingExperimentStoreError("expired budget dispatch marker is invalid")


@dataclass(frozen=True)
class RoutingExperimentUnresolvedBudgetReservation:
    """One latest budget head that blocks a resumed provider run."""

    reservation_id: str
    binding_id: str
    credit_microunits: int
    event_type: str
    lease_expired: bool
    dispatch_started: bool

    def __post_init__(self) -> None:
        for field_name, value in (
            ("reservation id", self.reservation_id),
            ("binding id", self.binding_id),
        ):
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", str(value or "")
            ):
                raise RoutingExperimentStoreError(f"unresolved budget {field_name} is invalid")
        if type(self.credit_microunits) is not int or self.credit_microunits < 0:
            raise RoutingExperimentStoreError("unresolved budget credit is invalid")
        if self.event_type not in {"reserve", "uncertain", "recover"}:
            raise RoutingExperimentStoreError("unresolved budget state is invalid")
        if type(self.lease_expired) is not bool or type(self.dispatch_started) is not bool:
            raise RoutingExperimentStoreError("unresolved budget flags are invalid")


def _response_data(response: Any) -> Any:
    if isinstance(response, Mapping):
        return response.get("data")
    return getattr(response, "data", None)


def _require_hash(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise RoutingExperimentStoreError(f"{field_name} must be a sha256 digest")
    return normalized


def routing_claim_fence_hash_v3(
    *, experiment_hash: str, claim_key: str, claim_generation: int
) -> str:
    """Build the public, deterministic identity fence for one SQL claim."""

    normalized_experiment = _require_hash(experiment_hash, "claim experiment_hash")
    normalized_key = _require_hash(claim_key, "claim claim_key")
    if type(claim_generation) is not int or claim_generation < 1:
        raise RoutingExperimentStoreError("claim generation is invalid")
    return sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_claim_fence.v3",
            "experiment_hash": normalized_experiment,
            "claim_key": normalized_key,
            "claim_generation": claim_generation,
        }
    )


# Compatibility export for read-only callers.  New claims use the v3
# bearer-free identity above.
routing_claim_fence_hash_v2 = routing_claim_fence_hash_v3


def _claim_fence_rpc_params(
    claim: RoutingExperimentExecutionClaim,
) -> Mapping[str, Any]:
    """Return the complete non-secret SQL fence after local validation."""

    if claim.claim_fence_hash != routing_claim_fence_hash_v3(
        experiment_hash=claim.experiment_hash,
        claim_key=claim.claim_key,
        claim_generation=claim.claim_generation,
    ):
        raise RoutingExperimentStoreError("claim fence hash differs")
    return {
        "p_claim_key": claim.claim_key,
        "p_claim_generation": claim.claim_generation,
    }


def _event_hash(event_type: str, document: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_event.v2",
            "event_type": event_type,
            "document": dict(document),
        }
    )


class SupabaseRoutingExperimentStore:
    """Small synchronous client over the service-only routing authority RPCs."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        return self._client if self._client is not None else get_write_client()

    def _rpc(self, name: str, params: Mapping[str, Any]) -> Any:
        try:
            response = self.client.rpc(name, dict(params)).execute()
        except Exception as exc:  # noqa: BLE001 - preserve the authority boundary
            raise RoutingExperimentStoreError(
                f"routing authority RPC failed:{name}:{type(exc).__name__}"
            ) from exc
        return _response_data(response)

    def _select_one(
        self,
        table: str,
        *,
        column: str,
        value: str,
    ) -> Mapping[str, Any] | None:
        try:
            response = (
                self.client.table(table)
                .select("*")
                .eq(column, value)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            raise RoutingExperimentStoreError(
                f"routing authority select failed:{table}:{type(exc).__name__}"
            ) from exc
        data = _response_data(response) or []
        if not data:
            return None
        if not isinstance(data[0], Mapping):
            raise RoutingExperimentStoreError(f"routing authority select malformed:{table}")
        return dict(data[0])

    def _select_keys(self, table: str, *, column: str, experiment_hash: str) -> tuple[str, ...]:
        try:
            response = (
                self.client.table(table)
                .select(column)
                .eq("experiment_hash", experiment_hash)
                .order(column)
                .limit(10_000)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            raise RoutingExperimentStoreError(
                f"routing authority key select failed:{table}:{type(exc).__name__}"
            ) from exc
        values: list[str] = []
        for row in (_response_data(response) or []):
            if not isinstance(row, Mapping) or not isinstance(row.get(column), str):
                raise RoutingExperimentStoreError(f"routing authority key select malformed:{table}")
            values.append(str(row[column]))
        if len(values) != len(set(values)):
            raise RoutingExperimentStoreError(f"routing authority duplicate keys:{table}")
        return tuple(values)

    def _select_rows(
        self,
        table: str,
        *,
        experiment_hash: str,
        order_column: str,
    ) -> tuple[Mapping[str, Any], ...]:
        """Read one bounded authoritative set; fail rather than truncate it."""

        try:
            response = (
                self.client.table(table)
                .select("*")
                .eq("experiment_hash", experiment_hash)
                .order(order_column)
                .limit(10_001)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            raise RoutingExperimentStoreError(
                f"routing authority row select failed:{table}:{type(exc).__name__}"
            ) from exc
        rows = _response_data(response) or []
        if not isinstance(rows, list) or any(not isinstance(item, Mapping) for item in rows):
            raise RoutingExperimentStoreError(f"routing authority row select malformed:{table}")
        if len(rows) > 10_000:
            raise RoutingExperimentStoreError(f"routing authority row select exceeds bound:{table}")
        return tuple(dict(item) for item in rows)

    def submit(
        self,
        spec: RoutingExperimentV2Spec,
        *,
        execution_envelope: RoutingExperimentExecutionEnvelopeV2 | None = None,
    ) -> Mapping[str, Any]:
        experiment_hash = spec.experiment_hash()
        if execution_envelope is not None and execution_envelope.experiment_hash != experiment_hash:
            raise RoutingExperimentStoreError(
                "routing execution envelope belongs to another experiment"
            )
        if execution_envelope is not None:
            try:
                validate_routing_execution_envelope_v2(
                    spec=spec,
                    envelope=execution_envelope,
                )
            except ValueError as exc:
                raise RoutingExperimentStoreError(
                    "routing execution envelope is invalid"
                ) from exc
        if spec.allow_live_credit_spend and execution_envelope is None:
            raise RoutingExperimentStoreError(
                "routing live experiment requires an execution envelope"
            )
        event_doc = {
            "schema_version": "leadpoet.research_lab.routing_event.v2",
            "experiment_hash": experiment_hash,
            "event_type": "submitted",
        }
        return self._rpc(
            "research_lab_routing_submit_experiment_v2",
            {
                "p_experiment_hash": experiment_hash,
                "p_experiment_id": spec.experiment_id,
                "p_spec_doc": spec.to_dict(),
                "p_receipt_execution_mode": spec.receipt_execution_mode,
                "p_allow_live_credit_spend": spec.allow_live_credit_spend,
                "p_event_hash": _event_hash("submitted", event_doc),
                "p_event_doc": event_doc,
                "p_execution_envelope_hash": (
                    execution_envelope.envelope_hash()
                    if execution_envelope is not None
                    else None
                ),
                "p_execution_envelope_doc": (
                    execution_envelope.to_dict()
                    if execution_envelope is not None
                    else None
                ),
            },
        )

    def load_spec(self, experiment_hash: str) -> RoutingExperimentV2Spec | None:
        normalized_hash = _require_hash(experiment_hash, "experiment_hash")
        row = self._select_one(
            "research_lab_routing_experiments_v2",
            column="experiment_hash",
            value=normalized_hash,
        )
        if row is None:
            return None
        document = row.get("spec_doc")
        if not isinstance(document, Mapping):
            raise RoutingExperimentStoreError("stored routing spec document is malformed")
        spec = RoutingExperimentV2Spec.from_mapping(document)
        if spec.experiment_hash() != normalized_hash:
            raise RoutingExperimentStoreError("stored routing spec hash is inconsistent")
        return spec

    def request_execution(self, experiment_hash: str) -> Mapping[str, Any]:
        normalized = _require_hash(experiment_hash, "experiment_hash")
        document = {
            "schema_version": "leadpoet.research_lab.routing_execution_request.v2",
            "experiment_hash": normalized,
        }
        request_hash = sha256_json(document)
        result = self._rpc(
            "research_lab_routing_request_execution_v2",
            {
                "p_request_hash": request_hash,
                "p_experiment_hash": normalized,
                "p_request_doc": document,
            },
        )
        if (
            not isinstance(result, Mapping)
            or result.get("request_hash") != request_hash
            or result.get("experiment_hash") != normalized
        ):
            raise RoutingExperimentStoreError(
                "routing execution request result is malformed"
            )
        return dict(result)

    def execution_request(self, experiment_hash: str) -> Mapping[str, Any] | None:
        return self._select_one(
            "research_lab_routing_execution_requests_v2",
            column="experiment_hash",
            value=_require_hash(experiment_hash, "experiment_hash"),
        )

    def claim_pending_execution_requests(
        self,
        *,
        worker_ref: str,
        batch_size: int = 1,
        lease_seconds: int = 300,
    ) -> tuple[RoutingExecutionRequestLease, ...]:
        """Claim a bounded queue batch through the SKIP LOCKED RPC.

        The database owns row locking and generation fencing.  This adapter
        accepts no raw claim nonce, endpoint, credential, or provider data.
        """

        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", str(worker_ref or "")
        ):
            raise RoutingExperimentStoreError("execution request worker ref is invalid")
        if type(batch_size) is not int or batch_size < 1 or batch_size > 100:
            raise RoutingExperimentStoreError("execution request batch size is invalid")
        if type(lease_seconds) is not int or lease_seconds < 30 or lease_seconds > 3600:
            raise RoutingExperimentStoreError("execution request lease seconds are invalid")
        result = self._rpc(
            "research_lab_routing_claim_execution_requests_v2",
            {
                "p_worker_ref": worker_ref,
                "p_batch_size": batch_size,
                "p_lease_seconds": lease_seconds,
            },
        )
        if not isinstance(result, Mapping) or set(result) != {"requests"}:
            raise RoutingExperimentStoreError("execution request queue result is malformed")
        rows = result.get("requests")
        if not isinstance(rows, list) or len(rows) > batch_size:
            raise RoutingExperimentStoreError("execution request queue batch is malformed")
        leases: list[RoutingExecutionRequestLease] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "request_hash",
                "experiment_hash",
                "lease_hash",
                "worker_ref",
                "lease_generation",
                "lease_expires_at",
            }:
                raise RoutingExperimentStoreError("execution request lease row is malformed")
            lease = RoutingExecutionRequestLease(
                request_hash=str(row["request_hash"]),
                experiment_hash=str(row["experiment_hash"]),
                lease_hash=str(row["lease_hash"]),
                worker_ref=str(row["worker_ref"]),
                lease_generation=row["lease_generation"],
                lease_expires_at=str(row["lease_expires_at"]),
            )
            if lease.worker_ref != worker_ref:
                raise RoutingExperimentStoreError("execution request lease worker differs")
            leases.append(lease)
        return tuple(leases)

    def renew_execution_request_lease(
        self,
        *,
        lease: RoutingExecutionRequestLease,
        lease_seconds: int,
    ) -> Mapping[str, Any]:
        if type(lease_seconds) is not int or lease_seconds < 30 or lease_seconds > 3600:
            raise RoutingExperimentStoreError("execution request lease seconds are invalid")
        result = self._rpc(
            "research_lab_routing_renew_execution_request_lease_v2",
            {
                "p_request_hash": lease.request_hash,
                "p_worker_ref": lease.worker_ref,
                "p_lease_hash": lease.lease_hash,
                "p_lease_generation": lease.lease_generation,
                "p_lease_seconds": lease_seconds,
            },
        )
        if not isinstance(result, Mapping) or set(result) != {
            "renewed", "request_hash", "lease_generation", "lease_expires_at"
        }:
            raise RoutingExperimentStoreError("execution request renewal result is malformed")
        if (
            result.get("request_hash") != lease.request_hash
            or result.get("lease_generation") != lease.lease_generation
            or type(result.get("renewed")) is not bool
        ):
            raise RoutingExperimentStoreError("execution request renewal identity differs")
        return dict(result)

    def close_execution_request_lease(
        self,
        *,
        lease: RoutingExecutionRequestLease,
        close_reason: str,
    ) -> Mapping[str, Any]:
        if close_reason not in {"completed", "failed", "recovered"}:
            raise RoutingExperimentStoreError("execution request close reason is invalid")
        result = self._rpc(
            "research_lab_routing_close_execution_request_lease_v2",
            {
                "p_request_hash": lease.request_hash,
                "p_worker_ref": lease.worker_ref,
                "p_lease_hash": lease.lease_hash,
                "p_lease_generation": lease.lease_generation,
                "p_close_reason": close_reason,
            },
        )
        if not isinstance(result, Mapping) or set(result) not in ({
            "closed", "stale", "request_hash", "lease_generation"
        }, {
            "closed", "stale", "request_hash", "lease_generation", "close_reason"
        }):
            raise RoutingExperimentStoreError("execution request close result is malformed")
        if (
            result.get("request_hash") != lease.request_hash
            or result.get("lease_generation") != lease.lease_generation
            or type(result.get("closed")) is not bool
            or type(result.get("stale")) is not bool
        ):
            raise RoutingExperimentStoreError("execution request close identity differs")
        if result.get("closed") and result.get("close_reason") != close_reason:
            raise RoutingExperimentStoreError("execution request close reason differs")
        return dict(result)

    def append_event(
        self,
        *,
        experiment_hash: str,
        event_type: str,
        event_doc: Mapping[str, Any],
        claim: RoutingExperimentExecutionClaim,
    ) -> Mapping[str, Any]:
        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError("routing event claim belongs to another experiment")
        document = {
            "schema_version": "leadpoet.research_lab.routing_event.v2",
            **dict(event_doc),
        }
        params = {
                "p_event_hash": _event_hash(event_type, document),
                "p_experiment_hash": normalized_experiment_hash,
                "p_event_type": event_type,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_event_doc": document,
            }
        result = self._rpc("research_lab_routing_append_fenced_event_v3", params)
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing event result is malformed")
        return dict(result)

    def load_model_transition_marker(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        idempotency_key: str,
    ) -> Mapping[str, Any] | None:
        """Read one redacted paid-call/recovery marker, never a provider body."""

        normalized_experiment_hash = _require_hash(
            experiment_hash, "experiment_hash"
        )
        if not re.fullmatch(r"[0-9a-f]{64}", str(idempotency_key or "")):
            raise RoutingExperimentStoreError(
                "Model transition idempotency key is invalid"
            )
        matches: list[Mapping[str, Any]] = []
        for row in self._select_rows(
            "research_lab_routing_experiment_events_v2",
            experiment_hash=normalized_experiment_hash,
            order_column="created_at",
        ):
            if row.get("event_type") != "model_transition_completed":
                continue
            document = row.get("event_doc")
            if not isinstance(document, Mapping):
                raise RoutingExperimentStoreError(
                    "Model transition marker is malformed"
                )
            if (
                document.get("variant_id") == variant_id
                and document.get("unit_ref") == unit_ref
                and document.get("idempotency_key") == idempotency_key
            ):
                matches.append(dict(document))
        if not matches:
            return None
        if len(matches) != 1:
            raise RoutingExperimentStoreError(
                "Model transition marker is duplicated"
            )
        marker = matches[0]
        expected = {
            "schema_version",
            "event_schema_version",
            "variant_id",
            "unit_ref",
            "idempotency_key",
            "action_sha256",
            "continuation_sha256",
            "completion_sha256",
            "provider_response_sha256",
            "provider_receipt",
            "protected_dispatch_job_id",
            "terminal_receipt_hash",
            "model_completion_contract_hash",
        }
        if (
            set(marker) != expected
            or marker.get("schema_version")
            != "leadpoet.research_lab.routing_event.v2"
            or marker.get("event_schema_version")
            != "leadpoet.research_lab.model_transition.v1"
        ):
            raise RoutingExperimentStoreError(
                "Model transition marker fields are malformed"
            )
        for name in (
            "continuation_sha256",
            "provider_response_sha256",
        ):
            _require_hash(marker.get(name), name)
        for name in ("action_sha256", "completion_sha256"):
            value = str(marker.get(name) or "")
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise RoutingExperimentStoreError(
                    f"Model transition {name} is invalid"
                )
        raw_receipt = marker.get("provider_receipt")
        if raw_receipt is not None:
            try:
                receipt = ProviderReceipt.from_mapping(raw_receipt)
            except (RoutingExperimentError, TypeError, ValueError) as exc:
                raise RoutingExperimentStoreError(
                    "Model transition provider receipt is invalid"
                ) from exc
            errors = validate_provider_receipt(receipt)
            if errors:
                raise RoutingExperimentStoreError(
                    "Model transition provider receipt is invalid"
                )
            if (
                not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}",
                    str(marker.get("protected_dispatch_job_id") or ""),
                )
                or any(
                    not re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(marker.get(name) or ""),
                    )
                    for name in (
                        "terminal_receipt_hash",
                        "model_completion_contract_hash",
                    )
                )
            ):
                raise RoutingExperimentStoreError(
                    "Model transition protected replay reference is invalid"
                )
        elif any(
            marker.get(name) is not None
            for name in (
                "protected_dispatch_job_id",
                "terminal_receipt_hash",
                "model_completion_contract_hash",
            )
        ):
            raise RoutingExperimentStoreError(
                "Model verifier transition has protected replay state"
            )
        return marker

    def append_adapter_failure(
        self,
        *,
        experiment_hash: str,
        key: str,
        receipt: ProviderReceipt,
        variant_id: str,
        claim: RoutingExperimentExecutionClaim,
    ) -> Mapping[str, Any]:
        """Persist a zero-cost failure proven before provider dispatch.

        This is deliberately a separate authority path from provider attempts:
        no terminal, protected-release, admission, or billing proof is
        synthesized for a failure raised before dispatch.
        """

        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "routing adapter failure claim belongs to another experiment"
            )
        errors = validate_provider_receipt(receipt)
        if errors:
            raise RoutingExperimentStoreError(
                "invalid routing adapter failure receipt: " + "; ".join(errors)
            )
        if receipt.outcome != ProviderOutcome.ADAPTER_FAILURE.value:
            raise RoutingExperimentStoreError(
                "routing adapter failure receipt outcome is invalid"
            )
        if receipt.credit_microunits != 0:
            raise RoutingExperimentStoreError(
                "routing adapter failure receipt must be zero cost"
            )
        expected_key = provider_receipt_key(
            tool_id=receipt.tool_id,
            binding_version=receipt.binding_version,
            request_fingerprint=receipt.request_fingerprint,
        )
        if str(key) != expected_key:
            raise RoutingExperimentStoreError("routing adapter failure key mismatch")
        normalized_variant_id = str(variant_id or "")
        if not normalized_variant_id:
            raise RoutingExperimentStoreError(
                "routing adapter failure variant identity is invalid"
            )
        failure_doc = {
            "schema_version": "leadpoet.research_lab.routing_adapter_failure.v3",
            "failure_key": str(key),
            "experiment_hash": normalized_experiment_hash,
            "binding_id": receipt.binding_id,
            "tool_id": receipt.tool_id,
            "variant_id": normalized_variant_id,
            "unit_ref": receipt.unit_ref,
            "claim_key": claim.claim_key,
            "claim_generation": claim.claim_generation,
            "request_fingerprint": receipt.request_fingerprint,
            "outcome": ProviderOutcome.ADAPTER_FAILURE.value,
            "credit_microunits": 0,
            "latency_ms": receipt.latency_ms,
            "execution_mode": receipt.execution_mode,
            "pre_dispatch": True,
            "provider_receipt": receipt.to_dict(),
        }
        result = self._rpc(
            "research_lab_routing_append_adapter_failure_v3",
            {
                "p_failure_key": str(key),
                "p_experiment_hash": normalized_experiment_hash,
                "p_provider_receipt_ref": receipt.receipt_ref,
                "p_binding_id": receipt.binding_id,
                "p_tool_id": receipt.tool_id,
                "p_variant_id": normalized_variant_id,
                "p_unit_ref": receipt.unit_ref,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_request_fingerprint": receipt.request_fingerprint,
                "p_latency_ms": receipt.latency_ms,
                "p_execution_mode": receipt.execution_mode,
                "p_failure_doc": failure_doc,
            },
        )
        if (
            not isinstance(result, Mapping)
            or result.get("failure_key") != str(key)
            or type(result.get("idempotent")) is not bool
        ):
            raise RoutingExperimentStoreError(
                "routing adapter failure append result is malformed"
            )
        return dict(result)

    def claim_execution(
        self,
        *,
        experiment_hash: str,
        claim_key: str,
        request_hash: str,
        lease_hash: str,
        lease_generation: int,
        worker_ref: str,
        lease_seconds: int,
        claim_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        event_doc = {
            "schema_version": "leadpoet.research_lab.routing_event.v2",
            "experiment_hash": experiment_hash,
            "request_hash": request_hash,
            "lease_hash": lease_hash,
            "lease_generation": int(lease_generation),
            "claim_key": claim_key,
            "worker_ref": worker_ref,
            "event_type": "claimed",
        }
        result = self._rpc(
            "research_lab_routing_claim_execution_v3",
            {
                "p_request_hash": request_hash,
                "p_lease_hash": lease_hash,
                "p_lease_generation": int(lease_generation),
                "p_worker_ref": worker_ref,
                "p_claim_key": claim_key,
                "p_claim_lease_seconds": int(lease_seconds),
                "p_claim_doc": dict(claim_doc),
                "p_event_hash": _event_hash("claimed", event_doc),
                "p_event_doc": event_doc,
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing claim result is malformed")
        return dict(result)

    # Kept as a non-bearer compatibility alias for test doubles and older
    # callers.  Production code uses claim_execution so the queue lease is
    # explicit at the call site.
    claim = claim_execution

    def recover_claim(
        self,
        *,
        experiment_hash: str,
        recovery_key: str,
        worker_ref: str,
        recovery_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        event_doc = {
            "schema_version": "leadpoet.research_lab.routing_event.v2",
            "experiment_hash": experiment_hash,
            "recovery_key": recovery_key,
            "worker_ref": worker_ref,
            "event_type": "claim_recovered",
        }
        result = self._rpc(
            "research_lab_routing_recover_claim_v3",
            {
                "p_experiment_hash": experiment_hash,
                "p_recovery_key": recovery_key,
                "p_worker_ref": worker_ref,
                "p_recovery_doc": dict(recovery_doc),
                "p_event_hash": _event_hash("claim_recovered", event_doc),
                "p_event_doc": event_doc,
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing claim recovery result is malformed")
        return dict(result)

    def renew_claim(
        self,
        *,
        experiment_hash: str,
        claim: RoutingExperimentExecutionClaim,
        heartbeat_key: str,
        lease_seconds: int,
        heartbeat_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Extend a claim lease through the bearer-free v3 authority."""

        if claim.experiment_hash != _require_hash(experiment_hash, "experiment_hash"):
            raise RoutingExperimentStoreError("routing heartbeat claim belongs to another experiment")
        result = self._rpc(
            "research_lab_routing_renew_claim_v3",
            {
                "p_heartbeat_key": _require_hash(heartbeat_key, "heartbeat key"),
                "p_experiment_hash": claim.experiment_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_lease_seconds": int(lease_seconds),
                "p_heartbeat_doc": dict(heartbeat_doc),
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing claim renewal result is malformed")
        return dict(result)

    def close_claim(
        self,
        *,
        experiment_hash: str,
        claim: RoutingExperimentExecutionClaim,
        close_key: str,
        close_reason: str,
        close_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Close a claim through the bearer-free v3 authority."""

        if claim.experiment_hash != _require_hash(experiment_hash, "experiment_hash"):
            raise RoutingExperimentStoreError("routing close claim belongs to another experiment")
        if close_reason not in {"completed", "failed", "cancelled"}:
            raise RoutingExperimentStoreError("routing claim close reason is invalid")
        result = self._rpc(
            "research_lab_routing_close_claim_v3",
            {
                "p_close_key": _require_hash(close_key, "close key"),
                "p_experiment_hash": claim.experiment_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_close_reason": close_reason,
                "p_close_doc": dict(close_doc),
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing claim close result is malformed")
        return dict(result)

    def append_provider_attempt(
        self,
        *,
        experiment_hash: str,
        key: str,
        receipt: ProviderReceipt,
        variant_id: str,
        reservation_id: str,
        action_id: str,
        authorization: RoutingProviderCallAuthorizationV2,
        authorization_proof_hash: str,
        authorization_receipt: Mapping[str, Any],
        claim: RoutingExperimentExecutionClaim,
        terminal_proof: Mapping[str, Any] | None = None,
        protected_release_receipt: Mapping[str, Any] | None = None,
        admission_bundle: Mapping[str, Any] | RoutingAdmissionBundleV2 | None = None,
        billing_state: str | None = None,
        authoritative_billed_credit_microunits: int | None = None,
    ) -> Mapping[str, Any]:
        errors = validate_provider_receipt(receipt)
        if errors:
            raise RoutingExperimentStoreError("invalid routing provider receipt: " + "; ".join(errors))
        if not isinstance(terminal_proof, Mapping):
            raise RoutingExperimentStoreError(
                "signed routing provider terminal proof is required"
            )
        if not isinstance(protected_release_receipt, Mapping):
            raise RoutingExperimentStoreError(
                "signed routing provider protected release receipt is required"
            )
        if isinstance(admission_bundle, RoutingAdmissionBundleV2):
            admission_document = admission_bundle.to_dict()
        elif isinstance(admission_bundle, Mapping):
            admission_document = dict(admission_bundle)
        else:
            raise RoutingExperimentStoreError(
                "routing provider admission bundle is required"
            )
        try:
            admission = RoutingAdmissionBundleV2.from_mapping(admission_document)
        except (TypeError, ValueError, RoutingAdmissionError) as exc:
            raise RoutingExperimentStoreError(
                "routing provider admission bundle is invalid"
            ) from exc
        # The pure receipt preserves a known provider charge so metrics remain
        # honest.  The SQL ledger keeps synthetic adapter failures at zero and
        # records a separate broker-authoritative bill; an unknown failure is
        # marked uncertain and must retain its reservation.
        adapter_failure = receipt.outcome == ProviderOutcome.ADAPTER_FAILURE.value
        normalized_billing_state = str(billing_state or "").strip()
        if not normalized_billing_state:
            normalized_billing_state = "uncertain" if adapter_failure else "known"
        billed_credit: int | None = authoritative_billed_credit_microunits
        if billed_credit is None and normalized_billing_state == "known":
            billed_credit = receipt.credit_microunits
        stored_credit = receipt.credit_microunits
        if adapter_failure:
            stored_credit = 0
        if normalized_billing_state not in {"known", "uncertain"}:
            raise RoutingExperimentStoreError("routing provider billing state is invalid")
        if normalized_billing_state == "uncertain":
            billed_credit = None
        elif type(billed_credit) is not int or billed_credit < 0:
            raise RoutingExperimentStoreError("routing provider billed credit is invalid")
        if not adapter_failure and billed_credit != receipt.credit_microunits:
            raise RoutingExperimentStoreError("routing provider bill differs from receipt")
        if claim.experiment_hash != _require_hash(experiment_hash, "experiment_hash"):
            raise RoutingExperimentStoreError("routing provider claim belongs to another experiment")
        if (
            authorization.experiment_hash != experiment_hash
            or authorization.variant_id != variant_id
            or authorization.binding.binding_id != receipt.binding_id
            or authorization.binding.tool_id != receipt.tool_id
            or authorization.unit_ref != receipt.unit_ref
            or authorization.core_request_fingerprint != receipt.request_fingerprint
            or authorization.action_id != action_id
        ):
            raise RoutingExperimentStoreError(
                "routing provider authorization differs from the receipt"
            )
        proof_hash = _require_hash(
            authorization_proof_hash, "routing authorization proof hash"
        )
        if (
            admission.identity_hash() != authorization.admission_bundle_hash
            or admission.experiment_hash != experiment_hash
            or admission.experiment_id != authorization.experiment_id
            or admission.job_id != authorization.job_id
            or admission.protected_release_hash != authorization.protected_release_hash
            or admission.protected_boot_identity_hash
            != authorization.protected_boot_identity_hash
            or admission.protected_receipt_hash
            != protected_release_receipt.get("receipt_hash")
        ):
            raise RoutingExperimentStoreError(
                "routing provider admission authority differs"
            )
        try:
            validate_signed_execution_receipt(protected_release_receipt)
        except Exception as exc:
            raise RoutingExperimentStoreError(
                "routing provider protected release receipt signature is invalid"
            ) from exc
        if (
            protected_release_receipt.get("role") != admission.role
            or protected_release_receipt.get("purpose") != admission.purpose
            or protected_release_receipt.get("status") != "succeeded"
            or protected_release_receipt.get("job_id") != admission.job_id
            or protected_release_receipt.get("receipt_hash") != admission.protected_receipt_hash
            or protected_release_receipt.get("commit_sha")
            != admission.protected_commit_sha
            or protected_release_receipt.get("pcr0") != admission.protected_pcr0
            or protected_release_receipt.get("build_manifest_hash")
            != admission.protected_build_manifest_hash
            or protected_release_receipt.get("dependency_lock_hash")
            != admission.protected_dependency_lock_hash
            or protected_release_receipt.get("config_hash")
            != admission.protected_config_hash
            or protected_release_receipt.get("boot_identity_hash")
            != admission.protected_boot_identity_hash
            or protected_release_receipt.get("enclave_pubkey")
            != admission.protected_enclave_pubkey
        ):
            raise RoutingExperimentStoreError(
                "routing provider protected release identity differs"
            )
        if any(
            authorization_receipt.get(name)
            != protected_release_receipt.get(name)
            or terminal_execution_receipt.get(name)
            != protected_release_receipt.get(name)
            for name in ("boot_identity_hash", "enclave_pubkey")
        ):
            raise RoutingExperimentStoreError(
                "routing provider signed receipt signer identity differs"
            )
        try:
            projected_terminal = validate_routing_provider_terminal_v2(
                terminal=terminal_proof,
                binding=authorization.binding,
                protected_receipt=protected_release_receipt,
                expected_job_id=authorization.job_id,
                expected_experiment_hash=experiment_hash,
                expected_admission_bundle_hash=admission.identity_hash(),
                expected_authorization_hash=authorization.authorization_hash(),
                expected_authorization_proof_hash=proof_hash,
            )
        except (RoutingProviderTerminalError, TypeError, ValueError) as exc:
            raise RoutingExperimentStoreError(
                "routing provider terminal proof is invalid"
            ) from exc
        if projected_terminal != receipt.to_dict():
            raise RoutingExperimentStoreError(
                "routing provider terminal receipt differs"
            )
        terminal_body = terminal_proof.get("body")
        terminal_receipt = terminal_proof.get("receipt")
        if not isinstance(terminal_body, Mapping) or not isinstance(
            terminal_receipt, Mapping
        ):
            raise RoutingExperimentStoreError(
                "routing provider terminal proof fields are invalid"
            )
        terminal_receipt_hash = _require_hash(
            terminal_receipt.get("receipt_hash"),
            "routing terminal receipt hash",
        )
        terminal_provider_record_hash = _require_hash(
            terminal_body.get("provider_record_hash"),
            "routing terminal provider record hash",
        )
        terminal_billing_projection_hash = _require_hash(
            terminal_body.get("billing_projection_hash"),
            "routing terminal billing projection hash",
        )
        protected_release_receipt_hash = _require_hash(
            protected_release_receipt.get("receipt_hash"),
            "routing protected release receipt hash",
        )
        admission_bundle_hash = admission.identity_hash()
        try:
            validate_signed_execution_receipt(authorization_receipt)
        except Exception as exc:
            raise RoutingExperimentStoreError(
                "routing provider authorization receipt signature is invalid"
            ) from exc
        expected_authorization = execute_routing_provider_call_authorization_v2(
            authorization.to_dict()
        )
        if (
            authorization_receipt.get("receipt_hash") != proof_hash
            or authorization_receipt.get("role") != "gateway_scoring"
            or authorization_receipt.get("purpose")
                != "research_lab.routing_provider_evidence.v2"
            or authorization_receipt.get("status") != "succeeded"
            or authorization_receipt.get("input_root")
                != authorization.authorization_hash()
            or authorization_receipt.get("output_root")
                != expected_authorization["output_root"]
        ):
            raise RoutingExperimentStoreError(
                "routing provider authorization receipt is not exact"
            )
        # Migration 157's V3 RPC has a separate request-root argument.  The
        # legacy host path is retained only for fixture/replay callers, but it
        # must still send the exact named argument whenever it reaches the
        # durable boundary.  In this compatibility shape the signed receipt's
        # input root is the only authoritative request-root source.
        authorization_request_hash = _require_hash(
            authorization_receipt.get("input_root"),
            "routing authorization request hash",
        )
        attempt_doc = {
            "schema_version": "leadpoet.research_lab.routing_provider_attempt.v2",
            "binding_id": receipt.binding_id,
            "tool_id": receipt.tool_id,
            "action_id": action_id,
            "binding_catalog_manifest_hash": (
                authorization.binding_catalog_manifest_hash
            ),
            "call_grant_hash": authorization.authorization_hash(),
            "call_grant_proof_hash": proof_hash,
            "authorization_request_hash": authorization_request_hash,
            "request_body_hash": authorization.request_body_hash,
            "variant_id": str(variant_id),
            "unit_ref": receipt.unit_ref,
            "reservation_id": str(reservation_id),
            "request_fingerprint": receipt.request_fingerprint,
            "execution_mode": receipt.execution_mode,
            "provider_receipt": receipt.to_dict(),
            "call_grant": authorization.to_dict(),
            "call_grant_result": expected_authorization,
            "call_grant_receipt": dict(authorization_receipt),
            "terminal_proof": dict(terminal_proof),
            "protected_release_receipt": dict(protected_release_receipt),
            "admission_bundle": admission.to_dict(),
        }
        return self._rpc(
            "research_lab_routing_append_provider_attempt_v3",
            {
                "p_attempt_key": key,
                "p_experiment_hash": experiment_hash,
                "p_provider_receipt_ref": receipt.receipt_ref,
                "p_binding_id": receipt.binding_id,
                "p_tool_id": receipt.tool_id,
                "p_variant_id": str(variant_id),
                "p_unit_ref": receipt.unit_ref,
                "p_reservation_id": str(reservation_id),
                "p_action_id": str(action_id),
                "p_binding_catalog_manifest_hash": (
                    authorization.binding_catalog_manifest_hash
                ),
                "p_authorization_hash": authorization.authorization_hash(),
                "p_authorization_request_hash": authorization_request_hash,
                "p_authorization_proof_hash": proof_hash,
                "p_request_body_hash": authorization.request_body_hash,
                "p_terminal_receipt_hash": terminal_receipt_hash,
                "p_protected_release_receipt_hash": (
                    protected_release_receipt_hash
                ),
                "p_admission_bundle_hash": admission_bundle_hash,
                "p_terminal_provider_record_hash": (
                    terminal_provider_record_hash
                ),
                "p_terminal_billing_projection_hash": (
                    terminal_billing_projection_hash
                ),
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_request_fingerprint": receipt.request_fingerprint,
                "p_outcome": receipt.outcome,
                "p_credit_microunits": stored_credit,
                "p_latency_ms": receipt.latency_ms,
                "p_execution_mode": receipt.execution_mode,
                "p_billing_state": normalized_billing_state,
                "p_authoritative_billed_credit_microunits": billed_credit,
                "p_attempt_doc": attempt_doc,
            },
        )

    def append_protected_provider_attempt(
        self,
        *,
        experiment_hash: str,
        key: str,
        receipt: ProviderReceipt,
        variant_id: str,
        reservation_id: str,
        action_id: str,
        authorization: RoutingProviderCallAuthorizationV2,
        authorization_proof_hash: str,
        authorization_request_hash: str,
        authorization_receipt: Mapping[str, Any],
        terminal_result: Mapping[str, Any],
        terminal_execution_receipt: Mapping[str, Any],
        protected_release_receipt: Mapping[str, Any],
        admission_bundle: Mapping[str, Any] | RoutingAdmissionBundleV2,
        claim: RoutingExperimentExecutionClaim,
        billing_state: str,
        authoritative_billed_credit_microunits: int | None,
    ) -> Mapping[str, Any]:
        """Persist one provider attempt rooted in standard signed receipts.

        The admission job, authorization job, and terminal job are separate
        identities.  The authorization job comes only from the signed
        authorization receipt.  The terminal job is a new standard receipt
        whose parent is that authorization receipt; no host terminal proof is
        accepted here.
        """

        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "routing provider claim belongs to another experiment"
            )
        if type(authorization) is not RoutingProviderCallAuthorizationV2:
            raise RoutingExperimentStoreError("routing provider authorization is invalid")
        errors = validate_provider_receipt(receipt)
        if errors:
            raise RoutingExperimentStoreError(
                "invalid routing provider receipt: " + "; ".join(errors)
            )
        if not isinstance(admission_bundle, (RoutingAdmissionBundleV2, Mapping)):
            raise RoutingExperimentStoreError(
                "routing provider admission bundle is required"
            )
        try:
            admission = (
                admission_bundle
                if isinstance(admission_bundle, RoutingAdmissionBundleV2)
                else RoutingAdmissionBundleV2.from_mapping(admission_bundle)
            )
        except (TypeError, ValueError, RoutingAdmissionError) as exc:
            raise RoutingExperimentStoreError(
                "routing provider admission bundle is invalid"
            ) from exc
        if (
            authorization.experiment_hash != normalized_experiment_hash
            or authorization.variant_id != str(variant_id)
            or authorization.action_id != str(action_id)
            or authorization.binding.binding_id != receipt.binding_id
            or authorization.binding.tool_id != receipt.tool_id
            or authorization.unit_ref != receipt.unit_ref
            or authorization.core_request_fingerprint != receipt.request_fingerprint
            or admission.experiment_hash != normalized_experiment_hash
            or admission.experiment_id != authorization.experiment_id
            or admission.job_id != authorization.admission_job_id
            or admission.identity_hash() != authorization.admission_bundle_hash
            or admission.protected_release_hash != authorization.protected_release_hash
            or admission.protected_boot_identity_hash
            != authorization.protected_boot_identity_hash
            or admission.protected_receipt_hash
            != protected_release_receipt.get("receipt_hash")
        ):
            raise RoutingExperimentStoreError(
                "routing provider authorization or admission identity differs"
            )
        for document, message in (
            (authorization_receipt, "authorization receipt"),
            (protected_release_receipt, "protected release receipt"),
            (terminal_execution_receipt, "terminal execution receipt"),
        ):
            if not isinstance(document, Mapping):
                raise RoutingExperimentStoreError(
                    f"routing provider {message} is unavailable"
                )
            try:
                validate_signed_execution_receipt(document)
            except Exception as exc:  # noqa: BLE001 - signed boundary
                raise RoutingExperimentStoreError(
                    f"routing provider {message} signature is invalid"
                ) from exc
        authorization_job_id = str(authorization_receipt.get("job_id") or "")
        authorization_proof_hash = _require_hash(
            authorization_proof_hash, "routing authorization proof hash"
        )
        authorization_request_hash = _require_hash(
            authorization_request_hash, "routing authorization request hash"
        )
        authorization_input_root = _require_hash(
            authorization_receipt.get("input_root"),
            "routing authorization input root",
        )
        if (
            authorization_receipt.get("receipt_hash") != authorization_proof_hash
            or authorization_receipt.get("role") != "gateway_scoring"
            or authorization_receipt.get("purpose") != authorization.purpose
            or authorization_receipt.get("status") != "succeeded"
            or authorization_receipt.get("job_id") != authorization_job_id
            or authorization_receipt.get("input_root") != authorization_request_hash
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", authorization_job_id
            )
        ):
            raise RoutingExperimentStoreError(
                "routing provider authorization receipt identity differs"
            )
        try:
            expected_authorization = execute_routing_provider_call_authorization_v2(
                authorization.to_dict(),
                authorization_job_id=authorization_job_id,
            )
        except Exception as exc:
            raise RoutingExperimentStoreError(
                "routing provider authorization result is invalid"
            ) from exc
        if authorization_receipt.get("output_root") != expected_authorization["output_root"]:
            raise RoutingExperimentStoreError(
                "routing provider authorization receipt output differs"
            )
        if not isinstance(terminal_result, Mapping):
            raise RoutingExperimentStoreError(
                "routing provider terminal result is unavailable"
            )
        if terminal_result.get("provider_receipt") != receipt.to_dict():
            raise RoutingExperimentStoreError(
                "routing provider terminal result receipt differs"
            )
        projection = terminal_result.get("projection")
        if not isinstance(projection, Mapping) or projection.get("billing_state") != "known":
            raise RoutingExperimentStoreError(
                "routing provider terminal billing projection is invalid"
            )
        if (
            projection.get("outcome") != receipt.outcome
            or projection.get("evidence_hash") != receipt.evidence_hash
            or projection.get("credit_microunits") != receipt.credit_microunits
            or projection.get("latency_ms") != receipt.latency_ms
            or projection.get("call_count") != receipt.call_count
            or projection.get("binding_id") != receipt.binding_id
            or projection.get("tool_id") != receipt.tool_id
            or projection.get("request_fingerprint") != receipt.request_fingerprint
        ):
            raise RoutingExperimentStoreError(
                "routing provider terminal projection differs"
            )
        terminal_receipt_hash = _require_hash(
            terminal_execution_receipt.get("receipt_hash"),
            "routing terminal execution receipt hash",
        )
        terminal_input_root = _require_hash(
            terminal_execution_receipt.get("input_root"),
            "routing terminal execution input root",
        )
        expected_terminal_job_id = routing_provider_dispatch_job_id_v2(
            {
                "authorization_hash": authorization.authorization_hash(),
                "authorization_proof_hash": authorization_proof_hash,
                "authorization_receipt": authorization_receipt,
            }
        )
        if (
            terminal_execution_receipt.get("receipt_hash") != terminal_receipt_hash
            or terminal_execution_receipt.get("role") != "gateway_scoring"
            or terminal_execution_receipt.get("purpose") != authorization.purpose
            or terminal_execution_receipt.get("status") != "succeeded"
            or terminal_execution_receipt.get("job_id")
            != expected_terminal_job_id
            or terminal_input_root == authorization_input_root
            or terminal_execution_receipt.get("output_root") != sha256_json(dict(terminal_result))
            or terminal_execution_receipt.get("parent_receipt_hashes")
            != [authorization_proof_hash]
        ):
            raise RoutingExperimentStoreError(
                "routing provider terminal execution receipt identity differs"
            )
        protected_receipt_hash = _require_hash(
            protected_release_receipt.get("receipt_hash"),
            "routing protected release receipt hash",
        )
        if (
            protected_release_receipt.get("role") != admission.role
            or protected_release_receipt.get("purpose") != admission.purpose
            or protected_release_receipt.get("status") != "succeeded"
            or protected_release_receipt.get("job_id") != admission.job_id
            or protected_release_receipt.get("receipt_hash") != protected_receipt_hash
            or protected_release_receipt.get("commit_sha")
            != admission.protected_commit_sha
            or protected_release_receipt.get("pcr0") != admission.protected_pcr0
            or protected_release_receipt.get("build_manifest_hash")
            != admission.protected_build_manifest_hash
            or protected_release_receipt.get("dependency_lock_hash")
            != admission.protected_dependency_lock_hash
            or protected_release_receipt.get("config_hash")
            != admission.protected_config_hash
            or protected_release_receipt.get("boot_identity_hash")
            != admission.protected_boot_identity_hash
            or protected_release_receipt.get("enclave_pubkey")
            != admission.protected_enclave_pubkey
        ):
            raise RoutingExperimentStoreError(
                "routing provider protected release identity differs"
            )
        normalized_billing_state = str(billing_state or "")
        if normalized_billing_state != "known":
            raise RoutingExperimentStoreError(
                "routing provider protected terminal billing must be known"
            )
        if (
            type(authoritative_billed_credit_microunits) is not int
            or authoritative_billed_credit_microunits != receipt.credit_microunits
        ):
            raise RoutingExperimentStoreError(
                "routing provider authoritative billing differs"
            )
        attempt_doc = {
            "schema_version": "leadpoet.research_lab.routing_provider_attempt.v3",
            "binding_id": receipt.binding_id,
            "tool_id": receipt.tool_id,
            "action_id": str(action_id),
            "binding_catalog_manifest_hash": authorization.binding_catalog_manifest_hash,
            "call_grant_hash": authorization.authorization_hash(),
            "call_grant_proof_hash": authorization_proof_hash,
            "authorization_request_hash": authorization_request_hash,
            "request_body_hash": authorization.request_body_hash,
            "variant_id": str(variant_id),
            "unit_ref": receipt.unit_ref,
            "reservation_id": str(reservation_id),
            "request_fingerprint": receipt.request_fingerprint,
            "execution_mode": receipt.execution_mode,
            "provider_receipt": receipt.to_dict(),
            "call_grant": authorization.to_dict(),
            "call_grant_result": expected_authorization,
            "call_grant_receipt": dict(authorization_receipt),
            "terminal_request_hash": terminal_input_root,
            "terminal_result": dict(terminal_result),
            "terminal_execution_receipt": dict(terminal_execution_receipt),
            "protected_release_receipt": dict(protected_release_receipt),
            "admission_bundle": admission.to_dict(),
        }
        return self._rpc(
            "research_lab_routing_append_provider_attempt_v3",
            {
                "p_attempt_key": key,
                "p_experiment_hash": normalized_experiment_hash,
                "p_provider_receipt_ref": receipt.receipt_ref,
                "p_binding_id": receipt.binding_id,
                "p_tool_id": receipt.tool_id,
                "p_variant_id": str(variant_id),
                "p_unit_ref": receipt.unit_ref,
                "p_reservation_id": str(reservation_id),
                "p_action_id": str(action_id),
                "p_binding_catalog_manifest_hash": authorization.binding_catalog_manifest_hash,
                "p_authorization_hash": authorization.authorization_hash(),
                "p_authorization_request_hash": authorization_request_hash,
                "p_authorization_proof_hash": authorization_proof_hash,
                "p_request_body_hash": authorization.request_body_hash,
                "p_terminal_receipt_hash": terminal_receipt_hash,
                "p_protected_release_receipt_hash": protected_receipt_hash,
                "p_admission_bundle_hash": admission.identity_hash(),
                "p_terminal_provider_record_hash": _require_hash(
                    terminal_result.get("provider_record_hash"),
                    "routing terminal provider record hash",
                ),
                "p_terminal_billing_projection_hash": sha256_json(dict(projection)),
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_terminal_request_hash": terminal_input_root,
                "p_terminal_result_hash": sha256_json(dict(terminal_result)),
                "p_request_fingerprint": receipt.request_fingerprint,
                "p_outcome": receipt.outcome,
                "p_credit_microunits": receipt.credit_microunits,
                "p_latency_ms": receipt.latency_ms,
                "p_execution_mode": receipt.execution_mode,
                "p_billing_state": normalized_billing_state,
                "p_authoritative_billed_credit_microunits": authoritative_billed_credit_microunits,
                "p_attempt_doc": attempt_doc,
            },
        )

    def provider_attempt_row(self, key: str) -> Mapping[str, Any] | None:
        return self._select_one(
            "research_lab_routing_provider_attempts_v2",
            column="attempt_key",
            value=key,
        )

    def provider_attempt_keys(self, experiment_hash: str) -> tuple[str, ...]:
        return self._select_keys(
            "research_lab_routing_provider_attempts_v2",
            column="attempt_key",
            experiment_hash=experiment_hash,
        )

    def adapter_failure_row(self, key: str) -> Mapping[str, Any] | None:
        return self._select_one(
            "research_lab_routing_adapter_failures_v2",
            column="failure_key",
            value=key,
        )

    def adapter_failure_keys(self, experiment_hash: str) -> tuple[str, ...]:
        return self._select_keys(
            "research_lab_routing_adapter_failures_v2",
            column="failure_key",
            experiment_hash=experiment_hash,
        )

    def append_decision(
        self,
        *,
        experiment_hash: str,
        receipt: RoutingDecisionReceiptV2,
        claim: RoutingExperimentExecutionClaim,
    ) -> Mapping[str, Any]:
        errors = validate_routing_decision_receipt(receipt)
        if errors:
            raise RoutingExperimentStoreError("invalid routing decision receipt: " + "; ".join(errors))
        if claim.experiment_hash != _require_hash(experiment_hash, "experiment_hash"):
            raise RoutingExperimentStoreError("routing decision claim belongs to another experiment")
        return self._rpc(
            "research_lab_routing_append_decision_receipt_v3",
            {
                "p_receipt_id": receipt.receipt_id,
                "p_experiment_hash": experiment_hash,
                "p_variant_id": receipt.variant_id,
                "p_unit_ref": receipt.unit_ref,
                "p_plan_hash": receipt.plan_hash,
                "p_route_hash": receipt.route_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_decision_doc": receipt.to_dict(),
            },
        )

    def decision_row(self, receipt_id: str) -> Mapping[str, Any] | None:
        return self._select_one(
            "research_lab_routing_decision_receipts_v2",
            column="receipt_id",
            value=receipt_id,
        )

    def decision_keys(self, experiment_hash: str) -> tuple[str, ...]:
        return self._select_keys(
            "research_lab_routing_decision_receipts_v2",
            column="receipt_id",
            experiment_hash=experiment_hash,
        )

    def append_evaluation(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        evaluation: RoutingExperimentV2Evaluation,
        claim: RoutingExperimentExecutionClaim,
    ) -> Mapping[str, Any]:
        document = evaluation.to_dict()
        if claim.experiment_hash != spec.experiment_hash():
            raise RoutingExperimentStoreError("routing evaluation claim belongs to another experiment")
        selected_storage_value = evaluation.selected_variant_id or "unselected"
        return self._rpc(
            "research_lab_routing_append_evaluation_v3",
            {
                "p_receipt_id": evaluation.receipt_id,
                "p_experiment_hash": spec.experiment_hash(),
                "p_evaluation_hash": sha256_json(document),
                "p_selected_variant_id": selected_storage_value,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_evaluation_doc": document,
            },
        )

    def append_candidate_waterfall_receipt(
        self,
        *,
        experiment_hash: str,
        receipt: Any,
        claim: RoutingExperimentExecutionClaim,
    ) -> Mapping[str, Any]:
        """Append one exact candidate sidecar through the claim-fenced RPC."""

        from research_lab.candidate_routing_experiments import CandidateWaterfallReceipt

        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if not isinstance(receipt, CandidateWaterfallReceipt):
            raise RoutingExperimentStoreError("candidate waterfall receipt is invalid")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "candidate waterfall receipt claim belongs to another experiment"
            )
        document = receipt.to_dict()
        if receipt.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "candidate waterfall receipt experiment differs"
            )
        result = self._rpc(
            "research_lab_candidate_append_waterfall_receipt_v1",
            {
                "p_receipt_id": receipt.receipt_id,
                "p_receipt_hash": receipt.receipt_hash,
                "p_experiment_hash": normalized_experiment_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_receipt_doc": document,
            },
        )
        if (
            not isinstance(result, Mapping)
            or result.get("receipt_id") != receipt.receipt_id
            or result.get("receipt_hash") != receipt.receipt_hash
            or type(result.get("idempotent")) is not bool
        ):
            raise RoutingExperimentStoreError(
                "candidate waterfall receipt append result is malformed"
            )
        return dict(result)

    def append_candidate_model_unit_terminal(
        self,
        *,
        experiment_hash: str,
        receipt: Any,
        claim: RoutingExperimentExecutionClaim,
    ) -> Mapping[str, Any]:
        """Append one exact Model unit-terminal authority before sidecars."""

        from research_lab.candidate_routing_experiments import CandidateModelUnitTerminalReceipt

        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if not isinstance(receipt, CandidateModelUnitTerminalReceipt):
            raise RoutingExperimentStoreError("candidate Model unit terminal is invalid")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "candidate Model unit terminal claim belongs to another experiment"
            )
        if receipt.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "candidate Model unit terminal experiment differs"
            )
        document = receipt.to_dict()
        result = self._rpc(
            "research_lab_candidate_append_model_unit_terminal_v1",
            {
                "p_receipt_id": receipt.receipt_id,
                "p_receipt_hash": receipt.receipt_hash,
                "p_experiment_hash": normalized_experiment_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_terminal_doc": document,
            },
        )
        if (
            not isinstance(result, Mapping)
            or result.get("receipt_id") != receipt.receipt_id
            or result.get("receipt_hash") != receipt.receipt_hash
            or type(result.get("idempotent")) is not bool
        ):
            raise RoutingExperimentStoreError(
                "candidate Model unit terminal append result is malformed"
            )
        return dict(result)

    def append_candidate_waterfall_metric(
        self,
        *,
        experiment_hash: str,
        metric: Any,
        claim: RoutingExperimentExecutionClaim,
    ) -> Mapping[str, Any]:
        """Append one exact candidate metric through the claim-fenced RPC."""

        from research_lab.candidate_routing_experiments import CandidateWaterfallMetric

        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if not isinstance(metric, CandidateWaterfallMetric):
            raise RoutingExperimentStoreError("candidate waterfall metric is invalid")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "candidate waterfall metric claim belongs to another experiment"
            )
        document = metric.to_dict()
        if metric.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError(
                "candidate waterfall metric experiment differs"
            )
        result = self._rpc(
            "research_lab_candidate_append_waterfall_metric_v1",
            {
                "p_metric_id": metric.metric_id,
                "p_metric_hash": metric.metric_hash,
                "p_experiment_hash": normalized_experiment_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_metric_doc": document,
            },
        )
        if (
            not isinstance(result, Mapping)
            or result.get("metric_id") != metric.metric_id
            or result.get("metric_hash") != metric.metric_hash
            or type(result.get("idempotent")) is not bool
        ):
            raise RoutingExperimentStoreError(
                "candidate waterfall metric append result is malformed"
            )
        return dict(result)

    def evaluation_row(self, receipt_id: str) -> Mapping[str, Any] | None:
        return self._select_one(
            "research_lab_routing_evaluation_receipts_v2",
            column="receipt_id",
            value=receipt_id,
        )

    def reserve_budget(
        self,
        *,
        event_key: str,
        reservation_id: str,
        experiment_hash: str,
        binding_id: str,
        claim: RoutingExperimentExecutionClaim,
        credit_microunits: int,
        lease_seconds: int,
        event_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._rpc(
            "research_lab_routing_reserve_budget_v3",
            {
                "p_event_key": event_key,
                "p_reservation_id": reservation_id,
                "p_experiment_hash": experiment_hash,
                "p_binding_id": binding_id,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_credit_microunits": int(credit_microunits),
                "p_lease_seconds": int(lease_seconds),
                "p_event_doc": dict(event_doc),
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing budget reserve result is malformed")
        return dict(result)

    def settle_budget(
        self,
        *,
        event_key: str,
        reservation_id: str,
        attempt_key: str,
        claim: RoutingExperimentExecutionClaim,
        event_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._rpc(
            "research_lab_routing_settle_budget_v3",
            {
                "p_event_key": event_key,
                "p_reservation_id": reservation_id,
                "p_attempt_key": attempt_key,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_event_doc": dict(event_doc),
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing budget settlement result is malformed")
        return dict(result)

    def mark_budget_uncertain(
        self,
        *,
        event_key: str,
        reservation_id: str,
        claim: RoutingExperimentExecutionClaim,
        event_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._rpc(
            "research_lab_routing_mark_budget_uncertain_v3",
            {
                "p_event_key": event_key,
                "p_reservation_id": reservation_id,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_event_doc": dict(event_doc),
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing uncertain budget result is malformed")
        return dict(result)

    def recover_budget(
        self,
        *,
        event_key: str,
        reservation_id: str,
        claim: RoutingExperimentExecutionClaim,
        event_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._rpc(
            "research_lab_routing_recover_budget_v3",
            {
                "p_event_key": event_key,
                "p_reservation_id": reservation_id,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
                "p_event_doc": dict(event_doc),
            },
        )
        if not isinstance(result, Mapping):
            raise RoutingExperimentStoreError("routing budget recovery result is malformed")
        return dict(result)

    def expired_open_budget_reservations(
        self,
        *,
        experiment_hash: str,
        claim: RoutingExperimentExecutionClaim,
    ) -> tuple[RoutingExperimentExpiredBudgetReservation, ...]:
        """Return only fenced, expired reservation heads for conservative closure.

        The authority does not claim that an absent dispatch marker proves no
        call occurred.  It is audit context only; every returned reservation
        remains a full-ceiling unknown charge until a broker authority settles
        it independently.
        """

        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError("expired budget claim belongs to another experiment")
        result = self._rpc(
            "research_lab_routing_list_expired_budget_reservations_v3",
            {
                "p_experiment_hash": normalized_experiment_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
            },
        )
        if not isinstance(result, Mapping) or set(result) != {"reservations"}:
            raise RoutingExperimentStoreError("expired budget list result is malformed")
        rows = result.get("reservations")
        if not isinstance(rows, list):
            raise RoutingExperimentStoreError("expired budget list is malformed")
        reservations: list[RoutingExperimentExpiredBudgetReservation] = []
        previous_reservation_id = ""
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "reservation_id",
                "binding_id",
                "credit_microunits",
                "dispatch_started",
            }:
                raise RoutingExperimentStoreError("expired budget row is malformed")
            reservation = RoutingExperimentExpiredBudgetReservation(
                reservation_id=str(row["reservation_id"]),
                binding_id=str(row["binding_id"]),
                credit_microunits=row["credit_microunits"],
                dispatch_started=row["dispatch_started"],
            )
            if reservation.reservation_id <= previous_reservation_id:
                raise RoutingExperimentStoreError("expired budget list is not canonical")
            previous_reservation_id = reservation.reservation_id
            reservations.append(reservation)
        return tuple(reservations)

    def unresolved_budget_reservations(
        self,
        *,
        experiment_hash: str,
        claim: RoutingExperimentExecutionClaim,
    ) -> tuple[RoutingExperimentUnresolvedBudgetReservation, ...]:
        """Read every non-settled head under the fresh claim fence."""

        normalized_experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if claim.experiment_hash != normalized_experiment_hash:
            raise RoutingExperimentStoreError("unresolved budget claim belongs to another experiment")
        result = self._rpc(
            "research_lab_routing_list_unresolved_budget_reservations_v3",
            {
                "p_experiment_hash": normalized_experiment_hash,
                "p_claim_key": claim.claim_key,
                "p_claim_generation": claim.claim_generation,
            },
        )
        if not isinstance(result, Mapping) or set(result) != {"reservations"}:
            raise RoutingExperimentStoreError("unresolved budget list result is malformed")
        rows = result.get("reservations")
        if not isinstance(rows, list):
            raise RoutingExperimentStoreError("unresolved budget list is malformed")
        reservations: list[RoutingExperimentUnresolvedBudgetReservation] = []
        previous_reservation_id = ""
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "reservation_id",
                "binding_id",
                "credit_microunits",
                "event_type",
                "lease_expired",
                "dispatch_started",
            }:
                raise RoutingExperimentStoreError("unresolved budget row is malformed")
            reservation = RoutingExperimentUnresolvedBudgetReservation(
                reservation_id=str(row["reservation_id"]),
                binding_id=str(row["binding_id"]),
                credit_microunits=row["credit_microunits"],
                event_type=str(row["event_type"]),
                lease_expired=row["lease_expired"],
                dispatch_started=row["dispatch_started"],
            )
            if reservation.reservation_id <= previous_reservation_id:
                raise RoutingExperimentStoreError("unresolved budget list is not canonical")
            previous_reservation_id = reservation.reservation_id
            reservations.append(reservation)
        return tuple(reservations)

    def reconcile(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        evaluation: RoutingExperimentV2Evaluation,
        attestor: RoutingExperimentEvaluationAttestor | None,
        gold_label_authority: Mapping[str, Any] | None = None,
        artifact_lineage: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        experiment_hash = spec.experiment_hash()
        spec_row = self._select_one(
            "research_lab_routing_experiments_v2",
            column="experiment_hash",
            value=experiment_hash,
        )
        if spec_row is None or spec_row.get("spec_doc") != spec.to_dict():
            raise RoutingExperimentStoreError("routing experiment spec is not authoritative")
        envelope_doc = spec_row.get("execution_envelope_doc")
        envelope_hash = spec_row.get("execution_envelope_hash")
        try:
            envelope = RoutingExperimentExecutionEnvelopeV2.from_mapping(
                envelope_doc if isinstance(envelope_doc, Mapping) else {}
            )
            validate_routing_execution_envelope_v2(
                spec=spec,
                envelope=envelope,
            )
        except ValueError as exc:
            raise RoutingExperimentStoreError(
                "routing execution envelope is not authoritative"
            ) from exc
        if envelope.envelope_hash() != envelope_hash:
            raise RoutingExperimentStoreError(
                "routing execution envelope hash is not authoritative"
            )
        evaluation_row = self.evaluation_row(evaluation.receipt_id)
        evaluation_hash = sha256_json(evaluation.to_dict())
        if (
            evaluation_row is None
            or evaluation_row.get("experiment_hash") != experiment_hash
            or evaluation_row.get("evaluation_hash") != evaluation_hash
            or evaluation_row.get("evaluation_doc") != evaluation.to_dict()
        ):
            raise RoutingExperimentStoreError("routing evaluation is not authoritative")
        if not evaluation.selected_variant_id:
            raise RoutingExperimentStoreError("routing evaluation has no selected variant")
        decisions = self._select_rows(
            "research_lab_routing_decision_receipts_v2",
            experiment_hash=experiment_hash,
            order_column="receipt_id",
        )
        attempts = self._select_rows(
            "research_lab_routing_provider_attempts_v2",
            experiment_hash=experiment_hash,
            order_column="attempt_key",
        )
        budgets = self._select_rows(
            "research_lab_routing_budget_events_v2",
            experiment_hash=experiment_hash,
            order_column="event_key",
        )
        actual_decision_refs = tuple(sorted(str(row.get("receipt_id") or "") for row in decisions))
        actual_provider_refs = tuple(sorted(str(row.get("provider_receipt_ref") or "") for row in attempts))
        if (
            not actual_decision_refs
            or not actual_provider_refs
            or actual_decision_refs != tuple(evaluation.decision_receipt_refs)
            or actual_provider_refs != tuple(evaluation.provider_receipt_refs)
        ):
            raise RoutingExperimentStoreError(
                "routing evaluation receipt references are incomplete or non-authoritative"
            )
        if spec.input.stage == "candidate_acquisition":
            sidecar_receipts = self._select_rows(
                "research_lab_candidate_waterfall_receipts",
                experiment_hash=experiment_hash,
                order_column="receipt_id",
            )
            sidecar_metrics = self._select_rows(
                "research_lab_candidate_waterfall_metrics",
                experiment_hash=experiment_hash,
                order_column="metric_id",
            )
            candidate_provider_refs = {
                str(row.get("provider_receipt_ref") or "")
                for row in sidecar_receipts
                if str(row.get("provider_receipt_ref") or "")
            }
            expected_candidate_provider_refs = {
                str(ref)
                for ref in evaluation.provider_receipt_refs
                if any(
                    str(attempt.get("provider_receipt_ref") or "") == str(ref)
                    and str(attempt.get("tool_id") or "").startswith("candidate.")
                    for attempt in attempts
                )
            }
            if (
                not sidecar_receipts
                or not sidecar_metrics
                or candidate_provider_refs != expected_candidate_provider_refs
            ):
                raise RoutingExperimentStoreError(
                    "candidate waterfall sidecar coverage is incomplete or non-authoritative"
                )
            expected_metric_keys = {
                (variant.variant_id, split)
                for variant in spec.variants
                for split in ("calibration", "holdout")
            }
            actual_metric_keys = {
                (str(row.get("variant_id") or ""), str(row.get("split") or ""))
                for row in sidecar_metrics
            }
            if (
                len(actual_metric_keys) != len(sidecar_metrics)
                or actual_metric_keys != expected_metric_keys
            ):
                raise RoutingExperimentStoreError(
                    "candidate waterfall metric coverage is incomplete or non-authoritative"
                )
            receipt_by_id: dict[str, Mapping[str, Any]] = {}
            receipt_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
            target_values: set[int] = set()
            for row in sidecar_receipts:
                receipt_id = str(row.get("receipt_id") or "")
                receipt_doc = row.get("receipt_doc")
                if not isinstance(receipt_doc, Mapping):
                    raise RoutingExperimentStoreError(
                        "candidate waterfall receipt authority differs"
                    )
                verification_hashes = receipt_doc.get(
                    "company_verification_receipt_sha256s"
                )
                if not isinstance(verification_hashes, list):
                    raise RoutingExperimentStoreError(
                        "candidate waterfall verification receipts are incomplete"
                    )
                expected_receipt_doc = {
                    key: row.get(key)
                    for key in (
                        "receipt_id", "receipt_hash", "contract_version",
                        "experiment_id", "experiment_hash", "variant_id",
                        "artifact_key", "decision_receipt_id",
                        "model_terminal_receipt_id", "model_terminal_receipt_hash",
                        "publication_projection_sha256",
                        "provider_receipt_ref", "unit_ref", "binding_id",
                        "tool_id", "execution_mode", "provider_outcome",
                        "decision_plan_hash", "decision_route_hash",
                        "model_contract_sha256", "model_plan_sha256",
                        "stop_policy_sha256", "attempt_receipt_sha256",
                        "prior_attempt_receipt_sha256", "attempt_chain_sha256",
                        "verification_receipt_sha256", "step_order",
                        "attempt_sequence", "target_verified_qualified_count",
                        "disposition", "outcome_code", "provider_call_count",
                        "billed_credit_microunits", "latency_ms", "raw_count",
                        "normalized_count", "unique_count",
                        "verified_qualified_count", "published_count", "immutable",
                    )
                }
                expected_receipt_doc[
                    "company_verification_receipt_sha256s"
                ] = verification_hashes
                if (
                    row.get("experiment_hash") != experiment_hash
                    or receipt_doc != expected_receipt_doc
                    or row.get("company_verification_receipt_sha256s")
                    != verification_hashes
                    or sha256_json(
                        {
                            key: value
                            for key, value in receipt_doc.items()
                            if key not in {"receipt_id", "receipt_hash"}
                        }
                    )
                    != row.get("receipt_hash")
                    or receipt_id != "candidate_waterfall:"
                    + str(row.get("receipt_hash") or "").split(":", 1)[-1][:24]
                ):
                    raise RoutingExperimentStoreError(
                        "candidate waterfall receipt authority differs"
                    )
                try:
                    target_values.add(int(row.get("target_verified_qualified_count")))
                except (TypeError, ValueError) as exc:
                    raise RoutingExperimentStoreError(
                        "candidate waterfall target is invalid"
                    ) from exc
                receipt_by_id[receipt_id] = row
                receipt_groups.setdefault(
                    (str(row.get("variant_id") or ""), str(row.get("unit_ref") or "")),
                    [],
                ).append(row)
            if len(target_values) != 1:
                raise RoutingExperimentStoreError(
                    "candidate waterfall target is inconsistent"
                )
            for group in receipt_groups.values():
                ordered = sorted(
                    group,
                    key=lambda item: (
                        int(item.get("step_order") or 0),
                        int(item.get("attempt_sequence") or 0),
                    ),
                )
                prefix_hashes: list[str] = []
                for expected_index, row in enumerate(ordered):
                    if (
                        row.get("step_order") != expected_index
                        or row.get("attempt_sequence") != expected_index
                        or row.get("prior_attempt_receipt_sha256")
                        != (prefix_hashes[-1] if prefix_hashes else "")
                    ):
                        raise RoutingExperimentStoreError(
                            "candidate waterfall attempt sequence is incomplete"
                        )
                    prefix_hashes.append(str(row.get("attempt_receipt_sha256") or ""))
                    if row.get("attempt_chain_sha256") != sha256_json(prefix_hashes).split(":", 1)[1]:
                        raise RoutingExperimentStoreError(
                            "candidate waterfall attempt chain is non-authoritative"
                        )
            split_units = {
                "calibration": set(spec.input.calibration_unit_refs),
                "holdout": set(spec.input.holdout_unit_refs),
            }
            evaluation_by_variant = {
                item.variant_id: item for item in evaluation.variants
            }
            for variant_id, variant_evaluation in evaluation_by_variant.items():
                variant_rows = [
                    row for row in sidecar_receipts
                    if row.get("variant_id") == variant_id
                ]
                if {
                    str(row.get("decision_receipt_id") or "") for row in variant_rows
                } != set(variant_evaluation.decision_receipt_refs):
                    raise RoutingExperimentStoreError(
                        "candidate waterfall decision sidecar coverage is incomplete"
                    )
                expected_variant_provider_refs = {
                    str(attempt.get("provider_receipt_ref") or "")
                    for attempt in attempts
                    if attempt.get("variant_id") == variant_id
                    and str(attempt.get("tool_id") or "").startswith("candidate.")
                }
                actual_variant_provider_refs = {
                    str(row.get("provider_receipt_ref") or "")
                    for row in variant_rows
                    if str(row.get("provider_receipt_ref") or "")
                }
                if actual_variant_provider_refs != expected_variant_provider_refs:
                    raise RoutingExperimentStoreError(
                        "candidate waterfall provider sidecar coverage is incomplete"
                    )
            for row in sidecar_metrics:
                metric_doc = row.get("metric_doc")
                if not isinstance(metric_doc, Mapping):
                    raise RoutingExperimentStoreError(
                        "candidate waterfall metric authority differs"
                    )
                expected_metric_doc = {
                    key: row.get(key)
                    for key in (
                        "metric_id", "metric_hash", "contract_version",
                        "evaluation_receipt_id", "experiment_id", "experiment_hash",
                        "variant_id", "split", "target_verified_qualified_count",
                        "unit_count", "fulfilled_unit_count", "waterfall_attempt_count",
                        "provider_call_count", "total_billed_credit_microunits",
                        "total_latency_ms", "raw_count", "normalized_count",
                        "unique_count", "verified_qualified_count", "published_count",
                        "failed_attempt_count", "missed_attempt_count", "fulfillment_rate",
                        "verification_rate", "publication_rate",
                        "verified_qualified_per_credit", "immutable",
                    )
                }
                for field_name in (
                    "waterfall_receipt_refs",
                    "provider_receipt_refs",
                    "decision_receipt_refs",
                ):
                    expected_metric_doc[field_name] = metric_doc.get(field_name)
                if (
                    metric_doc != expected_metric_doc
                    or sha256_json(
                        {
                            key: value
                            for key, value in metric_doc.items()
                            if key not in {"metric_id", "metric_hash"}
                        }
                    )
                    != row.get("metric_hash")
                    or str(row.get("evaluation_receipt_id") or "")
                    != evaluation.receipt_id
                    or row.get("target_verified_qualified_count")
                    != next(iter(target_values))
                ):
                    raise RoutingExperimentStoreError(
                        "candidate waterfall metric authority differs"
                    )
                variant_id = str(row.get("variant_id") or "")
                split = str(row.get("split") or "")
                selected = sorted(
                    (
                        receipt
                        for receipt in sidecar_receipts
                        if receipt.get("variant_id") == variant_id
                        and receipt.get("unit_ref") in split_units.get(split, set())
                    ),
                    key=lambda item: (
                        str(item.get("unit_ref") or ""),
                        int(item.get("step_order") or 0),
                        int(item.get("attempt_sequence") or 0),
                    ),
                )
                expected_waterfall_refs = [
                    str(receipt.get("receipt_id") or "") for receipt in selected
                ]
                expected_provider_refs = sorted(
                    {
                        str(receipt.get("provider_receipt_ref") or "")
                        for receipt in selected
                        if str(receipt.get("provider_receipt_ref") or "")
                    }
                )
                expected_decision_refs = sorted(
                    {
                        str(receipt.get("decision_receipt_id") or "")
                        for receipt in selected
                    }
                )
                if (
                    metric_doc.get("waterfall_receipt_refs") != expected_waterfall_refs
                    or metric_doc.get("provider_receipt_refs") != expected_provider_refs
                    or metric_doc.get("decision_receipt_refs") != expected_decision_refs
                    or any(
                        ref not in receipt_by_id
                        for ref in expected_waterfall_refs
                    )
                ):
                    raise RoutingExperimentStoreError(
                        "candidate waterfall metric receipt coverage is incomplete"
                    )
        if any(
            row.get("outcome") == ProviderOutcome.ADAPTER_FAILURE.value
            or row.get("billing_state") != "known"
            for row in attempts
        ):
            raise RoutingExperimentStoreError("routing evaluation contains a failed or uncertain provider attempt")
        if spec.allow_live_credit_spend:
            terminal_by_reservation: dict[str, Mapping[str, Any]] = {}
            # ``_select_rows`` uses a stable key order for roots.  Ledger
            # event keys are hashes, however, so that order is unrelated to
            # event time.  Select a head by the durable append timestamp (and
            # event key only as its deterministic same-timestamp tie-breaker).
            for row in sorted(
                budgets,
                key=lambda item: (
                    str(item.get("created_at") or ""),
                    str(item.get("event_key") or ""),
                ),
            ):
                reservation_id = str(row.get("reservation_id") or "")
                if reservation_id:
                    terminal_by_reservation[reservation_id] = row
            if not terminal_by_reservation or any(
                row.get("event_type") != "settle"
                for row in terminal_by_reservation.values()
            ):
                raise RoutingExperimentStoreError("routing live budget settlement is incomplete")
            billed_total = sum(
                int(row.get("authoritative_billed_credit_microunits") or 0)
                for row in attempts
            )
            if billed_total != evaluation.billing_rollup_total_credit_microunits:
                raise RoutingExperimentStoreError("routing authoritative billing total differs")
        if not isinstance(gold_label_authority, Mapping):
            raise RoutingExperimentStoreError("signed routing gold labels are required")
        if not isinstance(artifact_lineage, Mapping):
            raise RoutingExperimentStoreError("signed routing artifact lineage is required")
        decision_projections = tuple(
            {
                "receipt_id": str(row.get("receipt_id") or ""),
                "experiment_hash": str(row.get("experiment_hash") or ""),
                "decision_doc": row.get("decision_doc"),
            }
            for row in decisions
        )
        attempt_fields = (
            "attempt_key",
            "experiment_hash",
            "provider_receipt_ref",
            "binding_id",
            "tool_id",
            "variant_id",
            "unit_ref",
            "reservation_id",
            "action_id",
            "binding_catalog_manifest_hash",
            "authorization_hash",
            "authorization_proof_hash",
            "request_body_hash",
            "request_fingerprint",
            "outcome",
            "credit_microunits",
            "latency_ms",
            "execution_mode",
            "billing_state",
            "authoritative_billed_credit_microunits",
            "attempt_doc",
        )
        attempt_projections = tuple(
            {field_name: row.get(field_name) for field_name in attempt_fields}
            for row in attempts
        )
        budget_fields = (
            "event_key",
            "experiment_hash",
            "reservation_id",
            "binding_id",
            "attempt_key",
            "event_type",
            "credit_microunits",
            "event_doc",
        )
        budget_projections = tuple(
            {field_name: row.get(field_name) for field_name in budget_fields}
            for row in budgets
        )
        attestation_payload = build_routing_experiment_attestation_input_v2(
            spec_doc=spec.to_dict(),
            evaluation_doc=evaluation.to_dict(),
            gold_label_authority=dict(gold_label_authority),
            artifact_lineage=dict(artifact_lineage),
            execution_envelope=envelope.to_dict(),
            decision_receipts=decision_projections,
            provider_attempts=attempt_projections,
            budget_events=budget_projections,
        )
        if attestor is None:
            raise RoutingExperimentStoreError("routing evaluation attestation authority is required")
        authority = attestor.attest(attestation_payload)
        if not isinstance(authority, Mapping):
            raise RoutingExperimentStoreError("routing evaluation attestation is malformed")
        result = authority.get("result")
        receipt = authority.get("receipt")
        expected_result = execute_routing_experiment_attestation_v2(attestation_payload)
        if (
            not isinstance(result, Mapping)
            or not isinstance(receipt, Mapping)
            or dict(result) != expected_result
            or receipt.get("role") != "gateway_scoring"
            or receipt.get("purpose") != "research_lab.routing_experiment.v2"
            or receipt.get("status") != "succeeded"
            or receipt.get("input_root") != expected_result["input_root"]
            or receipt.get("output_root") != expected_result["output_root"]
        ):
            raise RoutingExperimentStoreError("routing evaluation attestation is not authoritative")
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise RoutingExperimentStoreError(
                "routing evaluation attestation signature is invalid"
            ) from exc
        persisted_receipt = self._select_one(
            "research_lab_attested_execution_receipts_v2",
            column="receipt_hash",
            value=str(receipt["receipt_hash"]),
        )
        if (
            persisted_receipt is None
            or persisted_receipt.get("receipt_doc") != dict(receipt)
        ):
            raise RoutingExperimentStoreError(
                "routing evaluation attestation receipt is not persisted"
            )
        return {
            "reconciled": True,
            "experiment_hash": experiment_hash,
            "evaluation_receipt_id": evaluation.receipt_id,
            "evaluation_hash": evaluation_hash,
            "selected_variant_id": evaluation.selected_variant_id,
            "authority_receipt_hash": str(receipt["receipt_hash"]),
            "authority_input_root": str(result["input_root"]),
            "authority_output_root": str(result["output_root"]),
            "decision_receipts_root": str(result["decision_receipts_root"]),
            "provider_attempts_root": str(result["provider_attempts_root"]),
            "budget_events_root": str(result["budget_events_root"]),
            "artifact_lineage_hash": str(result["artifact_lineage_hash"]),
            "gold_label_manifest_hash": str(result["gold_label_manifest_hash"]),
            "execution_envelope_hash": str(result["execution_envelope_hash"]),
            "authority_commit_sha": str(receipt["commit_sha"]),
            "authority_pcr0": str(receipt["pcr0"]),
            "authority_build_manifest_hash": str(
                receipt["build_manifest_hash"]
            ),
            "authority_boot_identity_hash": str(receipt["boot_identity_hash"]),
            "artifact_pointer_document_hash": envelope.pointer_document_hash,
            "authoritative_billed_credit_microunits": int(
                result["authoritative_billed_credit_microunits"]
            ),
        }

    def promote(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        evaluation: RoutingExperimentV2Evaluation,
        reconciliation: Mapping[str, Any],
    ) -> str:
        if not isinstance(reconciliation, Mapping):
            raise RoutingExperimentStoreError(
                "routing promotion reconciliation is invalid"
            )
        if (
            reconciliation.get("reconciled") is not True
            or reconciliation.get("experiment_hash") != spec.experiment_hash()
            or reconciliation.get("evaluation_receipt_id") != evaluation.receipt_id
            or reconciliation.get("evaluation_hash") != sha256_json(evaluation.to_dict())
        ):
            raise RoutingExperimentStoreError(
                "routing promotion requires an authoritative reconciliation"
            )
        reference_hash = sha256_json(
            {
                "contract_version": "leadpoet.routing_experiment_v2_lab_reference:v2",
                "experiment_hash": spec.experiment_hash(),
                "evaluation_hash": sha256_json(evaluation.to_dict()),
                "evaluation_receipt_id": evaluation.receipt_id,
                "selected_variant_id": evaluation.selected_variant_id,
                "reconciliation": dict(reconciliation),
            }
        )
        event_doc = {
            "schema_version": "leadpoet.research_lab.routing_event.v2",
            "event_type": "promoted",
            "experiment_hash": spec.experiment_hash(),
            "evaluation_receipt_id": evaluation.receipt_id,
            "reference_hash": reference_hash,
        }
        result = self._rpc(
            "research_lab_routing_promote_v3",
            {
                "p_reference_hash": reference_hash,
                "p_experiment_hash": spec.experiment_hash(),
                "p_evaluation_receipt_id": evaluation.receipt_id,
                "p_evaluation_hash": sha256_json(evaluation.to_dict()),
                "p_selected_variant_id": evaluation.selected_variant_id,
                "p_reconciliation_doc": dict(reconciliation),
                "p_event_hash": _event_hash("promoted", event_doc),
                "p_event_doc": event_doc,
            },
        )
        if not isinstance(result, Mapping) or result.get("reference_hash") != reference_hash:
            raise RoutingExperimentStoreError("routing promotion result is malformed")
        return reference_hash


class SupabaseRoutingProviderReceiptRepository:
    """Provider receipt repository backed by the routing authority RPC."""

    durable = True

    def __init__(
        self,
        *,
        store: SupabaseRoutingExperimentStore,
        experiment_hash: str,
        claim: RoutingExperimentExecutionClaim,
        billing_metadata: Callable[[ProviderReceipt], Mapping[str, Any]] | None = None,
        after_append: Callable[[str, ProviderReceipt], None] | None = None,
    ) -> None:
        self.store = store
        self.experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if claim.experiment_hash != self.experiment_hash:
            raise RoutingExperimentStoreError("provider receipt claim belongs to another experiment")
        self.claim = claim
        self._billing_metadata = billing_metadata
        self._after_append = after_append

    def get(self, key: str) -> ProviderReceipt | None:
        row = self.store.provider_attempt_row(str(key))
        if row is None:
            failure_row_reader = getattr(self.store, "adapter_failure_row", None)
            row = (
                failure_row_reader(str(key))
                if callable(failure_row_reader)
                else None
            )
            if row is None:
                return None
        if row.get("experiment_hash") != self.experiment_hash:
            raise RoutingExperimentStoreError("provider receipt belongs to another routing experiment")
        document = row.get("attempt_doc") or row.get("failure_doc")
        if not isinstance(document, Mapping):
            raise RoutingExperimentStoreError("stored provider receipt document is malformed")
        # V3 provider attempts and pre-dispatch failures wrap the typed
        # receipt inside their redacted authority document.  Accepting only
        # this exact nested field prevents the surrounding proof document
        # from being mistaken for a ProviderReceipt.
        if isinstance(document.get("provider_receipt"), Mapping):
            document = document["provider_receipt"]
        receipt = ProviderReceipt.from_mapping(document)
        expected = provider_receipt_key(
            tool_id=receipt.tool_id,
            binding_version=receipt.binding_version,
            request_fingerprint=receipt.request_fingerprint,
        )
        if expected != str(key):
            raise RoutingExperimentStoreError("stored provider receipt key is inconsistent")
        return receipt

    def append(self, key: str, receipt: ProviderReceipt) -> ProviderReceipt:
        raise RoutingExperimentStoreError(
            "routing provider receipt append requires immutable execution context"
        )

    def append_with_context(
        self,
        key: str,
        receipt: ProviderReceipt,
        execution_context: Mapping[str, Any],
    ) -> ProviderReceipt:
        expected = provider_receipt_key(
            tool_id=receipt.tool_id,
            binding_version=receipt.binding_version,
            request_fingerprint=receipt.request_fingerprint,
        )
        if str(key) != expected:
            raise RoutingExperimentError("provider receipt key mismatch")
        existing = self.get(str(key))
        if existing is not None:
            if existing.to_dict() != receipt.to_dict():
                raise RoutingExperimentStoreError("routing provider receipt key collision")
            return existing
        variant_id = str(execution_context.get("variant_id") or "")
        unit_ref = str(execution_context.get("unit_ref") or "")
        if not variant_id or unit_ref != receipt.unit_ref:
            raise RoutingExperimentStoreError("routing provider execution context is invalid")
        if receipt.outcome == ProviderOutcome.ADAPTER_FAILURE.value:
            append_adapter_failure = getattr(self.store, "append_adapter_failure", None)
            if not callable(append_adapter_failure):
                raise RoutingExperimentStoreError(
                    "routing adapter failure persistence is unavailable"
                )
            append_adapter_failure(
                experiment_hash=self.experiment_hash,
                key=str(key),
                receipt=receipt,
                variant_id=variant_id,
                claim=self.claim,
            )
        else:
            self.store.append_provider_attempt(
                experiment_hash=self.experiment_hash,
                key=str(key),
                receipt=receipt,
                variant_id=variant_id,
                claim=self.claim,
                **self._provider_billing_metadata(receipt),
            )
        if self._after_append is not None:
            self._after_append(str(key), receipt)
        return receipt

    def _provider_billing_metadata(self, receipt: ProviderReceipt) -> Mapping[str, Any]:
        if self._billing_metadata is None:
            return {}
        metadata = self._billing_metadata(receipt)
        if not isinstance(metadata, Mapping):
            raise RoutingExperimentStoreError("routing provider billing metadata is malformed")
        allowed = {"billing_state", "authoritative_billed_credit_microunits"}
        if set(metadata) - allowed:
            raise RoutingExperimentStoreError("routing provider billing metadata has unknown fields")
        return dict(metadata)

    def keys(self) -> Iterable[str]:
        failure_key_reader = getattr(self.store, "adapter_failure_keys", None)
        failure_keys = (
            failure_key_reader(self.experiment_hash)
            if callable(failure_key_reader)
            else ()
        )
        return tuple(
            sorted(
                set(self.store.provider_attempt_keys(self.experiment_hash))
                | set(failure_keys)
            )
        )


class SupabaseRoutingDecisionReceiptRepository:
    """Decision receipt repository backed by the routing authority RPC."""

    durable = True

    def __init__(
        self,
        *,
        store: SupabaseRoutingExperimentStore,
        experiment_hash: str,
        claim: RoutingExperimentExecutionClaim,
    ) -> None:
        self.store = store
        self.experiment_hash = _require_hash(experiment_hash, "experiment_hash")
        if claim.experiment_hash != self.experiment_hash:
            raise RoutingExperimentStoreError("decision receipt claim belongs to another experiment")
        self.claim = claim

    def get(self, key: str) -> RoutingDecisionReceiptV2 | None:
        row = self.store.decision_row(str(key))
        if row is None:
            return None
        if row.get("experiment_hash") != self.experiment_hash:
            raise RoutingExperimentStoreError("decision receipt belongs to another routing experiment")
        document = row.get("decision_doc")
        if not isinstance(document, Mapping):
            raise RoutingExperimentStoreError("stored decision receipt document is malformed")
        return RoutingDecisionReceiptV2.from_mapping(document)

    def append(
        self,
        key: str,
        receipt: RoutingDecisionReceiptV2,
    ) -> RoutingDecisionReceiptV2:
        if str(key) != receipt.receipt_id:
            raise RoutingExperimentError("v2_decision_receipt_repository_key_mismatch")
        self.store.append_decision(
            experiment_hash=self.experiment_hash,
            receipt=receipt,
            claim=self.claim,
        )
        return receipt

    def keys(self) -> Iterable[str]:
        return self.store.decision_keys(self.experiment_hash)


class SupabaseRoutingExperimentPromotionAuthority:
    """The only production-capable authority for Lab reference creation."""

    authoritative = True

    def __init__(
        self,
        store: SupabaseRoutingExperimentStore,
        *,
        attestor: RoutingExperimentEvaluationAttestor | None = None,
        gold_label_authority: Mapping[str, Any] | None = None,
        artifact_lineage: Mapping[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.attestor = attestor
        self.gold_label_authority = (
            dict(gold_label_authority) if isinstance(gold_label_authority, Mapping) else None
        )
        self.artifact_lineage = (
            dict(artifact_lineage) if isinstance(artifact_lineage, Mapping) else None
        )

    def reconcile(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        evaluation: RoutingExperimentV2Evaluation,
    ) -> Mapping[str, Any]:
        return self.store.reconcile(
            spec=spec,
            evaluation=evaluation,
            attestor=self.attestor,
            gold_label_authority=self.gold_label_authority,
            artifact_lineage=self.artifact_lineage,
        )

    def promote(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        evaluation: RoutingExperimentV2Evaluation,
        reconciliation: Mapping[str, Any],
    ) -> str:
        return self.store.promote(
            spec=spec,
            evaluation=evaluation,
            reconciliation=reconciliation,
        )


__all__ = [
    "RoutingExperimentStoreError",
    "RoutingExperimentEvaluationAttestor",
    "RoutingExperimentExecutionClaim",
    "RoutingExperimentExpiredBudgetReservation",
    "RoutingExperimentUnresolvedBudgetReservation",
    "SupabaseRoutingExperimentStore",
    "SupabaseRoutingProviderReceiptRepository",
    "SupabaseRoutingDecisionReceiptRepository",
    "SupabaseRoutingExperimentPromotionAuthority",
]
