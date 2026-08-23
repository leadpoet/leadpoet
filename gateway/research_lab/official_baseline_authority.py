"""Concrete append-only authority for exact official-baseline model actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib
import os
import re
from typing import Any, Mapping, Protocol

from gateway.research_lab.common_model_experiment import (
    CommonModelExperimentRecoveryError,
    ProtectedModelActionResult,
    _validate_compiled_provider_dispatch,
)
from gateway.research_lab.official_baseline_custody import (
    OFFICIAL_BASELINE_CUSTODY_KMS_KEY_ENV,
    OFFICIAL_BASELINE_CUSTODY_S3_PREFIX_ENV,
    S3OfficialBaselineDocumentCustody,
    S3OfficialBaselineTransitionRepository,
    official_baseline_custody_configuration,
)
from gateway.research_lab.official_baseline_model_runner import (
    OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION,
    OFFICIAL_BASELINE_AUTHORITY_PREFLIGHT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION,
    OfficialBaselineAttemptStore,
    OfficialBaselineAuthorityUnavailable,
    OfficialBaselineDependencyContext,
    OfficialBaselineExactDependencies,
    OfficialBaselineModelError,
)
from gateway.research_lab.official_baseline_store import (
    official_baseline_action_replay_identity,
    official_baseline_terminal_store_outcome,
)
from research_lab.canonical import sha256_json
from research_lab.common_model_runner_host import HostActionResult
from research_lab.model_runner_protocol import ExactModelRunnerRegistration
from research_lab.routing_experiments import ProviderReceipt


OFFICIAL_BASELINE_PROTECTED_PREPARATION_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_protected_preparation.v1"
)
OFFICIAL_BASELINE_PROTECTED_TERMINAL_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_protected_terminal.v1"
)
OFFICIAL_BASELINE_PROTECTED_RECONCILIATION_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_protected_reconciliation.v1"
)
PROTECTED_ACTION_AUTHORITY_SCHEMA_VERSION = (
    "leadpoet.site.protected_action_authority.v1"
)
PROTECTED_ACTION_AUTHORITY_SHA256 = (
    "sha256:7f93061601526ce3d14b8555fefe388a1fd7322b565a748f06e232e2cb5c1b7a"
)

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}")
_UNIT_RE = re.compile(r"baseline_icp:[0-9a-f]{64}")
_PROVIDER_ACTION_TYPES = frozenset(
    {
        "normalize_icp",
        "execute_candidate_tool",
        "execute_intent_tool",
        "execute_contact_tool",
    }
)
_VERIFIER_ACTION_TYPES = frozenset(
    {"verify_company", "verify_intent", "verify_contact"}
)


class OfficialBaselineProtectedAuthorityError(OfficialBaselineModelError):
    """The protected action state machine cannot prove one exact outcome."""


def _prefixed_hash(value: Any, label: str) -> str:
    text = str(value or "").lower()
    normalized = text if text.startswith("sha256:") else "sha256:" + text
    if _HASH_RE.fullmatch(normalized) is None:
        raise OfficialBaselineProtectedAuthorityError(f"{label} is invalid")
    return normalized


def _safe_ref(value: Any, label: str) -> str:
    normalized = str(value or "")
    if _SAFE_REF_RE.fullmatch(normalized) is None:
        raise OfficialBaselineProtectedAuthorityError(f"{label} is invalid")
    return normalized


@dataclass(frozen=True)
class OfficialBaselineProtectedPreparation:
    """Side-effect-free protected job identity produced before SQL reserve."""

    authority_identity_sha256: str
    run_sha256: str
    unit_ref: str
    action_idempotency_sha256: str
    action_sha256: str
    action_sequence: int
    action_type: str
    tool_id: str
    binding_contract_sha256: str
    request_fingerprint_sha256: str
    request_body_sha256: str
    call_cap: int
    credit_cap_microunits: int
    timeout_ms: int
    protected_job_ref: str
    protected_request_sha256: str

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": OFFICIAL_BASELINE_PROTECTED_PREPARATION_SCHEMA_VERSION,
            **asdict(self),
        }

    @property
    def preparation_sha256(self) -> str:
        return sha256_json(self.document())


@dataclass(frozen=True)
class OfficialBaselineProtectedTerminal:
    """Known or uncertain protected job readback; never caller-fabricated."""

    state: str
    protected_action_result: ProtectedModelActionResult | None
    protected_result_sha256: str | None
    protected_terminal_receipt_ref: str | None
    protected_terminal_receipt_sha256: str | None
    provider_request_ref: str | None
    model_provider_response_sha256: str | None
    uncertainty_sha256: str | None = None


class OfficialBaselineProtectedActionBridge(Protocol):
    """Gateway-local implementation of the frozen static action contract.

    It has no Site run/request/lease context and no Site database or API.  The
    identity is the runtime-neutral prepare/execute/reconcile contract; live
    release, registry, image, handler, and credential identities are checked
    separately by the release loader.
    """

    @property
    def authority_identity_sha256(self) -> str: ...

    def prepare(
        self,
        *,
        run_identity: Mapping[str, Any],
        unit_ref: str,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedPreparation: ...

    def execute_prepared(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal: ...

    def reconcile(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal: ...


class OfficialBaselinePreparedActionExecutor(Protocol):
    """Release-bound compiler/broker seam under the durable outer claim."""

    def prepare(
        self,
        *,
        run_identity: Mapping[str, Any],
        unit_ref: str,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedPreparation: ...

    def execute_prepared(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal: ...

    def reconcile(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal: ...


@dataclass(frozen=True)
class OfficialBaselineReleaseComponents:
    """Frozen release-owned objects returned by the fixed production loader."""

    registration: ExactModelRunnerRegistration
    projector: Any
    protected_bridge: OfficialBaselineProtectedActionBridge


def _protected_result_document(result: ProtectedModelActionResult) -> dict[str, Any]:
    if not isinstance(result, ProtectedModelActionResult) or not isinstance(
        result.host_result, HostActionResult
    ):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected result is invalid"
        )
    host_document = asdict(result.host_result)
    for optional_custody_field in (
        "model_provider_response_ingestion",
        "provider_action_receipt_sha256",
    ):
        if host_document.get(optional_custody_field) is None:
            host_document.pop(optional_custody_field, None)
    document = {
        "schema_version": (
            "leadpoet.research_lab.official_baseline_protected_result.v2"
            if result.model_provider_response_ingestion is not None
            else "leadpoet.research_lab.official_baseline_protected_result.v1"
        ),
        "host_result": host_document,
        "provider_receipt": (
            None
            if result.provider_receipt is None
            else result.provider_receipt.to_dict()
        ),
        "replay_ref": (None if result.replay_ref is None else dict(result.replay_ref)),
    }
    if result.model_provider_response_ingestion is not None:
        if not isinstance(result.model_provider_response_ingestion, Mapping):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider response ingestion is invalid"
            )
        document["model_provider_response_ingestion"] = dict(
            result.model_provider_response_ingestion
        )
    return document


def _protected_result_from_document(
    value: Mapping[str, Any],
) -> ProtectedModelActionResult:
    if not isinstance(value, Mapping):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected result document is invalid"
        )
    schema_version = value.get("schema_version")
    expected_fields = {
        "schema_version",
        "host_result",
        "provider_receipt",
        "replay_ref",
    }
    if schema_version == (
        "leadpoet.research_lab.official_baseline_protected_result.v2"
    ):
        expected_fields.add("model_provider_response_ingestion")
    elif schema_version != (
        "leadpoet.research_lab.official_baseline_protected_result.v1"
    ):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected result document is invalid"
        )
    if set(value) != expected_fields:
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected result document is invalid"
        )
    host_value = value.get("host_result")
    if not isinstance(host_value, Mapping):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected host result is invalid"
        )
    try:
        host = HostActionResult(**dict(host_value))
        receipt_value = value.get("provider_receipt")
        receipt = (
            None
            if receipt_value is None
            else ProviderReceipt.from_mapping(receipt_value)
        )
    except Exception as exc:
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected result cannot be reconstructed"
        ) from exc
    replay = value.get("replay_ref")
    if replay is not None and not isinstance(replay, Mapping):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected replay reference is invalid"
        )
    ingestion = value.get("model_provider_response_ingestion")
    if ingestion is not None and not isinstance(ingestion, Mapping):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected ingestion receipt is invalid"
        )
    result = ProtectedModelActionResult(
        host_result=host,
        provider_receipt=receipt,
        replay_ref=None if replay is None else dict(replay),
        model_provider_response_ingestion=(
            None if ingestion is None else dict(ingestion)
        ),
    )
    if _protected_result_document(result) != dict(value):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline protected result reconstruction differs"
        )
    return result


def _protected_terminal_document(
    terminal: OfficialBaselineProtectedTerminal,
    *,
    preparation_sha256: str,
) -> dict[str, Any]:
    if terminal.state != "known" or terminal.protected_action_result is None:
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline durable terminal is not known"
        )
    result_document = _protected_result_document(
        terminal.protected_action_result
    )
    if (
        terminal.protected_result_sha256 != sha256_json(result_document)
        or terminal.model_provider_response_sha256
        != sha256_json(
            terminal.protected_action_result.host_result.provider_response
        )
        or _SAFE_REF_RE.fullmatch(
            str(terminal.protected_terminal_receipt_ref or "")
        )
        is None
        or _HASH_RE.fullmatch(
            str(terminal.protected_terminal_receipt_sha256 or "")
        )
        is None
        or (
            terminal.provider_request_ref is not None
            and _SAFE_REF_RE.fullmatch(terminal.provider_request_ref) is None
        )
        or terminal.uncertainty_sha256 is not None
    ):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline durable terminal custody differs"
        )
    body = {
        "schema_version": OFFICIAL_BASELINE_PROTECTED_TERMINAL_SCHEMA_VERSION,
        "preparation_sha256": preparation_sha256,
        "state": "known",
        "protected_action_result": result_document,
        "protected_result_sha256": terminal.protected_result_sha256,
        "protected_terminal_receipt_ref": (
            terminal.protected_terminal_receipt_ref
        ),
        "protected_terminal_receipt_sha256": (
            terminal.protected_terminal_receipt_sha256
        ),
        "provider_request_ref": terminal.provider_request_ref,
        "model_provider_response_sha256": (
            terminal.model_provider_response_sha256
        ),
        "uncertainty_sha256": None,
    }
    return {**body, "terminal_sha256": sha256_json(body)}


def _protected_terminal_from_document(
    value: Mapping[str, Any],
    *,
    preparation_sha256: str,
) -> OfficialBaselineProtectedTerminal:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "preparation_sha256",
        "state",
        "protected_action_result",
        "protected_result_sha256",
        "protected_terminal_receipt_ref",
        "protected_terminal_receipt_sha256",
        "provider_request_ref",
        "model_provider_response_sha256",
        "uncertainty_sha256",
        "terminal_sha256",
    }:
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline durable terminal is not closed"
        )
    body = dict(value)
    claimed = body.pop("terminal_sha256")
    if (
        body.get("schema_version")
        != OFFICIAL_BASELINE_PROTECTED_TERMINAL_SCHEMA_VERSION
        or body.get("preparation_sha256") != preparation_sha256
        or body.get("state") != "known"
        or body.get("uncertainty_sha256") is not None
        or sha256_json(body) != claimed
    ):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline durable terminal identity differs"
        )
    terminal = OfficialBaselineProtectedTerminal(
        state="known",
        protected_action_result=_protected_result_from_document(
            body["protected_action_result"]
        ),
        protected_result_sha256=body["protected_result_sha256"],
        protected_terminal_receipt_ref=body[
            "protected_terminal_receipt_ref"
        ],
        protected_terminal_receipt_sha256=body[
            "protected_terminal_receipt_sha256"
        ],
        provider_request_ref=body["provider_request_ref"],
        model_provider_response_sha256=body[
            "model_provider_response_sha256"
        ],
    )
    if _protected_terminal_document(
        terminal,
        preparation_sha256=preparation_sha256,
    ) != dict(value):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline durable terminal reconstruction differs"
        )
    return terminal


class GatewayLocalProtectedActionBridge:
    """S3-claimed gateway implementation of the static action authority."""

    authority_identity_sha256 = PROTECTED_ACTION_AUTHORITY_SHA256

    def __init__(
        self,
        *,
        custody: S3OfficialBaselineDocumentCustody,
        executor: OfficialBaselinePreparedActionExecutor,
    ) -> None:
        if not isinstance(custody, S3OfficialBaselineDocumentCustody) or any(
            not callable(getattr(executor, method, None))
            for method in ("prepare", "execute_prepared", "reconcile")
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline gateway action bridge is incomplete"
            )
        self._custody = custody
        self._executor = executor

    def prepare(
        self,
        *,
        run_identity: Mapping[str, Any],
        unit_ref: str,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedPreparation:
        return self._executor.prepare(
            run_identity=run_identity,
            unit_ref=unit_ref,
            action=action,
        )

    @staticmethod
    def _claim_document(
        preparation: OfficialBaselineProtectedPreparation,
    ) -> dict[str, Any]:
        body = {
            "schema_version": (
                "leadpoet.research_lab.official_baseline_physical_claim.v1"
            ),
            "authority_identity_sha256": PROTECTED_ACTION_AUTHORITY_SHA256,
            "preparation": preparation.document(),
            "preparation_sha256": preparation.preparation_sha256,
        }
        return {**body, "claim_sha256": sha256_json(body)}

    def _known_readback(
        self, preparation: OfficialBaselineProtectedPreparation
    ) -> OfficialBaselineProtectedTerminal | None:
        value = self._custody.load_protected_action_terminal(
            preparation_sha256=preparation.preparation_sha256
        )
        if value is None:
            return None
        return _protected_terminal_from_document(
            value, preparation_sha256=preparation.preparation_sha256
        )

    def _persist_known(
        self,
        preparation: OfficialBaselineProtectedPreparation,
        terminal: OfficialBaselineProtectedTerminal,
    ) -> OfficialBaselineProtectedTerminal:
        self._custody.persist_protected_action_terminal(
            preparation_sha256=preparation.preparation_sha256,
            terminal=_protected_terminal_document(
                terminal,
                preparation_sha256=preparation.preparation_sha256,
            ),
        )
        readback = self._known_readback(preparation)
        if readback is None:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected terminal readback is missing"
            )
        return readback

    @staticmethod
    def _uncertain(
        preparation: OfficialBaselineProtectedPreparation,
        *,
        provider_request_ref: str | None,
    ) -> OfficialBaselineProtectedTerminal:
        return OfficialBaselineProtectedTerminal(
            state="uncertain",
            protected_action_result=None,
            protected_result_sha256=None,
            protected_terminal_receipt_ref=None,
            protected_terminal_receipt_sha256=None,
            provider_request_ref=provider_request_ref,
            model_provider_response_sha256=None,
            uncertainty_sha256=sha256_json(
                {
                    "schema_version": (
                        "leadpoet.research_lab.official_baseline_claim_uncertainty.v1"
                    ),
                    "preparation_sha256": preparation.preparation_sha256,
                    "protected_job_ref": preparation.protected_job_ref,
                }
            ),
        )

    def reconcile(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal:
        known = self._known_readback(preparation)
        if known is not None:
            return known
        claim = self._custody.load_protected_action_claim(
            preparation_sha256=preparation.preparation_sha256
        )
        if claim is None:
            return OfficialBaselineProtectedTerminal(
                state="not_started",
                protected_action_result=None,
                protected_result_sha256=None,
                protected_terminal_receipt_ref=None,
                protected_terminal_receipt_sha256=None,
                provider_request_ref=None,
                model_provider_response_sha256=None,
            )
        if claim != self._claim_document(preparation):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected physical claim differs"
            )
        try:
            recovered = self._executor.reconcile(
                preparation=preparation,
                action=action,
            )
        except Exception:
            return self._uncertain(
                preparation, provider_request_ref=None
            )
        if recovered.state == "known":
            return self._persist_known(preparation, recovered)
        return self._uncertain(
            preparation,
            provider_request_ref=recovered.provider_request_ref,
        )

    def execute_prepared(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal:
        claim = self._claim_document(preparation)
        claimed_new = self._custody.append_protected_action_claim(
            preparation_sha256=preparation.preparation_sha256,
            claim=claim,
        )
        if not claimed_new:
            return self.reconcile(preparation=preparation, action=action)
        try:
            terminal = self._executor.execute_prepared(
                preparation=preparation,
                action=action,
            )
        except Exception:
            return self.reconcile(preparation=preparation, action=action)
        if terminal.state == "known":
            return self._persist_known(preparation, terminal)
        return self._uncertain(
            preparation,
            provider_request_ref=terminal.provider_request_ref,
        )


class _ReservedOfficialBaselineDispatcher:
    """Reserve, reconcile, execute, and terminalize exactly one model action."""

    def __init__(
        self,
        *,
        authority: "AppendOnlyOfficialBaselineAuthority",
        run_identity: Mapping[str, Any],
        unit_ref: str,
        transitions: S3OfficialBaselineTransitionRepository,
    ) -> None:
        self._authority = authority
        self._run_identity = dict(run_identity)
        self._run_sha256 = sha256_json(self._run_identity)
        self._unit_ref = str(unit_ref)
        self._transitions = transitions

    def _preparation(
        self,
        action: Mapping[str, Any],
        compiled_dispatch: Mapping[str, Any] | None = None,
    ) -> OfficialBaselineProtectedPreparation:
        generation = self._authority._registration.protocol_generation
        if (
            action.get("action_type") in _PROVIDER_ACTION_TYPES
            and getattr(
                generation,
                "requires_raw_provider_response_custody",
                False,
            )
        ):
            _validate_compiled_provider_dispatch(
                protocol=self._authority._registration.protocol,
                action=action,
                compiled_dispatch=compiled_dispatch,
            )
        value = self._authority.bridge.prepare(
            run_identity=self._run_identity,
            unit_ref=self._unit_ref,
            action=action,
        )
        if not isinstance(value, OfficialBaselineProtectedPreparation):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected preparation is invalid"
            )
        identity = official_baseline_action_replay_identity(
            run_sha256=self._run_sha256,
            unit_ref=self._unit_ref,
            action=action,
        )
        sequence = action.get("sequence")
        action_type = str(action.get("action_type") or "")
        if (
            value.authority_identity_sha256 != self._authority.authority_identity_sha256
            or value.run_sha256 != self._run_sha256
            or value.unit_ref != self._unit_ref
            or value.action_idempotency_sha256 != identity["action_idempotency_sha256"]
            or value.action_sha256 != identity["action_sha256"]
            or value.request_fingerprint_sha256
            != identity["request_fingerprint_sha256"]
            or value.action_sequence != sequence
            or value.action_type != action_type
            or value.tool_id != str(action.get("tool_id") or "")
            or value.binding_contract_sha256
            != _prefixed_hash(
                action.get("binding_contract_sha256"),
                "official baseline action binding hash",
            )
            or type(value.action_sequence) is not int
            or not 0 <= value.action_sequence <= 9_999
            or type(value.call_cap) is not int
            or not 0 <= value.call_cap <= 100_000
            or type(value.credit_cap_microunits) is not int
            or not 0 <= value.credit_cap_microunits <= 100_000_000
            or type(value.timeout_ms) is not int
            or not 1 <= value.timeout_ms <= 900_000
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected preparation identity differs"
            )
        for claimed, label in (
            (value.authority_identity_sha256, "authority identity"),
            (value.run_sha256, "run hash"),
            (value.request_body_sha256, "request body hash"),
            (value.protected_request_sha256, "protected request hash"),
        ):
            _prefixed_hash(claimed, f"official baseline {label}")
        _safe_ref(value.protected_job_ref, "official baseline protected job ref")
        verifier = action_type in _VERIFIER_ACTION_TYPES
        provider = action_type in _PROVIDER_ACTION_TYPES
        if (
            not verifier
            and not provider
            or verifier
            and (value.call_cap != 0 or value.credit_cap_microunits != 0)
            or provider
            and value.call_cap < 1
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected action accounting differs"
            )
        return value

    def _authorization(
        self,
        *,
        action: Mapping[str, Any],
        preparation: OfficialBaselineProtectedPreparation,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        identity = official_baseline_action_replay_identity(
            run_sha256=self._run_sha256,
            unit_ref=self._unit_ref,
            action=action,
        )
        expected_frontier = self._transitions.expected_frontier_sha256(
            preparation.action_sequence
        )
        authorization = {
            "schema_version": OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION,
            **identity,
            "action_sequence": preparation.action_sequence,
            "action_type": preparation.action_type,
            "tool_id": preparation.tool_id,
            "binding_contract_sha256": preparation.binding_contract_sha256,
            "request_body_sha256": preparation.request_body_sha256,
            "call_cap": preparation.call_cap,
            "credit_cap_microunits": preparation.credit_cap_microunits,
            "timeout_ms": preparation.timeout_ms,
            "protected_job_ref": preparation.protected_job_ref,
            "protected_request_sha256": preparation.protected_request_sha256,
            "lease_holder_sha256": self._authority.lease_holder_sha256,
            "expected_frontier_sha256": expected_frontier,
        }
        authorization.pop(
            "schema_version", None
        )  # replay identity has its own schema version
        authorization = {
            "schema_version": OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION,
            **authorization,
        }
        return identity, authorization

    def _validate_terminal(
        self,
        terminal: OfficialBaselineProtectedTerminal,
        *,
        preparation: OfficialBaselineProtectedPreparation,
    ) -> OfficialBaselineProtectedTerminal:
        if not isinstance(
            terminal, OfficialBaselineProtectedTerminal
        ) or terminal.state not in {
            "not_started",
            "known",
            "uncertain",
        }:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected reconciliation is invalid"
            )
        if terminal.state == "not_started":
            if any(
                value is not None
                for value in (
                    terminal.protected_action_result,
                    terminal.protected_result_sha256,
                    terminal.protected_terminal_receipt_ref,
                    terminal.protected_terminal_receipt_sha256,
                    terminal.provider_request_ref,
                    terminal.model_provider_response_sha256,
                    terminal.uncertainty_sha256,
                )
            ):
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline authoritative absence contains custody"
                )
            return terminal
        if terminal.state == "uncertain":
            if (
                terminal.protected_action_result is not None
                or terminal.protected_result_sha256 is not None
                or terminal.protected_terminal_receipt_ref is not None
                or terminal.protected_terminal_receipt_sha256 is not None
                or terminal.model_provider_response_sha256 is not None
                or _HASH_RE.fullmatch(str(terminal.uncertainty_sha256 or "")) is None
                or (
                    terminal.provider_request_ref is not None
                    and _SAFE_REF_RE.fullmatch(terminal.provider_request_ref) is None
                )
            ):
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline protected uncertainty is invalid"
                )
            return terminal
        result = terminal.protected_action_result
        if not isinstance(result, ProtectedModelActionResult):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline known protected result is missing"
            )
        expected_result_sha = sha256_json(_protected_result_document(result))
        host = result.host_result
        response_sha = sha256_json(host.provider_response)
        if (
            terminal.protected_result_sha256 != expected_result_sha
            or terminal.model_provider_response_sha256 != response_sha
            or _HASH_RE.fullmatch(str(terminal.protected_terminal_receipt_sha256 or ""))
            is None
            or _SAFE_REF_RE.fullmatch(
                str(terminal.protected_terminal_receipt_ref or "")
            )
            is None
            or terminal.uncertainty_sha256 is not None
            or type(host.calls) is not int
            or host.calls < 0
            or isinstance(host.cost_credits, bool)
            or not isinstance(host.cost_credits, (int, float))
            or host.cost_credits < 0
            or isinstance(host.latency_ms, bool)
            or not isinstance(host.latency_ms, (int, float))
            or host.latency_ms < 0
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline known protected custody differs"
            )
        provider = preparation.action_type in _PROVIDER_ACTION_TYPES
        if provider:
            if (
                result.provider_receipt is None
                or not isinstance(result.replay_ref, Mapping)
                or terminal.provider_request_ref is None
                or host.calls < 1
                or not host.provider_receipt_ref
                or not host.provider_receipt_sha256
                or not host.provider_identity_sha256
                or host.model_provider_response_ingestion is not None
                or host.provider_action_receipt_sha256 is not None
            ):
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline provider result custody is incomplete"
                )
            _safe_ref(
                terminal.provider_request_ref,
                "official baseline provider request ref",
            )
        elif (
            result.provider_receipt is not None
            or result.replay_ref is not None
            or result.model_provider_response_ingestion is not None
            or terminal.provider_request_ref is not None
            or host.calls != 0
            or float(host.cost_credits) != 0.0
            or host.provider_receipt_ref is not None
            or host.provider_receipt_sha256 is not None
            or host.provider_identity_sha256 is not None
            or host.model_provider_response_ingestion is not None
            or host.provider_action_receipt_sha256 is not None
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline verifier result contains provider custody"
            )
        return terminal

    def _terminal_document(
        self,
        *,
        replay: Mapping[str, Any],
        terminal: OfficialBaselineProtectedTerminal,
        preparation: OfficialBaselineProtectedPreparation,
    ) -> dict[str, Any]:
        validated = self._validate_terminal(terminal, preparation=preparation)
        if validated.state != "known" or validated.protected_action_result is None:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline known terminal is unavailable"
            )
        host = validated.protected_action_result.host_result
        provider = preparation.action_type in _PROVIDER_ACTION_TYPES
        return {
            "schema_version": OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION,
            "attempt_key": replay["attempt_key"],
            "reservation_ref": replay["reservation_ref"],
            "lease_generation": replay["lease_generation"],
            "protected_job_ref": preparation.protected_job_ref,
            "protected_request_sha256": preparation.protected_request_sha256,
            "protected_result_sha256": validated.protected_result_sha256,
            "protected_terminal_receipt_ref": (
                validated.protected_terminal_receipt_ref
            ),
            "protected_terminal_receipt_sha256": (
                validated.protected_terminal_receipt_sha256
            ),
            "provider_request_ref": (
                validated.provider_request_ref if provider else None
            ),
            "provider_receipt_ref": host.provider_receipt_ref if provider else None,
            "provider_receipt_sha256": (
                _prefixed_hash(
                    host.provider_receipt_sha256,
                    "official baseline provider receipt hash",
                )
                if provider
                else None
            ),
            "provider_identity_sha256": (
                _prefixed_hash(
                    host.provider_identity_sha256,
                    "official baseline provider identity hash",
                )
                if provider
                else None
            ),
            "model_provider_response_sha256": (
                validated.model_provider_response_sha256
            ),
            "outcome": official_baseline_terminal_store_outcome(host.outcome),
            "call_count": host.calls,
            "cost_microunits": int(round(float(host.cost_credits) * 1_000_000)),
            "latency_ms": int(round(float(host.latency_ms))),
        }

    def _persist_known(
        self,
        *,
        identity: Mapping[str, Any],
        terminal: OfficialBaselineProtectedTerminal,
        preparation: OfficialBaselineProtectedPreparation,
    ) -> ProtectedModelActionResult:
        replay = self._authority.store.load_replay(identity=identity)
        if replay.get("state") not in {"reserved", "terminal_known"}:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline known terminal reservation is unavailable"
            )
        document = self._terminal_document(
            replay=replay,
            terminal=terminal,
            preparation=preparation,
        )
        self._authority.store.record_terminal_known(terminal=document)
        readback = self._authority.store.load_replay(identity=identity)
        expected = {
            key: value
            for key, value in document.items()
            if key
            not in {
                "schema_version",
                "reservation_ref",
                "lease_generation",
            }
        }
        if (
            readback.get("state") != "terminal_known"
            or any(readback.get(key) != value for key, value in expected.items())
            or readback.get("reservation_ref") != document["reservation_ref"]
            or readback.get("lease_generation") != document["lease_generation"]
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline terminal-known readback differs"
            )
        assert terminal.protected_action_result is not None
        protected = terminal.protected_action_result
        if preparation.action_type not in _PROVIDER_ACTION_TYPES:
            return protected
        receipt_sha256 = str(
            terminal.protected_terminal_receipt_sha256 or ""
        ).removeprefix("sha256:")
        if re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider action custody hash is invalid"
            )
        return replace(
            protected,
            host_result=replace(
                protected.host_result,
                model_provider_response_ingestion=(
                    protected.model_provider_response_ingestion
                ),
                provider_action_receipt_sha256=receipt_sha256,
            ),
        )

    def _persist_uncertain(
        self,
        *,
        identity: Mapping[str, Any],
        terminal: OfficialBaselineProtectedTerminal,
        preparation: OfficialBaselineProtectedPreparation,
    ) -> None:
        validated = self._validate_terminal(terminal, preparation=preparation)
        if validated.state != "uncertain":
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline terminal uncertainty is unavailable"
            )
        replay = self._authority.store.load_replay(identity=identity)
        if replay.get("state") == "terminal_uncertain":
            raise CommonModelExperimentRecoveryError(
                "official baseline protected call is terminal uncertain"
            )
        if replay.get("state") != "reserved":
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline uncertain reservation is unavailable"
            )
        self._authority.store.record_terminal_uncertain(
            uncertainty={
                "schema_version": (
                    OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION
                ),
                "attempt_key": identity["attempt_key"],
                "reservation_ref": replay["reservation_ref"],
                "lease_generation": replay["lease_generation"],
                "protected_job_ref": preparation.protected_job_ref,
                "protected_request_sha256": preparation.protected_request_sha256,
                "provider_request_ref": validated.provider_request_ref,
                "uncertainty_sha256": validated.uncertainty_sha256,
            }
        )
        raise CommonModelExperimentRecoveryError(
            "official baseline protected call is terminal uncertain"
        )

    def _reconcile(
        self,
        *,
        action: Mapping[str, Any],
        identity: Mapping[str, Any],
        preparation: OfficialBaselineProtectedPreparation,
        allow_execute_after_absence: bool,
    ) -> ProtectedModelActionResult:
        terminal = self._validate_terminal(
            self._authority.bridge.reconcile(
                preparation=preparation,
                action=action,
            ),
            preparation=preparation,
        )
        if terminal.state == "known":
            return self._persist_known(
                identity=identity,
                terminal=terminal,
                preparation=preparation,
            )
        if terminal.state == "uncertain":
            self._persist_uncertain(
                identity=identity,
                terminal=terminal,
                preparation=preparation,
            )
        if not allow_execute_after_absence:
            raise CommonModelExperimentRecoveryError(
                "official baseline protected terminal is absent"
            )
        return self._execute(
            action=action,
            identity=identity,
            preparation=preparation,
        )

    def _execute(
        self,
        *,
        action: Mapping[str, Any],
        identity: Mapping[str, Any],
        preparation: OfficialBaselineProtectedPreparation,
    ) -> ProtectedModelActionResult:
        try:
            terminal = self._validate_terminal(
                self._authority.bridge.execute_prepared(
                    preparation=preparation,
                    action=action,
                ),
                preparation=preparation,
            )
        except Exception:
            reconciliation = self._validate_terminal(
                self._authority.bridge.reconcile(
                    preparation=preparation,
                    action=action,
                ),
                preparation=preparation,
            )
            if reconciliation.state == "known":
                return self._persist_known(
                    identity=identity,
                    terminal=reconciliation,
                    preparation=preparation,
                )
            if reconciliation.state == "uncertain":
                self._persist_uncertain(
                    identity=identity,
                    terminal=reconciliation,
                    preparation=preparation,
                )
            raise
        if terminal.state == "known":
            return self._persist_known(
                identity=identity,
                terminal=terminal,
                preparation=preparation,
            )
        if terminal.state == "uncertain":
            self._persist_uncertain(
                identity=identity,
                terminal=terminal,
                preparation=preparation,
            )
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline execute returned authoritative absence"
        )

    def _run_action(
        self,
        action: Mapping[str, Any],
        *,
        compiled_dispatch: Mapping[str, Any] | None = None,
    ) -> ProtectedModelActionResult:
        preparation = self._preparation(action, compiled_dispatch)
        identity, authorization = self._authorization(
            action=action,
            preparation=preparation,
        )
        replay = self._authority.store.load_replay(identity=identity)
        state = replay.get("state")
        if state == "terminal_uncertain":
            raise CommonModelExperimentRecoveryError(
                "official baseline protected call is terminal uncertain"
            )
        if state == "terminal_known":
            return self._reconcile(
                action=action,
                identity=identity,
                preparation=preparation,
                allow_execute_after_absence=False,
            )
        if state == "reserved":
            # A replay row intentionally omits the holder. Re-submit the exact
            # authorization before reconciliation: migration 163 returns
            # reserved_existing only when the entire immutable document,
            # including lease holder and prepared job identities, is equal.
            reservation = self._authority.store.reserve_action(
                authorization=authorization
            )
            if reservation.get("disposition") != "reserved_existing":
                raise CommonModelExperimentRecoveryError(
                    "official baseline reservation is not same-holder replay"
                )
            return self._reconcile(
                action=action,
                identity=identity,
                preparation=preparation,
                allow_execute_after_absence=True,
            )
        if state == "absent":
            reservation = self._authority.store.reserve_action(
                authorization=authorization
            )
            disposition = reservation.get("disposition")
            if disposition == "reserved_new":
                return self._execute(
                    action=action,
                    identity=identity,
                    preparation=preparation,
                )
            if disposition == "reserved_existing":
                return self._reconcile(
                    action=action,
                    identity=identity,
                    preparation=preparation,
                    allow_execute_after_absence=True,
                )
            if disposition == "inflight":
                raise CommonModelExperimentRecoveryError(
                    "official baseline unit has a foreign inflight reservation"
                )
            replay = self._authority.store.load_replay(identity=identity)
            state = replay.get("state")
        if state == "terminal_known":
            return self._reconcile(
                action=action,
                identity=identity,
                preparation=preparation,
                allow_execute_after_absence=False,
            )
        if state == "terminal_uncertain":
            raise CommonModelExperimentRecoveryError(
                "official baseline protected call is terminal uncertain"
            )
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline reservation state is invalid"
        )

    def dispatch_provider_action(
        self,
        *,
        action: Mapping[str, Any],
        variant_id: str,
        unit_ref: str,
        compiled_dispatch: Mapping[str, Any] | None = None,
    ) -> ProtectedModelActionResult:
        if (
            variant_id != "official_baseline"
            or unit_ref != self._unit_ref
            or action.get("action_type") not in _PROVIDER_ACTION_TYPES
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider dispatch identity differs"
            )
        return self._run_action(
            action,
            compiled_dispatch=compiled_dispatch,
        )

    def replay_provider_action(
        self,
        *,
        action: Mapping[str, Any],
        variant_id: str,
        unit_ref: str,
        replay_ref: Mapping[str, Any],
        compiled_dispatch: Mapping[str, Any] | None = None,
    ) -> ProtectedModelActionResult:
        result = self.dispatch_provider_action(
            action=action,
            variant_id=variant_id,
            unit_ref=unit_ref,
            compiled_dispatch=compiled_dispatch,
        )
        if not isinstance(replay_ref, Mapping) or result.replay_ref != replay_ref:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected replay reference differs"
            )
        return result

    def _verify(
        self, *, action: Mapping[str, Any], unit_ref: str, action_type: str
    ) -> HostActionResult:
        if unit_ref != self._unit_ref or action.get("action_type") != action_type:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline verifier action identity differs"
            )
        return self._run_action(action).host_result

    def verify_company_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult:
        return self._verify(
            action=action, unit_ref=unit_ref, action_type="verify_company"
        )

    def verify_intent_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult:
        return self._verify(
            action=action, unit_ref=unit_ref, action_type="verify_intent"
        )

    def verify_contact_action(
        self, *, action: Mapping[str, Any], unit_ref: str
    ) -> HostActionResult:
        return self._verify(
            action=action, unit_ref=unit_ref, action_type="verify_contact"
        )


class AppendOnlyOfficialBaselineAuthority:
    """Concrete SQL + protected bridge + encrypted transition authority."""

    def __init__(
        self,
        *,
        context: OfficialBaselineDependencyContext,
        registration: ExactModelRunnerRegistration,
        store: OfficialBaselineAttemptStore,
        bridge: OfficialBaselineProtectedActionBridge,
        custody: S3OfficialBaselineDocumentCustody,
    ) -> None:
        context.validate()
        if (
            not isinstance(registration, ExactModelRunnerRegistration)
            or not isinstance(custody, S3OfficialBaselineDocumentCustody)
            or any(
                not callable(getattr(store, method, None))
                for method in (
                    "register_run",
                    "reserve_action",
                    "record_terminal_known",
                    "record_terminal_uncertain",
                    "load_replay",
                    "close_unit",
                    "load_frontier",
                )
            )
            or any(
                not callable(getattr(bridge, method, None))
                for method in ("prepare", "execute_prepared", "reconcile")
            )
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline concrete protected authority is incomplete"
            )
        expected = context.selection.selection_document[
            "protected_action_authority_sha256"
        ]
        if (
            expected != PROTECTED_ACTION_AUTHORITY_SHA256
            or str(getattr(bridge, "authority_identity_sha256", ""))
            != PROTECTED_ACTION_AUTHORITY_SHA256
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline protected bridge identity differs"
            )
        self._context = context
        self._registration = registration
        self.store = store
        self.bridge = bridge
        self._custody = custody
        self._authority_identity_sha256 = expected
        self.lease_holder_sha256 = sha256_json(
            {
                "schema_version": (
                    "leadpoet.research_lab.official_baseline_lease_holder.v1"
                ),
                "worker_ref": context.worker_ref,
            }
        )

    @property
    def authority_identity_sha256(self) -> str:
        return self._authority_identity_sha256

    def preflight_run(
        self,
        *,
        run_identity: Mapping[str, Any],
        registration: ExactModelRunnerRegistration,
    ) -> Mapping[str, Any]:
        if (
            registration is not self._registration
            or not isinstance(run_identity, Mapping)
            or run_identity.get("authority_identity_sha256")
            != self.authority_identity_sha256
            or run_identity.get("protocol_generation_sha256")
            != registration.protocol_generation.protocol_generation_sha256
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline protected preflight identity differs"
            )
        run_sha256 = sha256_json(dict(run_identity))
        registration_doc = {
            **dict(run_identity),
            "schema_version": OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION,
            "run_sha256": run_sha256,
        }
        result = self.store.register_run(registration=registration_doc)
        if result.get("run_sha256") != run_sha256:
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline run registration readback differs"
            )
        return {
            "schema_version": OFFICIAL_BASELINE_AUTHORITY_PREFLIGHT_SCHEMA_VERSION,
            "run_sha256": run_sha256,
            "artifact_key_sha256": sha256_json({"artifact_key": registration.key}),
            "protocol_generation_sha256": (
                registration.protocol_generation.protocol_generation_sha256
            ),
            "authority_identity_sha256": self.authority_identity_sha256,
            "ready": True,
        }

    def _transitions(
        self, *, run_identity: Mapping[str, Any], unit_ref: str
    ) -> S3OfficialBaselineTransitionRepository:
        run_sha256 = sha256_json(dict(run_identity))
        if (
            dict(run_identity).get("authority_identity_sha256")
            != self.authority_identity_sha256
            or _UNIT_RE.fullmatch(str(unit_ref or "")) is None
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline unit authority identity differs"
            )
        return self._custody.transition_repository(
            run_sha256=run_sha256,
            unit_ref=unit_ref,
            registration=self._registration,
            attempt_store=self.store,
        )

    def dispatcher_for_unit(
        self, *, run_identity: Mapping[str, Any], unit_ref: str
    ) -> _ReservedOfficialBaselineDispatcher:
        transitions = self._transitions(run_identity=run_identity, unit_ref=unit_ref)
        return _ReservedOfficialBaselineDispatcher(
            authority=self,
            run_identity=run_identity,
            unit_ref=unit_ref,
            transitions=transitions,
        )

    def transition_repository_for_unit(
        self, *, run_identity: Mapping[str, Any], unit_ref: str
    ) -> S3OfficialBaselineTransitionRepository:
        return self._transitions(run_identity=run_identity, unit_ref=unit_ref)

    def close_unit(self, *, completion: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.store.close_unit(closure=completion)

    def load_frontier(self, *, run_sha256: str, unit_ref: str) -> Mapping[str, Any]:
        return self.store.load_frontier(run_sha256=run_sha256, unit_ref=unit_ref)


def _production_custody() -> S3OfficialBaselineDocumentCustody:
    try:
        configuration = official_baseline_custody_configuration(os.environ)
    except Exception as exc:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline encrypted custody is unavailable"
        ) from exc
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - production dependency
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline encrypted custody client is unavailable"
        ) from exc
    return S3OfficialBaselineDocumentCustody(
        client=boto3.client("s3"),
        bucket=configuration["bucket"],
        prefix=configuration["prefix"],
        kms_key_id=configuration["kms_key_id"],
    )


def build_production_official_baseline_exact_dependencies(
    context: OfficialBaselineDependencyContext,
    store: OfficialBaselineAttemptStore,
) -> OfficialBaselineExactDependencies:
    """Load one fixed release handoff; absence is an exact-v3 startup block."""

    context.validate()
    custody = _production_custody()
    try:
        module = importlib.import_module(
            "gateway.research_lab.official_baseline_release_dependencies"
        )
        loader = getattr(module, "load_official_baseline_release_components")
    except Exception as exc:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed release component handoff is unavailable"
        ) from exc
    if not callable(loader):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed release component loader is invalid"
        )
    components = loader(context=context, custody=custody)
    if not isinstance(components, OfficialBaselineReleaseComponents):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed release components are invalid"
        )
    authority = AppendOnlyOfficialBaselineAuthority(
        context=context,
        registration=components.registration,
        store=store,
        bridge=components.protected_bridge,
        custody=custody,
    )
    dependencies = OfficialBaselineExactDependencies(
        registration=components.registration,
        projector=components.projector,
        protected_authority=authority,
        terminal_authority=custody,
    )
    return dependencies


__all__ = [
    "AppendOnlyOfficialBaselineAuthority",
    "GatewayLocalProtectedActionBridge",
    "OFFICIAL_BASELINE_CUSTODY_KMS_KEY_ENV",
    "OFFICIAL_BASELINE_CUSTODY_S3_PREFIX_ENV",
    "OFFICIAL_BASELINE_PROTECTED_PREPARATION_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_PROTECTED_RECONCILIATION_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_PROTECTED_TERMINAL_SCHEMA_VERSION",
    "PROTECTED_ACTION_AUTHORITY_SCHEMA_VERSION",
    "PROTECTED_ACTION_AUTHORITY_SHA256",
    "OfficialBaselineProtectedActionBridge",
    "OfficialBaselinePreparedActionExecutor",
    "OfficialBaselineProtectedAuthorityError",
    "OfficialBaselineProtectedPreparation",
    "OfficialBaselineProtectedTerminal",
    "OfficialBaselineReleaseComponents",
    "build_production_official_baseline_exact_dependencies",
]
