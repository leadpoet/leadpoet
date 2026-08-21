"""Static composition for the reviewed Research Lab routing product.

The experiment API and queue consumer are intentionally not dependency
injection surfaces.  This module is the one reviewed bootstrap seam.  A
deployment may install a composition only after it has supplied the exact
signed model authorities and the typed TEE job RPC clients.  Missing or
changed inputs leave both consumers fail closed.

This module does not contain model routing rules and it does not construct a
provider client.  The model owns the adapter contract and the coordinator/TEE
owns provider credentials and terminal execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
    build_routing_execution_envelope_v2,
    validate_routing_execution_envelope_v2,
)
from gateway.research_lab.routing_execution_authorization import (
    ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
    ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
)
from gateway.research_lab.routing_authority_bundle import (
    VerifiedRoutingAuthorityBundle,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
    VerifiedRoutingGoldLabels,
)
from gateway.research_lab.routing_experiment_api import (
    RoutingExperimentApiService,
    RoutingExperimentSpecAdmissionAuthority,
)
from gateway.research_lab.routing_experiment_worker import (
    ExactModelEvaluationAdapter,
    ExactModelRoutingRunFactory,
    RoutingExperimentRunFactory,
)
from gateway.research_lab.common_model_experiment import (
    ReviewedModelVerificationAuthority,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
)
from gateway.research_lab.routing_model_binding_issuer import (
    RoutingModelBindingObservationIssuerV2,
    ScoringJobExecutorV2,
)
from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingProviderCallAuthority,
    AttestedScoringV2RoutingProviderDispatchAuthority,
    ReviewedProviderBrokerRoutingRunner,
    RoutingProviderDispatchExecutor,
    _ROUTING_DISPATCH_EXECUTOR_TOKEN,
)
from gateway.research_lab.routing_provider_bindings import (
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
    ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
    routing_provider_dispatch_receipt_output_v2,
)
from research_lab.routing_experiments import (
    RoutingExperimentArtifactAuthority,
    RoutingExperimentError,
    RoutingExperimentV2Spec,
)
from research_lab.model_runner_protocol import ExactModelRunnerRegistry
from research_lab.canonical import sha256_json
from leadpoet_canonical.attested_v2 import validate_signed_execution_receipt


PRODUCT_COMPOSITION_ENV = "RESEARCH_LAB_ROUTING_PRODUCT_COMPOSITION"
PRODUCT_COMPOSITION_VERSION = "reviewed_v2"
MODEL_COMMIT_SHA_ENV = "RESEARCH_LAB_ROUTING_MODEL_COMMIT_SHA"
MODEL_ROUTING_CATALOG_HASH_ENV = "RESEARCH_LAB_ROUTING_MODEL_CATALOG_HASH"
SITE_PRODUCTION_MODEL_RELEASE_IDENTITY_ENV = (
    "RESEARCH_LAB_SITE_PRODUCTION_MODEL_RELEASE_IDENTITY_SHA256"
)
BINDING_CATALOG_MANIFEST_HASH_ENV = (
    "RESEARCH_LAB_ROUTING_BINDING_CATALOG_MANIFEST_HASH"
)
ROUTING_CONTRACT_HASH_ENV = "RESEARCH_LAB_ROUTING_CONTRACT_HASH"
AUTHORITY_BUNDLE_HASH_ENV = "RESEARCH_LAB_ROUTING_AUTHORITY_BUNDLE_HASH"
PROTECTED_RELEASE_RECEIPT_HASH_ENV = (
    "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_RECEIPT_HASH"
)
PROTECTED_RELEASE_COMMIT_SHA_ENV = (
    "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_COMMIT_SHA"
)
PROTECTED_RELEASE_PCR0_ENV = "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_PCR0"
PROTECTED_RELEASE_BUILD_MANIFEST_HASH_ENV = (
    "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_BUILD_MANIFEST_HASH"
)
PROTECTED_RELEASE_DEPENDENCY_LOCK_HASH_ENV = (
    "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_DEPENDENCY_LOCK_HASH"
)
PROTECTED_RELEASE_CONFIG_HASH_ENV = (
    "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_CONFIG_HASH"
)
PROTECTED_RELEASE_BOOT_IDENTITY_HASH_ENV = (
    "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_BOOT_IDENTITY_HASH"
)
PROTECTED_RELEASE_ENCLAVE_PUBKEY_ENV = (
    "RESEARCH_LAB_ROUTING_PROTECTED_RELEASE_ENCLAVE_PUBKEY"
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_ED25519_PUBLIC_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


class RoutingProductCompositionError(RuntimeError):
    """The reviewed routing product cannot be safely composed."""


class RoutingTeeJobRpc(Protocol):
    """The only host-facing interface allowed for protected routing jobs.

    Implementations are coordinator/scoring TEE RPC clients.  They must not
    expose a provider URL, body template, credential, or a provider singleton.
    """

    def submit_job(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def put_chunk(
        self, *, job_id: str, offset: int, data_b64: str, chunk_sha256: str
    ) -> Mapping[str, Any]: ...

    def seal(self, job_id: str) -> Mapping[str, Any]: ...

    def status(self, job_id: str) -> Mapping[str, Any]: ...

    def result(self, job_id: str) -> Mapping[str, Any]: ...

    def receipts(self, job_id: str) -> Sequence[Mapping[str, Any]]: ...


class RoutingProviderDispatchTeeRpc:
    """Concrete scoring job RPC wrapper for the fixed dispatch operation.

    This wrapper is intentionally not a protocol.  A coordinator broker, a
    generic ``.execute`` object, or a request callback cannot satisfy it.
    """

    __slots__ = ("_rpc",)

    def __init__(self, rpc: RoutingTeeJobRpc) -> None:
        if type(rpc) is RoutingProviderDispatchTeeRpc:
            self._rpc = rpc._rpc
            return
        if callable(getattr(rpc, "execute", None)):
            raise RoutingProductCompositionError(
                "routing dispatch RPC cannot expose a generic execute method"
            )
        self._rpc = _require_rpc(rpc, "provider dispatch")

    def submit_job(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(self._rpc.submit_job(manifest))

    def put_chunk(
        self, *, job_id: str, offset: int, data_b64: str, chunk_sha256: str
    ) -> Mapping[str, Any]:
        return dict(
            self._rpc.put_chunk(
                job_id=job_id,
                offset=offset,
                data_b64=data_b64,
                chunk_sha256=chunk_sha256,
            )
        )

    def seal(self, job_id: str) -> Mapping[str, Any]:
        return dict(self._rpc.seal(job_id))

    def status(self, job_id: str) -> Mapping[str, Any]:
        return dict(self._rpc.status(job_id))

    def result(self, job_id: str) -> Mapping[str, Any]:
        return dict(self._rpc.result(job_id))

    def receipts(self, job_id: str) -> Sequence[Mapping[str, Any]]:
        return tuple(dict(item) for item in self._rpc.receipts(job_id))


@dataclass(frozen=True)
class ReviewedRoutingProtectedAuthorities:
    """Protected authority objects made from bootstrap TEE job RPCs."""

    model_binding_observation_issuer: RoutingModelBindingObservationIssuerV2
    call_authorization_authority: AttestedScoringV2RoutingProviderCallAuthority
    dispatch_authority: AttestedScoringV2RoutingProviderDispatchAuthority


class _ScoringJobRpcAdapter(ScoringJobExecutorV2):
    """Adapt the narrow six-method RPC to the existing issuer protocol."""

    def __init__(self, rpc: RoutingTeeJobRpc) -> None:
        self._rpc = _require_rpc(rpc, "scoring")

    def submit_job(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(self._rpc.submit_job(manifest))

    def put_chunk(self, **kwargs: Any) -> Mapping[str, Any]:
        return dict(self._rpc.put_chunk(**kwargs))

    def seal_job(self, job_id: str) -> Mapping[str, Any]:
        return dict(self._rpc.seal(job_id))

    def get_status(self, job_id: str) -> Mapping[str, Any]:
        return dict(self._rpc.status(job_id))

    def get_result_chunk(
        self, *, job_id: str, offset: int, max_bytes: int
    ) -> Mapping[str, Any]:
        try:
            return dict(self._rpc.result(job_id, offset=offset, max_bytes=max_bytes))  # type: ignore[call-arg]
        except TypeError:
            # A result RPC may return a complete bounded result document.  It
            # is still the typed RPC client that owns the transport.
            if offset != 0:
                raise
            return dict(self._rpc.result(job_id))

    def get_receipts(self, job_id: str) -> Sequence[Mapping[str, Any]]:
        return tuple(dict(item) for item in self._rpc.receipts(job_id))


class _TeeJobRpcOperationExecutor(RoutingProviderDispatchExecutor):
    """Run one protected operation through the six-method job RPC only."""

    def __init__(
        self,
        rpc: RoutingTeeJobRpc,
        *,
        allowed_operation: str | None = None,
        allowed_purpose: str | None = None,
        required_parent_receipt_hashes: Sequence[str] | None = None,
        required_parent_receipt_hashes_factory: Callable[
            [Mapping[str, Any]], Sequence[str]
        ] | None = None,
        dispatch_token: object | None = None,
    ) -> None:
        self._rpc = _require_rpc(rpc, "operation")
        self._allowed_operation = allowed_operation
        self._allowed_purpose = allowed_purpose
        if required_parent_receipt_hashes is None:
            self._required_parent_receipt_hashes = None
        else:
            normalized = tuple(
                str(item or "").strip().lower()
                for item in required_parent_receipt_hashes
            )
            if (
                not normalized
                or any(not _SHA256_RE.fullmatch(item) for item in normalized)
                or len(set(normalized)) != len(normalized)
            ):
                raise RoutingProductCompositionError(
                    "routing protected operation parent ancestry is invalid"
                )
            self._required_parent_receipt_hashes = normalized
        if (
            required_parent_receipt_hashes is not None
            and required_parent_receipt_hashes_factory is not None
        ):
            raise RoutingProductCompositionError(
                "routing protected operation parent ancestry has two authorities"
            )
        if required_parent_receipt_hashes_factory is not None and not callable(
            required_parent_receipt_hashes_factory
        ):
            raise RoutingProductCompositionError(
                "routing protected operation parent ancestry factory is invalid"
            )
        self._required_parent_receipt_hashes_factory = (
            required_parent_receipt_hashes_factory
        )
        self._routing_dispatch_executor_token = dispatch_token

    @staticmethod
    def _summary(
        value: Mapping[str, Any],
        *,
        job_id: str,
        operation: str,
        purpose: str,
        manifest_hash: str,
        payload_hash: str,
        payload_size: int,
    ) -> tuple[str, int]:
        if not isinstance(value, Mapping) or value.get("job_id") != job_id:
            raise RoutingProductCompositionError(
                "routing protected operation job summary identity differs"
            )
        for name, expected in (("operation", operation), ("purpose", purpose)):
            if name in value and value.get(name) != expected:
                raise RoutingProductCompositionError(
                    "routing protected operation job summary identity differs"
                )
        if value.get("manifest_hash") != manifest_hash or (
            "payload_sha256" in value
            and value.get("payload_sha256") != payload_hash
        ):
            raise RoutingProductCompositionError(
                "routing protected operation job summary manifest differs"
            )
        if value.get("expected_bytes") != payload_size:
            raise RoutingProductCompositionError(
                "routing protected operation job summary payload size differs"
            )
        uploaded = value.get("uploaded_bytes")
        if (
            isinstance(uploaded, bool)
            or not isinstance(uploaded, int)
            or uploaded < 0
            or uploaded > payload_size
        ):
            raise RoutingProductCompositionError(
                "routing protected operation upload state is invalid"
            )
        state = str(value.get("state") or "")
        if state not in {
            "uploading", "queued", "running", "succeeded", "failed", "cancelled"
        }:
            raise RoutingProductCompositionError(
                "routing protected operation job state is invalid"
            )
        return state, uploaded

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        operation = str(request.get("operation") or "").strip()
        purpose = str(request.get("purpose") or "").strip()
        payload = request.get("payload")
        if not operation or not purpose or not isinstance(payload, Mapping):
            raise RoutingProductCompositionError("routing protected operation request is invalid")
        if self._allowed_operation is not None and operation != self._allowed_operation:
            raise RoutingProductCompositionError(
                "routing protected operation is not the fixed dispatch operation"
            )
        if self._allowed_purpose is not None and purpose != self._allowed_purpose:
            raise RoutingProductCompositionError(
                "routing protected operation is not the fixed dispatch purpose"
            )
        required_parent_hashes = self._required_parent_receipt_hashes
        if self._required_parent_receipt_hashes_factory is not None:
            try:
                dynamic_hashes = tuple(
                    str(item or "").strip().lower()
                    for item in self._required_parent_receipt_hashes_factory(request)
                )
            except Exception as exc:  # noqa: BLE001 - protected composition boundary
                raise RoutingProductCompositionError(
                    "routing protected operation parent ancestry is unavailable"
                ) from exc
            if (
                not dynamic_hashes
                or any(not _SHA256_RE.fullmatch(item) for item in dynamic_hashes)
                or len(set(dynamic_hashes)) != len(dynamic_hashes)
            ):
                raise RoutingProductCompositionError(
                    "routing protected operation parent ancestry is invalid"
                )
            required_parent_hashes = dynamic_hashes
        if required_parent_hashes is not None:
            requested_parent_hashes = request.get("parent_receipt_hashes")
            if requested_parent_hashes != list(required_parent_hashes):
                raise RoutingProductCompositionError(
                    "routing protected operation parent ancestry differs"
                )
        body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
        requested_job_id = str(request.get("job_id") or "").strip()
        job_id = requested_job_id or (
            "routing-job:" + hashlib.sha256(
                (operation + "\0" + purpose + "\0").encode() + body
            ).hexdigest()[:32]
        )
        payload_hash = "sha256:" + hashlib.sha256(body).hexdigest()
        manifest = {
            "schema_version": "leadpoet.routing_protected_operation_manifest.v2",
            "job_id": job_id,
            "operation": operation,
            "purpose": purpose,
            "payload_sha256": payload_hash,
            "payload_size_bytes": len(body),
            "parent_receipt_hashes": list(request.get("parent_receipt_hashes") or ()),
        }
        manifest_hash = sha256_json(manifest)
        summary = dict(self._rpc.submit_job(manifest))
        state, uploaded = self._summary(
            summary,
            job_id=job_id,
            operation=operation,
            purpose=purpose,
            manifest_hash=manifest_hash,
            payload_hash=payload_hash,
            payload_size=len(body),
        )
        if uploaded < len(body):
            chunk = body[uploaded:]
            summary = dict(
                self._rpc.put_chunk(
                    job_id=job_id,
                    offset=uploaded,
                    data_b64=base64.b64encode(chunk).decode("ascii"),
                    chunk_sha256="sha256:" + hashlib.sha256(chunk).hexdigest(),
                )
            )
            state, next_uploaded = self._summary(
                summary,
                job_id=job_id,
                operation=operation,
                purpose=purpose,
                manifest_hash=manifest_hash,
                payload_hash=payload_hash,
                payload_size=len(body),
            )
            if next_uploaded <= uploaded:
                raise RoutingProductCompositionError(
                    "routing protected operation upload did not advance"
                )
            uploaded = next_uploaded
        if uploaded != len(body):
            raise RoutingProductCompositionError(
                "routing protected operation upload is incomplete"
            )
        if state == "uploading":
            summary = dict(self._rpc.seal(job_id))
            state, uploaded = self._summary(
                summary,
                job_id=job_id,
                operation=operation,
                purpose=purpose,
                manifest_hash=manifest_hash,
                payload_hash=payload_hash,
                payload_size=len(body),
            )
        deadline = time.monotonic() + 60.0
        while state not in {"succeeded", "failed", "cancelled"}:
            if time.monotonic() >= deadline:
                raise RoutingProductCompositionError("routing protected operation timed out")
            time.sleep(0.05)
            summary = dict(self._rpc.status(job_id))
            state, uploaded = self._summary(
                summary,
                job_id=job_id,
                operation=operation,
                purpose=purpose,
                manifest_hash=manifest_hash,
                payload_hash=payload_hash,
                payload_size=len(body),
            )
        if state != "succeeded":
            raise RoutingProductCompositionError("routing protected operation failed")
        result = dict(self._rpc.result(job_id))
        if result.get("job_id") != job_id:
            raise RoutingProductCompositionError(
                "routing protected operation result job identity differs"
            )
        for name, expected in (("operation", operation), ("purpose", purpose)):
            if name in result and result.get(name) != expected:
                raise RoutingProductCompositionError(
                    "routing protected operation result identity differs"
                )
        if "state" in result and result.get("state") != "succeeded":
            raise RoutingProductCompositionError(
                "routing protected operation result state differs"
            )
        receipts = tuple(dict(item) for item in self._rpc.receipts(job_id))
        receipt = result.get("execution_receipt") or result.get("receipt")
        if not isinstance(receipt, Mapping) and receipts:
            receipt = receipts[-1]
        result_payload = result.get("result")
        output_root_matches = False
        if isinstance(result_payload, Mapping):
            receipt_payload = dict(result_payload)
            if (
                (operation, purpose)
                == (
                    ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
                    ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
                )
            ):
                receipt_payload = routing_provider_dispatch_receipt_output_v2(
                    result_payload
                )
            output_root_matches = (
                receipt.get("output_root") == sha256_json(receipt_payload)
                if isinstance(receipt, Mapping)
                else False
            )
            if (
                (operation, purpose)
                == (
                    ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
                    ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
                )
                and _SHA256_RE.fullmatch(
                    str(result_payload.get("output_root") or "")
                )
            ):
                output_root_matches = output_root_matches or (
                    receipt.get("output_root") == result_payload["output_root"]
                    if isinstance(receipt, Mapping)
                    else False
                )
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(result_payload, Mapping)
            or receipt.get("job_id") != job_id
            or receipt.get("purpose") != purpose
            or receipt.get("status") != "succeeded"
            or receipt.get("input_root") != sha256_json(dict(payload))
            or not output_root_matches
            or receipt.get("parent_receipt_hashes")
            != manifest["parent_receipt_hashes"]
        ):
            raise RoutingProductCompositionError("routing protected operation result is malformed")
        receipt_hash = str(receipt.get("receipt_hash") or "")
        if (
            not _SHA256_RE.fullmatch(receipt_hash)
            or sum(item == dict(receipt) for item in receipts) != 1
        ):
            raise RoutingProductCompositionError(
                "routing protected operation receipt identity differs"
            )
        return {
            "status": "succeeded",
            "operation": operation,
            "purpose": purpose,
            "result": dict(result_payload),
            "execution_receipt": dict(receipt),
        }

    def replay_protected_model_result(
        self,
        replay_ref: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Read one completed dispatch job without submitting another job."""

        if (
            self._allowed_operation != ROUTING_PROVIDER_DISPATCH_OPERATION_V2
            or self._allowed_purpose != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
            or self._routing_dispatch_executor_token
            is not _ROUTING_DISPATCH_EXECUTOR_TOKEN
        ):
            raise RoutingProductCompositionError(
                "routing protected Model replay executor is not fixed"
            )
        expected_fields = {
            "schema_version",
            "protected_dispatch_job_id",
            "terminal_receipt_hash",
            "model_provider_response_sha256",
            "model_completion_contract_hash",
        }
        if (
            not isinstance(replay_ref, Mapping)
            or set(replay_ref) != expected_fields
            or replay_ref.get("schema_version")
            != "leadpoet.research_lab.protected_model_replay_ref.v1"
        ):
            raise RoutingProductCompositionError(
                "routing protected Model replay reference is invalid"
            )
        job_id = str(replay_ref.get("protected_dispatch_job_id") or "")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}", job_id)
            or any(
                not _SHA256_RE.fullmatch(str(replay_ref.get(name) or ""))
                for name in (
                    "terminal_receipt_hash",
                    "model_provider_response_sha256",
                    "model_completion_contract_hash",
                )
            )
        ):
            raise RoutingProductCompositionError(
                "routing protected Model replay commitments are invalid"
            )
        result = dict(self._rpc.result(job_id))
        receipts = tuple(dict(item) for item in self._rpc.receipts(job_id))
        result_payload = result.get("result")
        receipt = result.get("execution_receipt") or result.get("receipt")
        if not isinstance(receipt, Mapping) and receipts:
            receipt = receipts[-1]
        if (
            result.get("job_id") != job_id
            or not isinstance(result_payload, Mapping)
            or not isinstance(receipt, Mapping)
            or receipt.get("job_id") != job_id
            or receipt.get("role") != "gateway_scoring"
            or receipt.get("purpose")
            != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
            or receipt.get("status") != "succeeded"
            or receipt.get("receipt_hash")
            != replay_ref["terminal_receipt_hash"]
            or receipt.get("output_root")
            != sha256_json(
                routing_provider_dispatch_receipt_output_v2(result_payload)
            )
            or result_payload.get("model_provider_response_sha256")
            != replay_ref["model_provider_response_sha256"]
            or result_payload.get("model_completion_contract_hash")
            != replay_ref["model_completion_contract_hash"]
            or sum(item == dict(receipt) for item in receipts) != 1
        ):
            raise RoutingProductCompositionError(
                "routing protected Model replay result differs"
            )
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise RoutingProductCompositionError(
                "routing protected Model replay receipt is invalid"
            ) from exc
        return {
            "status": "succeeded",
            "operation": ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
            "purpose": ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
            "result": dict(result_payload),
            "execution_receipt": dict(receipt),
        }


def _routing_dispatch_parent_receipt_hashes(
    request: Mapping[str, Any],
) -> tuple[str, ...]:
    """Resolve the dispatch parent from the signed authorization receipt.

    The authorization receipt is created per call, so a static composition
    tuple cannot bind this operation to the correct parent.  The signed
    contract already carries that receipt in ``authorization_proof`` and the
    dispatch authority sends the same hash as the job parent list.  Keeping
    this check in the fixed-operation executor prevents a malformed or
    substituted request from reaching the TEE job RPC.
    """

    payload = request.get("payload")
    proof = payload.get("authorization_proof") if isinstance(payload, Mapping) else None
    receipt = proof.get("authorization_receipt") if isinstance(proof, Mapping) else None
    receipt_hash = str(receipt.get("receipt_hash") or "") if isinstance(receipt, Mapping) else ""
    if not _SHA256_RE.fullmatch(receipt_hash):
        raise RoutingProductCompositionError(
            "routing dispatch authorization receipt hash is unavailable"
        )
    return (receipt_hash,)


def build_attested_protected_authorities(
    inputs: ReviewedRoutingReleaseInputs,
    *,
    environment: Mapping[str, str] | None = None,
) -> ReviewedRoutingProtectedAuthorities:
    """Construct model, authorization, and dispatch authorities once at boot."""

    validate_reviewed_release_inputs(inputs, environment=environment)
    _require_rpc(inputs.scoring_job_rpc, "model observation")
    _require_rpc(inputs.call_authorization_job_rpc, "call authorization")
    dispatch_rpc = (
        inputs.dispatch_job_rpc
        if type(inputs.dispatch_job_rpc) is RoutingProviderDispatchTeeRpc
        else RoutingProviderDispatchTeeRpc(inputs.dispatch_job_rpc)
    )
    return ReviewedRoutingProtectedAuthorities(
        model_binding_observation_issuer=RoutingModelBindingObservationIssuerV2(
            executor=_ScoringJobRpcAdapter(inputs.scoring_job_rpc)
        ),
        call_authorization_authority=AttestedScoringV2RoutingProviderCallAuthority(
            executor=_TeeJobRpcOperationExecutor(
                inputs.call_authorization_job_rpc,
                allowed_operation=ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
                allowed_purpose=ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
                required_parent_receipt_hashes=(
                    inputs.protected_release_receipt["receipt_hash"],
                    inputs.model_binding_observation.observation_receipt_hash,
                ),
                dispatch_token=_ROUTING_DISPATCH_EXECUTOR_TOKEN,
            )
        ),
        dispatch_authority=AttestedScoringV2RoutingProviderDispatchAuthority(
            executor=_TeeJobRpcOperationExecutor(
                dispatch_rpc,
                allowed_operation=ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
                allowed_purpose=ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
                required_parent_receipt_hashes_factory=(
                    _routing_dispatch_parent_receipt_hashes
                ),
                dispatch_token=_ROUTING_DISPATCH_EXECUTOR_TOKEN,
            ),
            protected_release_receipt=inputs.protected_release_receipt,
        ),
    )


@dataclass(frozen=True)
class ReviewedRoutingReleaseInputs:
    """Verified release authorities captured at process bootstrap."""

    artifact_lineage: VerifiedRoutingArtifactLineage
    binding_catalog: VerifiedRoutingBindingCatalog
    unit_dataset: VerifiedRoutingUnitDataset
    authority_bundle: VerifiedRoutingAuthorityBundle
    gold_labels: VerifiedRoutingGoldLabels
    model_binding_observation: VerifiedRoutingModelBindingRequirements
    protected_release_receipt: Mapping[str, Any]
    artifact_authority: RoutingExperimentArtifactAuthority
    model_runner_registry: ExactModelRunnerRegistry
    model_verifier: ReviewedModelVerificationAuthority
    evaluation_adapter: ExactModelEvaluationAdapter
    scoring_job_rpc: RoutingTeeJobRpc
    call_authorization_job_rpc: RoutingTeeJobRpc
    dispatch_job_rpc: RoutingProviderDispatchTeeRpc


def _require_rpc(value: Any, name: str) -> RoutingTeeJobRpc:
    if value is None:
        raise RoutingProductCompositionError(f"routing {name} TEE job RPC is required")
    for method in ("submit_job", "put_chunk", "seal", "status", "result", "receipts"):
        if not callable(getattr(value, method, None)):
            raise RoutingProductCompositionError(
                f"routing {name} TEE job RPC is missing {method}"
            )
    return value


def _require_exact_model_authorities(inputs: ReviewedRoutingReleaseInputs) -> None:
    if not isinstance(inputs.model_runner_registry, ExactModelRunnerRegistry):
        raise RoutingProductCompositionError(
            "exact Model runner registry is unavailable"
        )
    try:
        inputs.model_runner_registry.preflight_all()
    except Exception as exc:
        raise RoutingProductCompositionError(
            "exact Model runner preflight failed"
        ) from exc
    for method in ("verify_company", "verify_intent", "verify_contact"):
        if not callable(getattr(inputs.model_verifier, method, None)):
            raise RoutingProductCompositionError(
                "reviewed Model verifier is unavailable"
            )
    if not callable(
        getattr(inputs.evaluation_adapter, "build_decision_receipts", None)
    ) or not callable(
        getattr(inputs.evaluation_adapter, "build_evaluation", None)
    ):
        raise RoutingProductCompositionError(
            "canonical Model evaluation adapter is unavailable"
        )


def _require_hash(value: Any, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise RoutingProductCompositionError(f"routing {name} is invalid")
    return normalized


def validate_reviewed_release_inputs(
    inputs: ReviewedRoutingReleaseInputs,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Check the exact release identity before API or queue construction."""

    env = os.environ if environment is None else environment
    if str(env.get(PRODUCT_COMPOSITION_ENV, "")).strip() != PRODUCT_COMPOSITION_VERSION:
        raise RoutingProductCompositionError(
            f"{PRODUCT_COMPOSITION_ENV} must be {PRODUCT_COMPOSITION_VERSION}"
        )
    commit = str(env.get(MODEL_COMMIT_SHA_ENV, "")).strip().lower()
    if not _COMMIT_RE.fullmatch(commit) or commit != inputs.artifact_lineage.commit_sha:
        raise RoutingProductCompositionError("routing model commit identity differs")
    model_catalog_hash = _require_hash(
        env.get(MODEL_ROUTING_CATALOG_HASH_ENV), "model routing catalog hash"
    )
    if model_catalog_hash != inputs.artifact_lineage.routing_catalog_hash:
        raise RoutingProductCompositionError(
            "routing model catalog identity differs"
        )
    _require_hash(
        env.get(SITE_PRODUCTION_MODEL_RELEASE_IDENTITY_ENV),
        "Site production model release identity",
    )
    binding_catalog_hash = _require_hash(
        env.get(BINDING_CATALOG_MANIFEST_HASH_ENV),
        "binding catalog manifest hash",
    )
    if binding_catalog_hash != inputs.binding_catalog.manifest_hash:
        raise RoutingProductCompositionError(
            "routing binding catalog identity differs"
        )
    contract_hash = _require_hash(env.get(ROUTING_CONTRACT_HASH_ENV), "contract hash")
    if contract_hash != inputs.artifact_lineage.routing_contract_hash:
        raise RoutingProductCompositionError("routing contract identity differs")
    authority_bundle_hash = _require_hash(
        env.get(AUTHORITY_BUNDLE_HASH_ENV), "authority bundle hash"
    )
    bundle = inputs.authority_bundle
    if not isinstance(bundle, VerifiedRoutingAuthorityBundle):
        raise RoutingProductCompositionError(
            "routing verified authority bundle is unavailable"
        )
    if bundle.bundle_hash != authority_bundle_hash:
        raise RoutingProductCompositionError(
            "routing authority bundle identity differs"
        )
    if (
        bundle.artifact_lineage != inputs.artifact_lineage
        or bundle.artifact_lineage.identity_hash()
        != inputs.artifact_lineage.identity_hash()
    ):
        raise RoutingProductCompositionError(
            "routing authority bundle artifact lineage differs"
        )
    if (
        bundle.binding_catalog != inputs.binding_catalog
        or bundle.binding_catalog.manifest_hash != binding_catalog_hash
    ):
        raise RoutingProductCompositionError(
            "routing authority bundle binding catalog differs"
        )
    if (
        bundle.unit_dataset != inputs.unit_dataset
        or bundle.unit_dataset.manifest_hash != inputs.unit_dataset.manifest_hash
        or bundle.unit_dataset.unit_set_hash != inputs.unit_dataset.unit_set_hash
    ):
        raise RoutingProductCompositionError(
            "routing authority bundle unit dataset differs"
        )
    _require_exact_model_authorities(inputs)
    if not callable(getattr(inputs.artifact_authority, "verify", None)):
        raise RoutingProductCompositionError(
            "routing artifact authority is unavailable"
        )
    _require_rpc(inputs.scoring_job_rpc, "model observation")
    _require_rpc(inputs.call_authorization_job_rpc, "call authorization")
    _require_rpc(inputs.dispatch_job_rpc, "provider dispatch")
    if not isinstance(inputs.protected_release_receipt, Mapping):
        raise RoutingProductCompositionError(
            "routing protected release receipt is unavailable"
        )
    release = inputs.protected_release_receipt
    try:
        validate_signed_execution_receipt(release)
    except Exception as exc:
        raise RoutingProductCompositionError(
            "routing protected release receipt is invalid"
        ) from exc
    if (
        release.get("role") != "gateway_scoring"
        or release.get("purpose") != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
        or release.get("status") != "succeeded"
        or release.get("failure_code") not in (None, "")
    ):
        raise RoutingProductCompositionError(
            "routing protected release receipt scope differs"
        )
    protected_hash_pins = {
        "receipt_hash": PROTECTED_RELEASE_RECEIPT_HASH_ENV,
        "build_manifest_hash": PROTECTED_RELEASE_BUILD_MANIFEST_HASH_ENV,
        "dependency_lock_hash": PROTECTED_RELEASE_DEPENDENCY_LOCK_HASH_ENV,
        "config_hash": PROTECTED_RELEASE_CONFIG_HASH_ENV,
        "boot_identity_hash": PROTECTED_RELEASE_BOOT_IDENTITY_HASH_ENV,
    }
    for receipt_field, environment_name in protected_hash_pins.items():
        if release.get(receipt_field) != _require_hash(
            env.get(environment_name), environment_name
        ):
            raise RoutingProductCompositionError(
                "routing protected release receipt identity differs"
            )
    protected_commit = str(env.get(PROTECTED_RELEASE_COMMIT_SHA_ENV, "")).strip().lower()
    protected_pcr0 = str(env.get(PROTECTED_RELEASE_PCR0_ENV, "")).strip().lower()
    protected_pubkey = str(
        env.get(PROTECTED_RELEASE_ENCLAVE_PUBKEY_ENV, "")
    ).strip().lower()
    if (
        not _COMMIT_RE.fullmatch(protected_commit)
        or not _PCR0_RE.fullmatch(protected_pcr0)
        or not _ED25519_PUBLIC_KEY_RE.fullmatch(protected_pubkey)
        or release.get("commit_sha") != protected_commit
        or release.get("pcr0") != protected_pcr0
        or release.get("enclave_pubkey") != protected_pubkey
    ):
        raise RoutingProductCompositionError(
            "routing protected release receipt identity differs"
        )


@dataclass(frozen=True)
class ReviewedRoutingAdmissionAuthority(RoutingExperimentSpecAdmissionAuthority):
    """Build and validate the immutable envelope before the first SQL write."""

    inputs: ReviewedRoutingReleaseInputs
    site_production_model_release_identity_sha256: str

    def admit(self, spec: RoutingExperimentV2Spec) -> RoutingExperimentExecutionEnvelopeV2:
        try:
            envelope = build_routing_execution_envelope_v2(
                spec=spec,
                artifact_lineage=self.inputs.artifact_lineage,
                binding_catalog=self.inputs.binding_catalog,
                unit_dataset=self.inputs.unit_dataset,
                gold_labels=self.inputs.gold_labels,
                model_binding_observation=self.inputs.model_binding_observation,
            )
            validate_routing_execution_envelope_v2(
                spec=spec,
                envelope=envelope,
                binding_catalog=self.inputs.binding_catalog,
            )
            bindings_by_tool = {
                binding.tool_id: binding
                for binding in spec.provider_bindings
            }
            if len(bindings_by_tool) != len(spec.provider_bindings):
                raise RoutingProductCompositionError(
                    "routing provider tool bindings are duplicated"
                )
            if len(spec.variants) < 2:
                raise RoutingProductCompositionError(
                    "exact baseline and challenger artifacts are required"
                )
            registrations = {}
            for variant in spec.variants:
                registration = self.inputs.model_runner_registry.resolve(
                    variant.artifact.to_dict()
                )
                registration.preflight()
                registration.validate_variant_audit_payload(
                    variant.routing_payload
                )
                registrations[variant.variant_id] = registration
                raw_bindings = registration.host_capability_manifest.get(
                    "bindings"
                )
                if not isinstance(raw_bindings, Sequence):
                    raise RoutingProductCompositionError(
                        "Model host capability bindings are invalid"
                    )
                for model_binding in raw_bindings:
                    if (
                        not isinstance(model_binding, Mapping)
                        or model_binding.get("available") is not True
                        or model_binding.get("action_type")
                        not in {
                            "execute_candidate_tool",
                            "execute_intent_tool",
                            "execute_contact_tool",
                        }
                    ):
                        continue
                    tool_id = str(model_binding.get("tool_id") or "")
                    binding = bindings_by_tool.get(tool_id)
                    if binding is None or binding.execution_contract_hash != (
                        "sha256:"
                        + str(
                            model_binding.get("binding_contract_sha256")
                            or ""
                        )
                    ):
                        raise RoutingProductCompositionError(
                            "Model provider binding registration differs"
                        )
            baseline = registrations.get(spec.baseline_variant_id)
            if baseline is None:
                raise RoutingProductCompositionError(
                    "exact baseline artifact is unavailable"
                )
            if baseline.artifact_identity.get("branch") != "main":
                raise RoutingProductCompositionError(
                    "baseline must use the Site-selected main artifact"
                )
            if baseline.protocol.release_identity.get(
                "release_identity_sha256"
            ) != self.site_production_model_release_identity_sha256:
                raise RoutingProductCompositionError(
                    "baseline differs from the Site production model release"
                )
            artifact_keys = set()
            for variant in spec.variants:
                registration = registrations[variant.variant_id]
                artifact_key = registration.key
                if (
                    variant.variant_id != spec.baseline_variant_id
                    and artifact_key == baseline.key
                ):
                    raise RoutingProductCompositionError(
                        "each challenger must use a distinct exact Model artifact"
                    )
                if (
                    variant.variant_id != spec.baseline_variant_id
                    and registration.artifact_identity.get("branch")
                    != "leadpoet-lab"
                ):
                    raise RoutingProductCompositionError(
                        "challenger must use a leadpoet-lab artifact"
                    )
                if artifact_key in artifact_keys:
                    raise RoutingProductCompositionError(
                        "exact Model artifact is duplicated across variants"
                    )
                artifact_keys.add(artifact_key)
            return envelope
        except Exception as exc:  # noqa: BLE001 - admission must fail closed
            if isinstance(exc, RoutingProductCompositionError):
                raise
            raise RoutingProductCompositionError(
                "routing model admission rejected the experiment"
            ) from exc


def build_reviewed_admission_authority(
    inputs: ReviewedRoutingReleaseInputs,
    *,
    environment: Mapping[str, str] | None = None,
) -> ReviewedRoutingAdmissionAuthority:
    """Build the API admission authority from verified bootstrap inputs."""

    validate_reviewed_release_inputs(inputs, environment=environment)
    env = os.environ if environment is None else environment
    site_release = _require_hash(
        env.get(SITE_PRODUCTION_MODEL_RELEASE_IDENTITY_ENV),
        "Site production model release identity",
    )
    return ReviewedRoutingAdmissionAuthority(
        inputs=inputs,
        site_production_model_release_identity_sha256=(
            site_release.removeprefix("sha256:")
        ),
    )


def build_exact_model_runner_factory(
    *,
    inputs: ReviewedRoutingReleaseInputs,
    reviewed_runner_factory: Callable[
        [RoutingExperimentV2Spec], ReviewedProviderBrokerRoutingRunner
    ],
    billing_rollup_factory: Callable[[RoutingExperimentV2Spec], Callable[..., Mapping[str, Any]]],
    execution_envelope_factory: Callable[
        [RoutingExperimentV2Spec], RoutingExperimentExecutionEnvelopeV2
    ],
    environment: Mapping[str, str] | None = None,
) -> RoutingExperimentRunFactory:
    """Build the sole PR274 run factory after static trust checks.

    ``reviewed_runner_factory`` is a release-owned constructor. It must close
    over typed TEE RPC authorities created at bootstrap. The function does not
    accept a broker, URL, request body, credential, import path, or provider
    client from a spec or queue row.
    """

    validate_reviewed_release_inputs(inputs, environment=environment)
    if not callable(reviewed_runner_factory):
        raise RoutingProductCompositionError(
            "reviewed routing runner factory is unavailable"
        )
    factory = ExactModelRoutingRunFactory(
        registry=inputs.model_runner_registry,
        gold_labels=inputs.gold_labels.labels,
        unit_dataset=inputs.unit_dataset,
        reviewed_runner_factory=reviewed_runner_factory,
        verifier=inputs.model_verifier,
        evaluation_adapter=inputs.evaluation_adapter,
        billing_rollup_factory=billing_rollup_factory,
        execution_envelope_factory=execution_envelope_factory,
    )
    if factory.name != "exact_model_runner_v3":
        raise RoutingProductCompositionError("reviewed routing factory name is invalid")
    return factory


def build_attested_provider_broker_factory(
    **kwargs: Any,
) -> RoutingExperimentRunFactory:
    """Compatibility alias for callers that have not renamed bootstrap code.

    The returned factory is still ``exact_model_runner_v3``. This alias does
    not install or expose the legacy provider-broker runner.
    """

    return build_exact_model_runner_factory(**kwargs)


@dataclass(frozen=True)
class ReviewedRoutingProductComposition:
    """The immutable objects installed by process bootstrap."""

    api_service: RoutingExperimentApiService
    run_factory: RoutingExperimentRunFactory
    protected_authorities: ReviewedRoutingProtectedAuthorities

    @property
    def factory_registry(self) -> Mapping[str, RoutingExperimentRunFactory]:
        if getattr(self.run_factory, "name", None) != "exact_model_runner_v3":
            raise RoutingProductCompositionError("reviewed routing factory name is invalid")
        return {"exact_model_runner_v3": self.run_factory}


def build_reviewed_routing_product(
    *,
    inputs: ReviewedRoutingReleaseInputs,
    reviewed_runner_factory: Callable[
        [RoutingExperimentV2Spec], ReviewedProviderBrokerRoutingRunner
    ],
    billing_rollup_factory: Callable[[RoutingExperimentV2Spec], Callable[..., Mapping[str, Any]]],
    execution_envelope_factory: Callable[
        [RoutingExperimentV2Spec], RoutingExperimentExecutionEnvelopeV2
    ],
    store_factory: Callable[[], Any],
    environment: Mapping[str, str] | None = None,
) -> ReviewedRoutingProductComposition:
    """Compose the API service and one named consumer factory."""

    authority = build_reviewed_admission_authority(inputs, environment=environment)
    protected_authorities = build_attested_protected_authorities(
        inputs, environment=environment
    )
    run_factory = build_exact_model_runner_factory(
        inputs=inputs,
        reviewed_runner_factory=reviewed_runner_factory,
        billing_rollup_factory=billing_rollup_factory,
        execution_envelope_factory=execution_envelope_factory,
        environment=environment,
    )
    return ReviewedRoutingProductComposition(
        api_service=RoutingExperimentApiService(
            store_factory=store_factory,
            admission_authority=authority,
        ),
        run_factory=run_factory,
        protected_authorities=protected_authorities,
    )


def bootstrap_reviewed_routing_product(
    *,
    environment: Mapping[str, str] | None = None,
    inputs: ReviewedRoutingReleaseInputs | None = None,
    reviewed_runner_factory: Callable[
        [RoutingExperimentV2Spec], ReviewedProviderBrokerRoutingRunner
    ] | None = None,
    billing_rollup_factory: Callable[[RoutingExperimentV2Spec], Callable[..., Mapping[str, Any]]] | None = None,
    execution_envelope_factory: Callable[
        [RoutingExperimentV2Spec], RoutingExperimentExecutionEnvelopeV2
    ] | None = None,
    store_factory: Callable[[], Any] | None = None,
) -> ReviewedRoutingProductComposition:
    """Bootstrap hook used by a reviewed deployment composition.

    This function intentionally has no environment-based dynamic loader.  A
    release must pass typed, already-verified authorities and constructors.
    If the model adapter worker is not exported by the exact model artifact,
    callers receive a typed failure before any store or queue construction.
    """

    if inputs is None:
        raise RoutingProductCompositionError(
            "reviewed routing bootstrap inputs are unavailable; exact Model runner is required"
        )
    if not callable(reviewed_runner_factory):
        raise RoutingProductCompositionError(
            "reviewed routing bootstrap runner factory is unavailable"
        )
    if not callable(billing_rollup_factory):
        raise RoutingProductCompositionError(
            "reviewed routing bootstrap billing authority is unavailable"
        )
    if not callable(execution_envelope_factory):
        raise RoutingProductCompositionError(
            "reviewed routing bootstrap envelope factory is unavailable"
        )
    if not callable(store_factory):
        raise RoutingProductCompositionError(
            "reviewed routing bootstrap store factory is unavailable"
        )
    return build_reviewed_routing_product(
        inputs=inputs,
        reviewed_runner_factory=reviewed_runner_factory,
        billing_rollup_factory=billing_rollup_factory,
        execution_envelope_factory=execution_envelope_factory,
        store_factory=store_factory,
        environment=environment,
    )


def install_reviewed_routing_product(
    composition: ReviewedRoutingProductComposition,
    *,
    app: Any,
) -> None:
    """Install one already-built composition into API and consumer state."""

    if not isinstance(composition, ReviewedRoutingProductComposition):
        raise RoutingProductCompositionError(
            "reviewed routing product composition is invalid"
        )
    if getattr(app, "state", None) is None:
        raise RoutingProductCompositionError("routing product app state is unavailable")
    from gateway.research_lab.routing_experiment_api import (
        install_routing_experiment_api_service,
    )
    from gateway.research_lab.routing_execution_consumer import (
        install_reviewed_routing_factory_registry,
    )

    install_routing_experiment_api_service(composition.api_service, app=app)
    install_reviewed_routing_factory_registry(composition.factory_registry)
    app.state.reviewed_routing_product_composition = composition


__all__ = [
    "PRODUCT_COMPOSITION_ENV",
    "PRODUCT_COMPOSITION_VERSION",
    "MODEL_COMMIT_SHA_ENV",
    "MODEL_ROUTING_CATALOG_HASH_ENV",
    "SITE_PRODUCTION_MODEL_RELEASE_IDENTITY_ENV",
    "BINDING_CATALOG_MANIFEST_HASH_ENV",
    "ROUTING_CONTRACT_HASH_ENV",
    "AUTHORITY_BUNDLE_HASH_ENV",
    "PROTECTED_RELEASE_RECEIPT_HASH_ENV",
    "PROTECTED_RELEASE_COMMIT_SHA_ENV",
    "PROTECTED_RELEASE_PCR0_ENV",
    "PROTECTED_RELEASE_BUILD_MANIFEST_HASH_ENV",
    "PROTECTED_RELEASE_DEPENDENCY_LOCK_HASH_ENV",
    "PROTECTED_RELEASE_CONFIG_HASH_ENV",
    "PROTECTED_RELEASE_BOOT_IDENTITY_HASH_ENV",
    "PROTECTED_RELEASE_ENCLAVE_PUBKEY_ENV",
    "RoutingProductCompositionError",
    "RoutingTeeJobRpc",
    "RoutingProviderDispatchTeeRpc",
    "ReviewedRoutingReleaseInputs",
    "ReviewedRoutingAdmissionAuthority",
    "ReviewedRoutingProductComposition",
    "ReviewedRoutingProtectedAuthorities",
    "validate_reviewed_release_inputs",
    "build_reviewed_admission_authority",
    "build_attested_provider_broker_factory",
    "build_exact_model_runner_factory",
    "build_attested_protected_authorities",
    "build_reviewed_routing_product",
    "bootstrap_reviewed_routing_product",
    "install_reviewed_routing_product",
]
