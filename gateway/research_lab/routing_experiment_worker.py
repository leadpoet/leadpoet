"""Independent, opt-in worker for measured routing experiments.

This worker deliberately does not join the scoring worker fleet. It has no
provider client or credential path of its own: live calls must arrive through
the reviewed scoring dispatch operation and its attested execution authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingProviderDispatchAuthority,
    ReviewedProviderBrokerRoutingRunner,
    RoutingExperimentDeferredRecoveryError,
    RoutingExperimentRuntimeConfig,
    RoutingExperimentRuntimeError,
    RoutingExperimentService,
)
from gateway.research_lab.routing_experiment_store import (
    RoutingExperimentExecutionClaim,
    RoutingExecutionRequestLease,
    SupabaseRoutingExperimentStore,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.common_model_experiment import (
    ExactModelExperimentCoordinator,
    ExactModelUnitResult,
    FencedModelTransitionRepository,
    ReviewedModelVerificationAuthority,
    ReviewedProtectedModelActionDispatcher,
)
from gateway.research_lab.routing_provider_bindings import (
    VerifiedRoutingUnitDataset,
)
from research_lab.model_runner_protocol import (
    ExactModelRunnerRegistration,
    ExactModelRunnerRegistry,
)
from research_lab.routing_experiments import (
    ProviderReceiptStore,
    RoutingDecisionReceiptV2,
    RoutingExperimentArtifactAuthority,
    RoutingExperimentV2Adapter,
    RoutingExperimentV2Evaluation,
    RoutingExperimentV2Spec,
)
from research_lab.canonical import sha256_json
from research_lab.candidate_routing_experiments import (
    CandidateModelUnitTerminalReceipt,
    CandidateWaterfallReceipt,
    candidate_model_unit_terminal_from_exact_model,
    candidate_waterfall_receipts_from_exact_model,
    evaluate_candidate_waterfall_metrics,
)


class RoutingExperimentWorkerError(RuntimeError):
    """The isolated routing worker cannot safely run the requested job."""


def _lease_deadline_monotonic(lease_expires_at: str) -> float:
    """Project the authoritative SQL expiry onto the monotonic clock.

    The 15-second margin absorbs clock and transport uncertainty.  The local
    deadline must never be rebuilt from the requested lease duration because
    a delayed renewal response could otherwise outlive the database lease.
    """

    try:
        expiry = datetime.fromisoformat(
            str(lease_expires_at or "").replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise RoutingExperimentWorkerError(
            "routing experiment claim lease expiry is invalid"
        ) from exc
    if expiry.tzinfo is None or expiry.utcoffset() is None:
        raise RoutingExperimentWorkerError(
            "routing experiment claim lease expiry is invalid"
        )
    remaining_seconds = (
        expiry.astimezone(timezone.utc) - datetime.now(timezone.utc)
    ).total_seconds()
    return time.monotonic() + remaining_seconds - 15.0


class _RoutingClaimHeartbeat:
    """Renew one SQL execution claim until its terminal close is confirmed."""

    def __init__(self, *, store: Any, claim: RoutingExperimentExecutionClaim, lease_seconds: int) -> None:
        self._store = store
        self._claim = claim
        self._lease_seconds = int(lease_seconds)
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._deadline_lock = threading.Lock()
        self._deadline_monotonic = _lease_deadline_monotonic(
            claim.lease_expires_at
        )
        self._thread = threading.Thread(
            target=self._run,
            name="routing-experiment-claim-heartbeat",
            daemon=True,
        )
        self._index = 0

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._lease_seconds / 2))

    def ensure_held(self) -> None:
        if self.lost or self.deadline_monotonic <= time.monotonic():
            raise RoutingExperimentDeferredRecoveryError(
                "routing experiment claim heartbeat was lost"
            )

    @property
    def deadline_monotonic(self) -> float:
        with self._deadline_lock:
            return self._deadline_monotonic

    def _run(self) -> None:
        interval = max(1.0, self._lease_seconds / 3)
        while not self._stop.wait(interval):
            try:
                self._renew_once()
            except Exception:
                self._lost.set()
                return

    def _renew_once(self) -> None:
        self._index += 1
        heartbeat_key = sha256_json(
            {
                "schema_version": "leadpoet.research_lab.routing_claim_heartbeat.v3",
                "experiment_hash": self._claim.experiment_hash,
                "claim_key": self._claim.claim_key,
                "claim_generation": self._claim.claim_generation,
                "worker_ref": self._claim.worker_ref,
                "index": self._index,
            }
        )
        result = self._store.renew_claim(
            experiment_hash=self._claim.experiment_hash,
            claim=self._claim,
            heartbeat_key=heartbeat_key,
            lease_seconds=self._lease_seconds,
            heartbeat_doc={
                "schema_version": "leadpoet.research_lab.routing_claim_heartbeat.v3",
                "worker_ref": self._claim.worker_ref,
                "index": self._index,
            },
        )
        if (
            not isinstance(result, Mapping)
            or set(result)
            != {
                "renewed",
                "idempotent",
                "heartbeat_key",
                "lease_expires_at",
            }
            or result.get("renewed") is not True
            or result.get("heartbeat_key") != heartbeat_key
        ):
            raise RoutingExperimentWorkerError(
                "routing experiment claim renewal result is invalid"
            )
        renewed_deadline = _lease_deadline_monotonic(
            str(result.get("lease_expires_at") or "")
        )
        with self._deadline_lock:
            self._deadline_monotonic = renewed_deadline


REVIEWED_ROUTING_FACTORY_NAME = "exact_model_runner_v3"
LEGACY_ROUTING_FACTORY_NAME = "attested_provider_broker_v2"
ROUTING_CLAIM_AUTHORITY_ENV = "RESEARCH_LAB_ROUTING_EXPERIMENT_CLAIM_AUTHORITY"
ROUTING_ATTESTATION_AUTHORITY_ENV = (
    "RESEARCH_LAB_ROUTING_EXPERIMENT_ATTESTATION_AUTHORITY"
)


def assert_reviewed_routing_runtime_registered(
    config: RoutingExperimentRuntimeConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Require the reviewed durable claim and attestation registrations.

    The CLI must not treat a configured provider proxy as proof that durable
    authority exists.  These explicit values are deployment-owned admission
    claims; missing or unknown values stop before the store is claimed.
    """

    if not isinstance(config, RoutingExperimentRuntimeConfig):
        raise RoutingExperimentWorkerError("routing runtime configuration is invalid")
    if not config.enabled:
        raise RoutingExperimentWorkerError("routing experiment worker is disabled")
    if config.attested_authority_mode != "attested":
        raise RoutingExperimentWorkerError(
            "routing experiment durable attestation authority is unavailable"
        )
    env = os.environ if environment is None else environment
    claim_authority = str(env.get(ROUTING_CLAIM_AUTHORITY_ENV) or "").strip()
    attestation_authority = str(
        env.get(ROUTING_ATTESTATION_AUTHORITY_ENV) or ""
    ).strip()
    if claim_authority != "supabase_v3":
        raise RoutingExperimentWorkerError(
            "routing experiment durable claim authority is unavailable"
        )
    if attestation_authority != "tee_v2":
        raise RoutingExperimentWorkerError(
            "routing experiment durable attestation registration is unavailable"
        )
    if not str(env.get("SUPABASE_URL") or "").strip() or not str(
        env.get("SUPABASE_SERVICE_ROLE_KEY") or ""
    ).strip():
        raise RoutingExperimentWorkerError(
            "routing experiment durable claim store credentials are unavailable"
        )


def assert_reviewed_routing_factory_ready(
    factory: RoutingExperimentRunFactory,
) -> None:
    """Validate release-owned factory dependencies before queue admission.

    A queue consumer must not claim work and only then discover that the
    immutable model adapter, label, runner, billing, or envelope dependency
    is missing.  The concrete factory owns the checks because this boundary
    must never inspect request-controlled imports, endpoints, or credentials.
    """

    if getattr(factory, "name", None) != REVIEWED_ROUTING_FACTORY_NAME:
        raise RoutingExperimentWorkerError(
            "reviewed routing factory name is inconsistent"
        )
    validate_readiness = getattr(factory, "validate_readiness", None)
    if not callable(validate_readiness):
        raise RoutingExperimentWorkerError(
            "reviewed routing factory readiness is unavailable"
        )
    try:
        validate_readiness()
    except RoutingExperimentWorkerError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert release failure to safe error
        raise RoutingExperimentWorkerError(
            "reviewed routing factory readiness is unavailable"
        ) from exc


def build_reviewed_routing_experiment_worker(
    *,
    worker_ref: str,
    config_factory: Callable[[], RoutingExperimentRuntimeConfig] = (
        RoutingExperimentRuntimeConfig.from_env
    ),
    store_factory: Callable[[], SupabaseRoutingExperimentStore] = (
        SupabaseRoutingExperimentStore
    ),
    environment: Mapping[str, str] | None = None,
) -> "RoutingExperimentWorker":
    """Build the only product-registered worker entrypoint.

    No module path, provider callback, credential, or factory name comes from
    the CLI.  The caller must separately provide a reviewed coordinator
    factory for adapter/label/broker wiring after this durable gate passes.
    """

    config = config_factory()
    assert_reviewed_routing_runtime_registered(config, environment=environment)
    return RoutingExperimentWorker(
        service=RoutingExperimentService(
            config=config,
            store=store_factory(),
        ),
        worker_ref=worker_ref,
    )


@dataclass(frozen=True)
class RoutingExperimentRunInputs:
    """All trusted inputs required to resume one exact persisted spec."""

    gold_labels: Mapping[str, bool]
    adapters: Mapping[str, RoutingExperimentV2Adapter]
    runner: Callable[..., Any]
    artifact_authority: RoutingExperimentArtifactAuthority | None
    execution_envelope: RoutingExperimentExecutionEnvelopeV2 | None = None
    authoritative_billing_rollup: Callable[[ProviderReceiptStore], Mapping[str, Any]] | None = None


class ExactModelEvaluationAdapter(Protocol):
    """Adapt canonical PR274 receipts into PR93 evaluation contracts only."""

    def build_decision_receipts(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        unit_results: Mapping[str, Mapping[str, ExactModelUnitResult]],
    ) -> tuple[RoutingDecisionReceiptV2, ...]: ...

    def build_evaluation(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        gold_labels: Mapping[str, bool],
        unit_results: Mapping[str, Mapping[str, ExactModelUnitResult]],
        authoritative_billing_rollup: Callable[..., Mapping[str, Any]],
    ) -> RoutingExperimentV2Evaluation: ...


@dataclass(frozen=True)
class ExactModelRoutingRunInputs:
    """Exact artifact and PR93 authorities validated before the V3 claim."""

    registry: ExactModelRunnerRegistry
    registry_registrations: Mapping[str, ExactModelRunnerRegistration]
    gold_labels: Mapping[str, bool]
    unit_dataset: VerifiedRoutingUnitDataset
    reviewed_runner: ReviewedProviderBrokerRoutingRunner
    verifier: ReviewedModelVerificationAuthority
    evaluation_adapter: ExactModelEvaluationAdapter
    execution_envelope: RoutingExperimentExecutionEnvelopeV2
    authoritative_billing_rollup: Callable[..., Mapping[str, Any]]


def _validate_exact_unit_and_label_identity(
    *,
    spec: RoutingExperimentV2Spec,
    unit_dataset: VerifiedRoutingUnitDataset,
    gold_labels: Mapping[str, bool],
) -> dict[str, bool]:
    """Bind signed units and labels to the spec before its first SQL write."""

    refs = tuple(spec.input.calibration_unit_refs) + tuple(
        spec.input.holdout_unit_refs
    )
    if set(refs) != set(unit_dataset.units):
        raise RoutingExperimentWorkerError(
            "reviewed routing unit dataset differs from the spec"
        )
    if (
        not spec.input.unit_input_set_hash
        or unit_dataset.unit_set_hash != spec.input.unit_input_set_hash
    ):
        raise RoutingExperimentWorkerError(
            "reviewed routing unit dataset hash differs from the spec"
        )
    if (
        not isinstance(gold_labels, Mapping)
        or set(gold_labels) != set(refs)
        or any(
            not isinstance(key, str) or type(value) is not bool
            for key, value in gold_labels.items()
        )
    ):
        raise RoutingExperimentWorkerError(
            "reviewed routing labels differ from the spec"
        )
    normalized = dict(gold_labels)
    if sha256_json({"labels": sorted(normalized.items())}) != (
        spec.input.gold_label_set_hash
    ):
        raise RoutingExperimentWorkerError(
            "reviewed routing label hash differs from the spec"
        )
    if spec.input.stage == "candidate_acquisition":
        target = spec.input.target_verified_qualified_count
        for unit_ref in refs:
            unit_input, _unit_hash = unit_dataset.resolve(unit_ref)
            if unit_input.get("target_count") != target:
                raise RoutingExperimentWorkerError(
                    "signed Model unit target differs from the exact experiment target"
                )
    return normalized


class RoutingExperimentRunFactory(Protocol):
    """A reviewed, named factory; never a user-supplied import path."""

    name: str

    def validate_readiness(self) -> None: ...

    def build(
        self, spec: RoutingExperimentV2Spec
    ) -> RoutingExperimentRunInputs | ExactModelRoutingRunInputs: ...


@dataclass(frozen=True)
class ExactModelRoutingRunFactory:
    """The only active factory: PR274 runner plus PR93 control authorities."""

    registry: ExactModelRunnerRegistry
    gold_labels: Mapping[str, bool]
    unit_dataset: VerifiedRoutingUnitDataset
    reviewed_runner_factory: Callable[
        [RoutingExperimentV2Spec], ReviewedProviderBrokerRoutingRunner
    ]
    verifier: ReviewedModelVerificationAuthority
    evaluation_adapter: ExactModelEvaluationAdapter
    execution_envelope_factory: Callable[
        [RoutingExperimentV2Spec], RoutingExperimentExecutionEnvelopeV2
    ]
    billing_rollup_factory: Callable[
        [RoutingExperimentV2Spec], Callable[..., Mapping[str, Any]]
    ]
    site_production_model_release_identity_sha256: str | None = None
    durable_authority_identity: str | None = None
    artifact_lineages: tuple[Any, ...] = ()
    name: str = REVIEWED_ROUTING_FACTORY_NAME

    def validate_readiness(self) -> None:
        if not isinstance(self.registry, ExactModelRunnerRegistry):
            raise RoutingExperimentWorkerError(
                "exact Model runner registry is unavailable"
            )
        self.registry.preflight_all()
        if not isinstance(self.unit_dataset, VerifiedRoutingUnitDataset):
            raise RoutingExperimentWorkerError(
                "reviewed routing unit dataset is unavailable"
            )
        for method in ("verify_company", "verify_intent", "verify_contact"):
            if not callable(getattr(self.verifier, method, None)):
                raise RoutingExperimentWorkerError(
                    "reviewed Model verifier is unavailable"
                )
        if not callable(
            getattr(self.evaluation_adapter, "build_decision_receipts", None)
        ) or not callable(
            getattr(self.evaluation_adapter, "build_evaluation", None)
        ):
            raise RoutingExperimentWorkerError(
                "canonical Model evaluation adapter is unavailable"
            )
        for dependency, label in (
            (self.reviewed_runner_factory, "protected runner"),
            (self.execution_envelope_factory, "execution envelope"),
            (self.billing_rollup_factory, "billing rollup"),
        ):
            if not callable(dependency):
                raise RoutingExperimentWorkerError(
                    f"reviewed routing {label} is unavailable"
                )

    def build(self, spec: RoutingExperimentV2Spec) -> ExactModelRoutingRunInputs:
        self.validate_readiness()
        if len(spec.variants) < 2:
            raise RoutingExperimentWorkerError(
                "exact baseline and challenger artifacts are required"
            )
        registrations = {
            variant.variant_id: self.registry.resolve(
                variant.artifact.to_dict()
            )
            for variant in spec.variants
        }
        baseline = registrations.get(spec.baseline_variant_id)
        if baseline is None:
            raise RoutingExperimentWorkerError(
                "exact baseline artifact is unavailable"
            )
        if baseline.artifact_identity.get("branch") != "main":
            raise RoutingExperimentWorkerError(
                "baseline must use the Site-selected main artifact"
            )
        if self.site_production_model_release_identity_sha256 is not None:
            if baseline.protocol.release_identity.get(
                "release_identity_sha256"
            ) != self.site_production_model_release_identity_sha256:
                raise RoutingExperimentWorkerError(
                    "baseline differs from the Site production model release"
                )
        artifact_keys: set[str] = set()
        for variant in spec.variants:
            registration = registrations[variant.variant_id]
            registration.validate_variant_audit_payload(
                variant.routing_payload
            )
            if (
                variant.variant_id != spec.baseline_variant_id
                and registration.artifact_identity.get("branch")
                != "leadpoet-lab"
            ):
                raise RoutingExperimentWorkerError(
                    "challenger must use a leadpoet-lab artifact"
                )
            if (
                variant.variant_id != spec.baseline_variant_id
                and registration.key == baseline.key
            ):
                raise RoutingExperimentWorkerError(
                    "each challenger must use a distinct exact Model artifact"
                )
            if registration.key in artifact_keys:
                raise RoutingExperimentWorkerError(
                    "exact Model artifact is duplicated across variants"
                )
            artifact_keys.add(registration.key)
        labels = _validate_exact_unit_and_label_identity(
            spec=spec,
            unit_dataset=self.unit_dataset,
            gold_labels=self.gold_labels,
        )
        runner = self.reviewed_runner_factory(spec)
        if not isinstance(runner, ReviewedProviderBrokerRoutingRunner):
            raise RoutingExperimentWorkerError(
                "reviewed protected provider runner is invalid"
            )
        runner.validate_composition()
        if self.artifact_lineages:
            lineage_by_artifact = {
                sha256_json(lineage.sourcing_model_identity().to_dict()): lineage
                for lineage in self.artifact_lineages
            }
            variant_lineages = {}
            for variant in spec.variants:
                lineage = lineage_by_artifact.get(
                    sha256_json(variant.artifact.to_dict())
                )
                if lineage is None:
                    raise RoutingExperimentWorkerError(
                        "reviewed routing variant artifact is not signed in the release bundle"
                    )
                variant_lineages[variant.variant_id] = lineage
            validate_lineages = getattr(runner, "validate_artifact_lineages", None)
            if not callable(validate_lineages):
                raise RoutingExperimentWorkerError(
                    "reviewed routing runner lacks variant lineage binding"
                )
            try:
                validate_lineages(variant_lineages)
            except Exception as exc:  # noqa: BLE001 - protected lineage boundary
                raise RoutingExperimentWorkerError(
                    "reviewed routing runner variant lineage binding failed"
                ) from exc
        if self.durable_authority_identity is not None and getattr(
            runner, "durable_authority_identity", None
        ) != self.durable_authority_identity:
            raise RoutingExperimentWorkerError(
                "reviewed routing durable authority identity differs"
            )
        envelope = self.execution_envelope_factory(spec)
        if not isinstance(envelope, RoutingExperimentExecutionEnvelopeV2):
            raise RoutingExperimentWorkerError(
                "reviewed routing execution envelope is invalid"
            )
        rollup = self.billing_rollup_factory(spec)
        if not callable(rollup):
            raise RoutingExperimentWorkerError(
                "reviewed routing billing authority is invalid"
            )
        return ExactModelRoutingRunInputs(
            registry=self.registry,
            registry_registrations=registrations,
            gold_labels=labels,
            unit_dataset=self.unit_dataset,
            reviewed_runner=runner,
            verifier=self.verifier,
            evaluation_adapter=self.evaluation_adapter,
            execution_envelope=envelope,
            authoritative_billing_rollup=rollup,
        )


@dataclass(frozen=True)
class AttestedProviderBrokerRoutingRunFactory:
    """Concrete bridge from an exact model adapter to protected dispatch.

    Deployment wiring supplies reviewed constructors for the immutable model
    adapter map, gold-label authority, artifact verifier, and the fully typed
    ``ReviewedProviderBrokerRoutingRunner``. This class accepts no module name,
    provider URL, credential, broker, or arbitrary tool callback from the CLI.
    """

    adapter_factory: Callable[[RoutingExperimentV2Spec], Mapping[str, RoutingExperimentV2Adapter]]
    gold_label_loader: Callable[[RoutingExperimentV2Spec], Mapping[str, bool]]
    reviewed_runner_factory: Callable[
        [RoutingExperimentV2Spec], ReviewedProviderBrokerRoutingRunner
    ]
    artifact_authority: RoutingExperimentArtifactAuthority
    billing_rollup_factory: Callable[[RoutingExperimentV2Spec], Callable[[ProviderReceiptStore], Mapping[str, Any]]]
    execution_envelope_factory: Callable[
        [RoutingExperimentV2Spec], RoutingExperimentExecutionEnvelopeV2
    ]
    name: str = "attested_provider_broker_v2"

    def validate_readiness(self) -> None:
        """Check static release-owned dependencies without running a spec.

        Spec-specific model and policy checks remain in ``build``.  This
        method only validates objects that must exist before a queue claim and
        never invokes a provider, adapter, broker, or TEE operation.
        """

        dependencies = (
            (self.adapter_factory, "adapter factory"),
            (self.gold_label_loader, "gold-label loader"),
            (self.reviewed_runner_factory, "reviewed routing runner factory"),
            (self.billing_rollup_factory, "billing authority"),
            (self.execution_envelope_factory, "execution envelope factory"),
        )
        for dependency, label in dependencies:
            if not callable(dependency):
                raise RoutingExperimentWorkerError(
                    f"reviewed routing {label} is unavailable"
                )
        if not callable(getattr(self.artifact_authority, "verify", None)):
            raise RoutingExperimentWorkerError(
                "reviewed routing artifact authority is unavailable"
            )
        runner_readiness = getattr(
            self.reviewed_runner_factory, "validate_readiness", None
        )
        if not callable(runner_readiness):
            raise RoutingExperimentWorkerError(
                "reviewed routing runner readiness is unavailable"
            )
        try:
            runner_readiness()
        except Exception as exc:  # noqa: BLE001 - release-owned hook
            raise RoutingExperimentWorkerError(
                "reviewed routing runner readiness is unavailable"
            ) from exc

    def build(self, spec: RoutingExperimentV2Spec) -> RoutingExperimentRunInputs:
        adapters = self.adapter_factory(spec)
        if not isinstance(adapters, Mapping) or set(adapters) != {
            variant.variant_id for variant in spec.variants
        }:
            raise RoutingExperimentWorkerError("reviewed routing adapter map is incomplete")
        if not callable(self.reviewed_runner_factory):
            raise RoutingExperimentWorkerError(
                "reviewed routing runner factory is unavailable"
            )
        runner = self.reviewed_runner_factory(spec)
        if not isinstance(runner, ReviewedProviderBrokerRoutingRunner):
            raise RoutingExperimentWorkerError("reviewed routing runner is invalid")
        try:
            runner.validate_composition()
        except Exception as exc:
            raise RoutingExperimentWorkerError(
                "reviewed routing runner dispatch composition is invalid"
            ) from exc
        if not isinstance(
            getattr(runner, "dispatch_authority", None),
            AttestedScoringV2RoutingProviderDispatchAuthority,
        ):
            raise RoutingExperimentWorkerError(
                "reviewed routing runner dispatch authority is invalid"
            )
        labels = self.gold_label_loader(spec)
        if not isinstance(labels, Mapping):
            raise RoutingExperimentWorkerError("reviewed routing labels are invalid")
        rollup = self.billing_rollup_factory(spec)
        if not callable(rollup):
            raise RoutingExperimentWorkerError("reviewed routing billing authority is invalid")
        execution_envelope = self.execution_envelope_factory(spec)
        if not isinstance(
            execution_envelope, RoutingExperimentExecutionEnvelopeV2
        ):
            raise RoutingExperimentWorkerError(
                "reviewed routing execution envelope is invalid"
            )
        return RoutingExperimentRunInputs(
            gold_labels=labels,
            adapters=adapters,
            runner=runner,
            artifact_authority=self.artifact_authority,
            execution_envelope=execution_envelope,
            authoritative_billing_rollup=rollup,
        )


class RoutingExperimentWorker:
    """Claim, run, and audit one immutable experiment without auto-starting."""

    def __init__(
        self,
        *,
        service: RoutingExperimentService,
        worker_ref: str,
    ) -> None:
        if not str(worker_ref or "").strip():
            raise RoutingExperimentWorkerError("routing worker reference is required")
        self.service = service
        self.worker_ref = str(worker_ref)

    def run(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        inputs: RoutingExperimentRunInputs | ExactModelRoutingRunInputs,
        lease: RoutingExecutionRequestLease | None = None,
    ) -> RoutingExperimentV2Evaluation:
        """Run one claimed spec and leave an append-only terminal event."""

        if isinstance(inputs, ExactModelRoutingRunInputs):
            return self._run_exact_model(spec=spec, inputs=inputs, lease=lease)

        for method_name in ("renew_claim", "close_claim"):
            if not callable(getattr(self.service.store, method_name, None)):
                raise RoutingExperimentWorkerError(
                    f"routing claim store is missing {method_name}"
                )
        self.service.submit(spec, execution_envelope=inputs.execution_envelope)
        claim = self.service.claim_execution(
            spec=spec,
            worker_ref=self.worker_ref,
            lease=lease,
        )
        heartbeat = _RoutingClaimHeartbeat(
            store=self.service.store,
            claim=claim,
            lease_seconds=self.service.config.worker_lease_seconds,
        )
        heartbeat.start()
        try:
            heartbeat.ensure_held()
            try:
                self._append_execution_event(
                    spec=spec,
                    claim=claim,
                    event_type="run_started",
                    event_doc={"worker_ref": self.worker_ref},
                )
            except Exception as event_error:
                raise RoutingExperimentDeferredRecoveryError(
                    "routing experiment start event could not be confirmed"
                ) from event_error
            heartbeat.ensure_held()
            try:
                evaluation = self.service.evaluate(
                    spec=spec,
                    gold_labels=inputs.gold_labels,
                    adapters=inputs.adapters,
                    runner=inputs.runner,
                    artifact_authority=inputs.artifact_authority,
                    execution_envelope=inputs.execution_envelope,
                    authoritative_billing_rollup=inputs.authoritative_billing_rollup,
                    worker_ref=self.worker_ref,
                    claim=claim,
                    claim_deadline_supplier=lambda: heartbeat.deadline_monotonic,
                )
            except Exception as exc:
                # This is an audit event only. It never includes raw exception
                # text, provider payloads, or credentials.
                if isinstance(exc, RoutingExperimentDeferredRecoveryError):
                    try:
                        self._append_execution_event(
                            spec=spec,
                            claim=claim,
                            event_type="run_failed",
                            event_doc={
                                "error_class": type(exc).__name__,
                                "worker_ref": self.worker_ref,
                            },
                        )
                    except Exception:
                        pass
                    raise
                try:
                    self._append_execution_event(
                        spec=spec,
                        claim=claim,
                        event_type="run_failed",
                        event_doc={
                            "error_class": type(exc).__name__,
                            "worker_ref": self.worker_ref,
                        },
                    )
                except Exception as event_error:
                    raise RoutingExperimentDeferredRecoveryError(
                        "routing experiment failure event could not be confirmed"
                    ) from event_error
                self._close_claim(spec=spec, claim=claim, reason="failed")
                raise
            heartbeat.ensure_held()
            self._close_claim(spec=spec, claim=claim, reason="completed")
            return evaluation
        finally:
            heartbeat.stop()

    def _run_exact_model(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        inputs: ExactModelRoutingRunInputs,
        lease: RoutingExecutionRequestLease | None,
    ) -> RoutingExperimentV2Evaluation:
        """Run baseline and challenger through only PR274's action protocol."""

        # Factory construction and exact artifact preflight happen before
        # this method. Submit is therefore the first SQL mutation.
        self.service.submit(spec, execution_envelope=inputs.execution_envelope)
        claim = self.service.claim_execution(
            spec=spec,
            worker_ref=self.worker_ref,
            lease=lease,
        )
        heartbeat = _RoutingClaimHeartbeat(
            store=self.service.store,
            claim=claim,
            lease_seconds=self.service.config.worker_lease_seconds,
        )
        heartbeat.start()
        try:
            heartbeat.ensure_held()
            self._append_execution_event(
                spec=spec,
                claim=claim,
                event_type="run_started",
                event_doc={
                    "worker_ref": self.worker_ref,
                    "runner_contract": "pr274_model_runner",
                },
            )
            dispatcher = ReviewedProtectedModelActionDispatcher(
                spec=spec,
                registrations=inputs.registry_registrations,
                runner=inputs.reviewed_runner,
                claim=claim,
                deadline_supplier=lambda: heartbeat.deadline_monotonic,
                verifier=inputs.verifier,
            )
            transitions = FencedModelTransitionRepository(
                store=self.service.store,
                claim=claim,
            )
            ordered_units = tuple(spec.input.calibration_unit_refs) + tuple(
                spec.input.holdout_unit_refs
            )
            unit_results: dict[str, dict[str, ExactModelUnitResult]] = {}
            for variant in spec.variants:
                registration = inputs.registry.resolve(
                    variant.artifact.to_dict()
                )
                per_unit: dict[str, ExactModelUnitResult] = {}
                for unit_ref in ordered_units:
                    heartbeat.ensure_held()
                    unit_input, _unit_hash = inputs.unit_dataset.resolve(unit_ref)
                    model_input = unit_input.get("model_input")
                    execution_mode = unit_input.get("execution_mode")
                    target_count = unit_input.get("target_count")
                    evaluated_on = unit_input.get("evaluated_on")
                    if (
                        not isinstance(model_input, Mapping)
                        or execution_mode
                        not in {
                            "full_company",
                            "full_contact_optional",
                            "full_contact_required",
                            "intent_refresh",
                        }
                        or type(target_count) is not int
                        or not 1 <= target_count <= 50
                        or not re.fullmatch(
                            r"\d{4}-\d{2}-\d{2}",
                            str(evaluated_on or ""),
                        )
                    ):
                        raise RoutingExperimentWorkerError(
                            "signed Model unit input is invalid"
                        )
                    if (
                        spec.input.stage == "candidate_acquisition"
                        and target_count != spec.input.target_verified_qualified_count
                    ):
                        raise RoutingExperimentWorkerError(
                            "signed Model unit target differs from the exact experiment target"
                        )
                    coordinator = ExactModelExperimentCoordinator(
                        experiment_hash=spec.experiment_hash(),
                        registration=registration,
                        dispatcher=dispatcher,
                        transitions=transitions,
                    )
                    per_unit[unit_ref] = coordinator.run_unit(
                        variant_id=variant.variant_id,
                        unit_ref=unit_ref,
                        model_input=model_input,
                        execution_mode=str(execution_mode),
                        target_count=target_count,
                        evaluated_on=str(evaluated_on),
                    )
                unit_results[variant.variant_id] = per_unit
            decisions = inputs.evaluation_adapter.build_decision_receipts(
                spec=spec,
                unit_results=unit_results,
            )
            if not isinstance(decisions, tuple):
                raise RoutingExperimentWorkerError(
                    "canonical Model decision receipts are invalid"
                )
            decisions_by_unit: dict[tuple[str, str], tuple[RoutingDecisionReceiptV2, ...]] = {}
            for receipt in decisions:
                if not isinstance(receipt, RoutingDecisionReceiptV2):
                    raise RoutingExperimentWorkerError(
                        "canonical Model decision receipt is invalid"
                    )
                self.service.store.append_decision(
                    experiment_hash=spec.experiment_hash(),
                    receipt=receipt,
                    claim=claim,
                )
                key = (receipt.variant_id, receipt.unit_ref)
                decisions_by_unit[key] = (*decisions_by_unit.get(key, ()), receipt)

            candidate_receipts: list[CandidateWaterfallReceipt] = []
            candidate_terminals: dict[tuple[str, str], CandidateModelUnitTerminalReceipt] = {}
            candidate_provider_receipts = []
            for variant in spec.variants:
                registration = inputs.registry.resolve(variant.artifact.to_dict())
                release_identity = registration.protocol.release_identity
                release_identity_sha256 = str(
                    release_identity.get("release_identity_sha256") or ""
                ).removeprefix("sha256:")
                binding_contracts_sha256 = str(
                    release_identity.get("tool_binding_manifest_sha256") or ""
                ).removeprefix("sha256:")
                candidate_waterfall_contract_sha256 = str(
                    release_identity.get("candidate_waterfall_contract_sha256") or ""
                ).removeprefix("sha256:")
                for unit_ref in ordered_units:
                    unit_result = unit_results[variant.variant_id][unit_ref]
                    unit_decisions = tuple(
                        item
                        for item in decisions_by_unit.get(
                            (variant.variant_id, unit_ref), ()
                        )
                        if item.stage == "candidate_acquisition"
                    )
                    if len(unit_decisions) != 1:
                        raise RoutingExperimentWorkerError(
                            "exact Model candidate decision coverage is invalid"
                        )
                    authoritative_candidate_receipts = tuple(
                        item
                        for item in unit_result.provider_receipts
                        if item.tool_id.startswith("candidate.")
                    )
                    candidate_provider_receipts.extend(authoritative_candidate_receipts)
                    terminal = candidate_model_unit_terminal_from_exact_model(
                        spec=spec,
                        variant_id=variant.variant_id,
                        decision_receipt=unit_decisions[0],
                        terminal_result=unit_result.terminal_result,
                        expected_release_identity_sha256=release_identity_sha256,
                        expected_binding_contracts_sha256=binding_contracts_sha256,
                        expected_candidate_waterfall_contract_sha256=(
                            candidate_waterfall_contract_sha256
                        ),
                        authoritative_provider_receipts=authoritative_candidate_receipts,
                    )
                    self.service.store.append_candidate_model_unit_terminal(
                        experiment_hash=spec.experiment_hash(),
                        receipt=terminal,
                        claim=claim,
                    )
                    candidate_terminals[(variant.variant_id, unit_ref)] = terminal
            for variant in spec.variants:
                registration = inputs.registry.resolve(variant.artifact.to_dict())
                release_identity = registration.protocol.release_identity
                release_identity_sha256 = str(
                    release_identity.get("release_identity_sha256") or ""
                ).removeprefix("sha256:")
                binding_contracts_sha256 = str(
                    release_identity.get("tool_binding_manifest_sha256") or ""
                ).removeprefix("sha256:")
                candidate_waterfall_contract_sha256 = str(
                    release_identity.get("candidate_waterfall_contract_sha256") or ""
                ).removeprefix("sha256:")
                for unit_ref in ordered_units:
                    unit_result = unit_results[variant.variant_id][unit_ref]
                    unit_decisions = tuple(
                        item
                        for item in decisions_by_unit.get(
                            (variant.variant_id, unit_ref), ()
                        )
                        if item.stage == "candidate_acquisition"
                    )
                    if len(unit_decisions) != 1:
                        raise RoutingExperimentWorkerError(
                            "exact Model candidate decision coverage is invalid"
                        )
                    authoritative_candidate_receipts = tuple(
                        item
                        for item in unit_result.provider_receipts
                        if item.tool_id.startswith("candidate.")
                    )
                    projected = candidate_waterfall_receipts_from_exact_model(
                        spec=spec,
                        variant_id=variant.variant_id,
                        decision_receipt=unit_decisions[0],
                        terminal_result=unit_result.terminal_result,
                        expected_release_identity_sha256=release_identity_sha256,
                        expected_binding_contracts_sha256=binding_contracts_sha256,
                        expected_candidate_waterfall_contract_sha256=(
                            candidate_waterfall_contract_sha256
                        ),
                        authoritative_provider_receipts=authoritative_candidate_receipts,
                        model_terminal_receipt=candidate_terminals[(variant.variant_id, unit_ref)],
                    )
                    candidate_receipts.extend(projected)
            if not candidate_receipts:
                raise RoutingExperimentWorkerError(
                    "exact Model candidate waterfall receipts are missing"
                )
            target_counts = {
                result.target_verified_qualified_count
                for per_variant in unit_results.values()
                for result in per_variant.values()
            }
            if len(target_counts) != 1:
                raise RoutingExperimentWorkerError(
                    "exact Model candidate target is inconsistent across units"
                )
            target_count = next(iter(target_counts))
            for receipt in candidate_receipts:
                self.service.store.append_candidate_waterfall_receipt(
                    experiment_hash=spec.experiment_hash(),
                    receipt=receipt,
                    claim=claim,
                )
            evaluation = inputs.evaluation_adapter.build_evaluation(
                spec=spec,
                gold_labels=inputs.gold_labels,
                unit_results=unit_results,
                authoritative_billing_rollup=(
                    inputs.authoritative_billing_rollup
                ),
            )
            if not isinstance(evaluation, RoutingExperimentV2Evaluation):
                raise RoutingExperimentWorkerError(
                    "canonical Model evaluation is invalid"
                )
            candidate_metrics = evaluate_candidate_waterfall_metrics(
                spec=spec,
                evaluation=evaluation,
                receipts=tuple(candidate_receipts),
                target_verified_qualified_count=target_count,
                authoritative_provider_receipts=tuple(candidate_provider_receipts),
            )
            self.service.store.append_evaluation(
                spec=spec,
                evaluation=evaluation,
                claim=claim,
            )
            for metric in candidate_metrics:
                self.service.store.append_candidate_waterfall_metric(
                    experiment_hash=spec.experiment_hash(),
                    metric=metric,
                    claim=claim,
                )
            self._append_execution_event(
                spec=spec,
                claim=claim,
                event_type="run_completed",
                event_doc={
                    "evaluation_receipt_id": evaluation.receipt_id,
                    "selected_variant_id": (
                        evaluation.selected_variant_id or "unselected"
                    ),
                    "runner_contract": "pr274_model_runner",
                },
            )
            heartbeat.ensure_held()
            self._close_claim(spec=spec, claim=claim, reason="completed")
            return evaluation
        except RoutingExperimentDeferredRecoveryError:
            raise
        except Exception as exc:
            try:
                self._append_execution_event(
                    spec=spec,
                    claim=claim,
                    event_type="run_failed",
                    event_doc={
                        "error_class": type(exc).__name__,
                        "worker_ref": self.worker_ref,
                        "runner_contract": "pr274_model_runner",
                    },
                )
            except Exception as event_error:
                raise RoutingExperimentDeferredRecoveryError(
                    "exact Model failure event could not be confirmed"
                ) from event_error
            self._close_claim(spec=spec, claim=claim, reason="failed")
            raise
        finally:
            heartbeat.stop()

    def _close_claim(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        claim: RoutingExperimentExecutionClaim,
        reason: str,
    ) -> None:
        close_key = sha256_json(
            {
                "schema_version": "leadpoet.research_lab.routing_claim_close.v3",
                "experiment_hash": spec.experiment_hash(),
                "claim_key": claim.claim_key,
                "claim_generation": claim.claim_generation,
                "close_reason": reason,
            }
        )
        try:
            result = self.service.store.close_claim(
                experiment_hash=spec.experiment_hash(),
                claim=claim,
                close_key=close_key,
                close_reason=reason,
                close_doc={
                    "schema_version": "leadpoet.research_lab.routing_claim_close.v3",
                    "worker_ref": self.worker_ref,
                    "close_reason": reason,
                },
            )
            if (
                not isinstance(result, Mapping)
                or result.get("closed") is not True
                or result.get("close_key") != close_key
            ):
                raise RoutingExperimentWorkerError(
                    "routing experiment claim close result is invalid"
                )
        except RoutingExperimentDeferredRecoveryError:
            raise
        except Exception as exc:
            raise RoutingExperimentDeferredRecoveryError(
                "routing experiment claim close could not be confirmed"
            ) from exc

    def resume(
        self,
        *,
        experiment_hash: str,
        input_factory: Callable[[RoutingExperimentV2Spec], RoutingExperimentRunInputs],
        lease: RoutingExecutionRequestLease | None = None,
    ) -> RoutingExperimentV2Evaluation:
        """Reload the immutable stored spec; callers must supply trusted inputs."""

        spec = self.service.store.load_spec(experiment_hash)
        if spec is None:
            raise RoutingExperimentWorkerError("routing experiment was not found")
        inputs = input_factory(spec)
        if not isinstance(
            inputs, (RoutingExperimentRunInputs, ExactModelRoutingRunInputs)
        ):
            raise RoutingExperimentWorkerError("routing worker input factory is invalid")
        return self.run(spec=spec, inputs=inputs, lease=lease)

    def _append_execution_event(
        self,
        *,
        spec: RoutingExperimentV2Spec,
        claim: RoutingExperimentExecutionClaim,
        event_type: str,
        event_doc: Mapping[str, Any],
    ) -> None:
        document = {
            "schema_version": "leadpoet.research_lab.routing_worker_event.v2",
            "worker_ref": self.worker_ref,
            **dict(event_doc),
        }
        self.service.store.append_event(
            experiment_hash=spec.experiment_hash(),
            event_type=event_type,
            event_doc=document,
            claim=claim,
        )


class RoutingExperimentCoordinator:
    """Named factory registry for a safe worker/CLI run or resume command."""

    def __init__(
        self,
        *,
        worker: RoutingExperimentWorker,
        factories: Mapping[str, RoutingExperimentRunFactory],
    ) -> None:
        self.worker = worker
        self._factories = dict(factories)
        for name, factory in self._factories.items():
            if name != getattr(factory, "name", None):
                raise RoutingExperimentWorkerError("routing factory name is inconsistent")

    def resume(
        self,
        *,
        experiment_hash: str,
        factory_name: str,
        lease: RoutingExecutionRequestLease | None = None,
    ) -> RoutingExperimentV2Evaluation:
        factory = self._factories.get(factory_name)
        if factory is None:
            raise RoutingExperimentWorkerError("reviewed routing runtime factory is unavailable")
        return self.worker.resume(
            experiment_hash=experiment_hash,
            input_factory=factory.build,
            lease=lease,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check-config",
        action="store_true",
        help="Validate the disabled-by-default routing worker configuration.",
    )
    action.add_argument(
        "--status",
        metavar="EXPERIMENT_HASH",
        help="Read an immutable experiment spec. This never invokes a provider.",
    )
    action.add_argument(
        "--run",
        metavar="EXPERIMENT_HASH",
        help="Claim and resume through a reviewed named factory; disabled by default.",
    )
    parser.add_argument(
        "--factory",
        default=REVIEWED_ROUTING_FACTORY_NAME,
        help=(
            "Reviewed routing runtime factory name "
            "(default: exact_model_runner_v3)."
        ),
    )
    return parser


def _status_document(store: SupabaseRoutingExperimentStore, experiment_hash: str) -> dict[str, Any]:
    spec = store.load_spec(experiment_hash)
    if spec is None:
        raise RoutingExperimentWorkerError("routing experiment was not found")
    return {
        "experiment_hash": spec.experiment_hash(),
        "experiment_id": spec.experiment_id,
        "receipt_execution_mode": spec.receipt_execution_mode,
        "allow_live_credit_spend": spec.allow_live_credit_spend,
        "variant_ids": [variant.variant_id for variant in spec.variants],
    }


def main(
    argv: list[str] | None = None,
    *,
    store_factory: Callable[[], SupabaseRoutingExperimentStore] = SupabaseRoutingExperimentStore,
    config_factory: Callable[[], RoutingExperimentRuntimeConfig] = RoutingExperimentRuntimeConfig.from_env,
    coordinator_factory: Callable[[RoutingExperimentWorker], RoutingExperimentCoordinator] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = config_factory()
        if args.check_config:
            print(
                json.dumps(
                    {
                        "enabled": config.enabled,
                        "live_execution_enabled": config.live_execution_enabled,
                        "attested_authority_mode": config.attested_authority_mode,
                        "provider_execution_available": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.status:
            store = store_factory()
            print(json.dumps(_status_document(store, args.status), sort_keys=True))
            return 0
        # A run must pass the explicit durable authority registration before
        # any claim RPC. Status remains a store-only read above.
        # Keep dependency-injected unit-test configs on the existing static
        # registry path; production uses the concrete runtime config and must
        # pass the durable authority gate above.
        if isinstance(config, RoutingExperimentRuntimeConfig):
            assert_reviewed_routing_runtime_registered(config)
        store = store_factory()
        service = RoutingExperimentService(config=config, store=store)
        worker = RoutingExperimentWorker(
            service=service,
            worker_ref="routing-experiment-cli",
        )
        coordinator = (
            coordinator_factory(worker)
            if coordinator_factory is not None
            else RoutingExperimentCoordinator(worker=worker, factories={})
        )
        evaluation = coordinator.resume(experiment_hash=args.run, factory_name=args.factory)
        print(
            json.dumps(
                {
                    "receipt_id": evaluation.receipt_id,
                    "selected_variant_id": evaluation.selected_variant_id,
                },
                sort_keys=True,
            )
        )
        return 0
    except (RoutingExperimentRuntimeError, RoutingExperimentWorkerError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "RoutingExperimentWorkerError",
    "REVIEWED_ROUTING_FACTORY_NAME",
    "ROUTING_CLAIM_AUTHORITY_ENV",
    "ROUTING_ATTESTATION_AUTHORITY_ENV",
    "assert_reviewed_routing_runtime_registered",
    "assert_reviewed_routing_factory_ready",
    "build_reviewed_routing_experiment_worker",
    "RoutingExperimentRunInputs",
    "RoutingExperimentRunFactory",
    "AttestedProviderBrokerRoutingRunFactory",
    "RoutingExperimentWorker",
    "RoutingExperimentCoordinator",
    "main",
]
