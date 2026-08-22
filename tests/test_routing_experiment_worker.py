from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import time
from types import SimpleNamespace

import pytest

import gateway.research_lab.routing_experiment_worker as worker_module

from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingProviderCallAuthority,
    AttestedScoringV2RoutingProviderDispatchAuthority,
    ReviewedProviderBrokerRoutingRunner,
    RoutingExperimentDeferredRecoveryError,
    RoutingProviderDispatchExecutor,
    RoutingExperimentRuntimeConfig,
    RoutingExperimentRuntimeError,
    _ROUTING_DISPATCH_EXECUTOR_TOKEN,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.routing_experiment_store import (
    RoutingExecutionRequestLease,
    RoutingExperimentExecutionClaim,
    RoutingExperimentStoreError,
    routing_claim_fence_hash_v3,
)
from gateway.research_lab.routing_experiment_worker import (
    AttestedProviderBrokerRoutingRunFactory,
    RoutingExperimentCoordinator,
    RoutingExperimentRunInputs,
    RoutingExperimentWorker,
    RoutingExperimentWorkerError,
    _RoutingClaimHeartbeat,
    main,
)
from research_lab.routing_experiments import (
    RoutingDecisionReceiptV2,
    RoutingExperimentV2Evaluation,
    RoutingExperimentV2VariantEvaluation,
    finalize_routing_decision_receipt_v2,
)
from tests.routing_experiment_authority_fixture import authority_fixture


def _hash(char: str) -> str:
    return "sha256:" + char * 64


@dataclass(frozen=True)
class _SyntheticMetrics:
    """Minimal synthetic metric document for worker-boundary tests."""

    def to_dict(self):
        return {"synthetic": True}


def _synthetic_intent_decision(
    *,
    experiment_id: str = "intent-experiment",
    variant_id: str = "baseline",
    artifact_key: str | None = None,
    stage: str = "intent_evidence",
    unit_ref: str = "unit-1",
    provider_receipt_refs: tuple[str, ...] = (),
) -> RoutingDecisionReceiptV2:
    tool_ids = tuple(
        f"intent.synthetic.{index}"
        for index, _receipt_ref in enumerate(provider_receipt_refs)
    )
    return finalize_routing_decision_receipt_v2(
        RoutingDecisionReceiptV2(
            receipt_id="routing_decision:pending",
            experiment_id=experiment_id,
            variant_id=variant_id,
            artifact_key=artifact_key or _hash("c"),
            stage=stage,
            unit_ref=unit_ref,
            plan_hash=_hash("d"),
            route_hash=_hash("e"),
            considered_tool_ids=tool_ids,
            attempted_tool_ids=tool_ids,
            skipped_tool_reasons=(),
            outcome_reasons=tuple(
                (tool_id, "verified") for tool_id in tool_ids
            ),
            provider_receipt_refs=provider_receipt_refs,
            total_credit_microunits=0,
            latency_ms=0,
            execution_mode="fixture",
        )
    )


def _synthetic_intent_evaluation(
    *,
    decision: RoutingDecisionReceiptV2,
    provider_receipt_refs: tuple[str, ...] = (),
    artifact_key: str | None = None,
    passed: bool = True,
) -> RoutingExperimentV2Evaluation:
    variant = RoutingExperimentV2VariantEvaluation(
        variant_id="baseline",
        artifact_key=artifact_key or _hash("c"),
        stage="intent_evidence",
        calibration=_SyntheticMetrics(),
        holdout=_SyntheticMetrics(),
        passed_precision_gate=passed,
        passed_recall_gate=passed,
        passed_cost_gate=passed,
        passed_efficiency_gate=passed,
        passed=passed,
        decision_receipt_refs=(decision.receipt_id,),
        provider_receipt_refs=provider_receipt_refs,
    )
    draft = RoutingExperimentV2Evaluation(
        receipt_id="routing_evaluation_v2:pending",
        experiment_id="intent-experiment",
        experiment_hash=_hash("a"),
        variants=(variant,),
        baseline_variant_id="baseline",
        selected_variant_id="baseline" if passed else "",
        decision_receipt_refs=(decision.receipt_id,),
        provider_receipt_refs=provider_receipt_refs,
        provider_cache_hits=0,
        provider_cache_misses=0,
    )
    return replace(
        draft,
        receipt_id=(
            "routing_evaluation_v2:"
            + worker_module.sha256_json(draft.to_dict()).split(":", 1)[1][:16]
        ),
    )


def _synthetic_intent_spec():
    variant = SimpleNamespace(
        variant_id="baseline",
        stage="intent_evidence",
        artifact=SimpleNamespace(to_dict=lambda: {"artifact": "exact"}),
    )
    return SimpleNamespace(
        experiment_id="intent-experiment",
        baseline_variant_id="baseline",
        allow_live_credit_spend=False,
        input=SimpleNamespace(
            stage="intent_evidence",
            calibration_unit_refs=("unit-1",),
            holdout_unit_refs=(),
        ),
        variants=(variant,),
        experiment_hash=lambda: _hash("a"),
    )


def _refinalize_decision(
    receipt: RoutingDecisionReceiptV2,
    **changes,
) -> RoutingDecisionReceiptV2:
    return finalize_routing_decision_receipt_v2(
        replace(receipt, receipt_id="routing_decision:pending", **changes)
    )


@dataclass(frozen=True)
class _Spec:
    value: str = _hash("a")
    experiment_id: str = "routing-worker"
    allow_live_credit_spend: bool = False
    receipt_execution_mode: str = "fixture"
    variants: tuple[object, ...] = ()

    def experiment_hash(self):
        return self.value


class _Store:
    def __init__(self, spec=None):
        self.spec = spec
        self.events = []
        self.closed = []

    def append_event(self, **kwargs):
        self.events.append(kwargs)
        return {"event_hash": "ok"}

    def load_spec(self, experiment_hash):
        return self.spec if self.spec and self.spec.experiment_hash() == experiment_hash else None

    def renew_claim(self, **kwargs):
        return {
            "renewed": True,
            "idempotent": False,
            "heartbeat_key": kwargs["heartbeat_key"],
            "lease_expires_at": "2099-01-01T00:00:00+00:00",
        }

    def close_claim(self, **kwargs):
        self.closed.append(kwargs)
        return {"closed": True, "close_key": kwargs["close_key"]}


class _Service:
    def __init__(self, *, fail=False, deferred=False):
        self.store = _Store(_Spec())
        self.config = type("Config", (), {"worker_lease_seconds": 30})()
        self.fail = fail
        self.deferred = deferred
        self.submitted = []
        self.lease = RoutingExecutionRequestLease(
            request_hash=_hash("d"),
            experiment_hash=_hash("a"),
            lease_hash=_hash("e"),
            worker_ref="worker-1",
            lease_generation=1,
            lease_expires_at="2099-01-01T00:00:00+00:00",
        )
        self.claim = RoutingExperimentExecutionClaim(
            _hash("a"),
            _hash("b"),
            1,
            routing_claim_fence_hash_v3(
                experiment_hash=_hash("a"),
                claim_key=_hash("b"),
                claim_generation=1,
            ),
            self.lease.request_hash,
            self.lease.lease_hash,
            self.lease.lease_generation,
            self.lease.worker_ref,
            self.lease.lease_expires_at,
        )
        self.evaluate_claim = None

    def submit(self, spec, *, execution_envelope=None):
        del execution_envelope
        self.submitted.append(spec)
        return {"submitted": True}

    def claim_execution(self, *, spec, worker_ref, lease=None):
        assert worker_ref == "worker-1"
        assert spec.experiment_hash() == self.claim.experiment_hash
        assert lease == self.lease
        return self.claim

    def evaluate(self, **kwargs):
        self.evaluate_claim = kwargs["claim"]
        if self.deferred:
            raise RoutingExperimentDeferredRecoveryError("claim cleanup is deferred")
        if self.fail:
            raise ValueError("provider payload must not leak")
        return type("Evaluation", (), {"receipt_id": "routing_evaluation_v2:" + "d" * 16, "selected_variant_id": "candidate"})()


def _inputs():
    return RoutingExperimentRunInputs(
        gold_labels={}, adapters={}, runner=lambda *_args: None, artifact_authority=None
    )


def test_worker_claims_before_run_and_fences_all_terminal_events():
    service = _Service()
    worker = RoutingExperimentWorker(service=service, worker_ref="worker-1")
    result = worker.run(spec=_Spec(), inputs=_inputs(), lease=service.lease)
    assert result.selected_variant_id == "candidate"
    assert service.evaluate_claim == service.claim
    assert [item["event_type"] for item in service.store.events] == ["run_started"]
    assert service.store.events[0]["claim"] == service.claim
    assert service.store.closed[0]["close_reason"] == "completed"


def test_worker_records_only_a_redacted_failure_class_when_run_fails():
    service = _Service(fail=True)
    worker = RoutingExperimentWorker(service=service, worker_ref="worker-1")
    with pytest.raises(ValueError, match="provider payload"):
        worker.run(spec=_Spec(), inputs=_inputs(), lease=service.lease)
    failure = service.store.events[-1]
    assert failure["event_type"] == "run_failed"
    assert failure["event_doc"]["error_class"] == "ValueError"
    assert "provider payload" not in str(failure["event_doc"])
    assert service.store.closed[0]["close_reason"] == "failed"


def test_worker_does_not_close_claim_when_recovery_is_deferred():
    service = _Service(deferred=True)
    worker = RoutingExperimentWorker(service=service, worker_ref="worker-1")
    with pytest.raises(RoutingExperimentDeferredRecoveryError):
        worker.run(spec=_Spec(), inputs=_inputs(), lease=service.lease)
    assert service.store.closed == []


def test_exact_model_worker_passes_variant_registrations_to_dispatcher(monkeypatch):
    service = _Service()
    worker = RoutingExperimentWorker(service=service, worker_ref="worker-1")
    registrations = {"baseline": object(), "challenger": object()}
    observed = {}

    class _StopAfterConstruction(Exception):
        pass

    class _Heartbeat:
        deadline_monotonic = time.monotonic() + 60

        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def ensure_held(self):
            pass

    class _Dispatcher:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            raise _StopAfterConstruction()

    monkeypatch.setattr(worker_module, "_RoutingClaimHeartbeat", _Heartbeat)
    monkeypatch.setattr(
        worker_module,
        "ReviewedProtectedModelActionDispatcher",
        _Dispatcher,
    )
    monkeypatch.setattr(worker, "_append_execution_event", lambda **_kwargs: None)
    monkeypatch.setattr(worker, "_close_claim", lambda **_kwargs: None)

    inputs = SimpleNamespace(
        execution_envelope=None,
        registry_registrations=registrations,
        reviewed_runner=object(),
        verifier=object(),
    )
    with pytest.raises(_StopAfterConstruction):
        worker._run_exact_model(spec=_Spec(), inputs=inputs, lease=service.lease)

    assert observed["registrations"] is registrations


def test_exact_intent_run_never_enters_candidate_terminal_path(monkeypatch):
    class _Heartbeat:
        deadline_monotonic = time.monotonic() + 60

        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def ensure_held(self):
            pass

    class _ExactStore(_Store):
        def __init__(self):
            super().__init__()
            self.decisions = []
            self.evaluations = []

        def append_decision(self, **values):
            self.decisions.append(values)

        def append_evaluation(self, **values):
            self.evaluations.append(values)

    class _Coordinator:
        def __init__(self, **_kwargs):
            pass

        def run_unit(self, **_kwargs):
            return SimpleNamespace(provider_receipts=(), terminal_result={})

    class _EvaluationAdapter:
        def build_decision_receipts(self, **_kwargs):
            return (_synthetic_intent_decision(),)

        def build_evaluation(self, **kwargs):
            decision = _synthetic_intent_decision()
            return _synthetic_intent_evaluation(decision=decision)

    def _candidate_path_must_not_run(**_kwargs):
        raise AssertionError("intent execution entered candidate-only sidecars")

    monkeypatch.setattr(worker_module, "_RoutingClaimHeartbeat", _Heartbeat)
    monkeypatch.setattr(
        worker_module,
        "ReviewedProtectedModelActionDispatcher",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        worker_module,
        "FencedModelTransitionRepository",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(worker_module, "ExactModelExperimentCoordinator", _Coordinator)
    monkeypatch.setattr(
        worker_module,
        "routing_experiment_v2_artifact_key",
        lambda _variant: _hash("c"),
    )
    monkeypatch.setattr(
        worker_module,
        "candidate_model_unit_terminal_from_exact_model",
        _candidate_path_must_not_run,
    )
    monkeypatch.setattr(
        worker_module,
        "candidate_waterfall_receipts_from_exact_model",
        _candidate_path_must_not_run,
    )
    monkeypatch.setattr(
        worker_module,
        "evaluate_candidate_waterfall_metrics",
        _candidate_path_must_not_run,
    )

    service = _Service()
    service.store = _ExactStore()
    worker = RoutingExperimentWorker(service=service, worker_ref="worker-1")
    variant = SimpleNamespace(
        variant_id="baseline",
        stage="intent_evidence",
        artifact=SimpleNamespace(to_dict=lambda: {"artifact": "exact"}),
    )
    spec = SimpleNamespace(
        experiment_id="intent-experiment",
        baseline_variant_id="baseline",
        allow_live_credit_spend=False,
        input=SimpleNamespace(
            stage="intent_evidence",
            calibration_unit_refs=("unit-1",),
            holdout_unit_refs=(),
        ),
        variants=(variant,),
        experiment_hash=lambda: _hash("a"),
    )
    inputs = SimpleNamespace(
        execution_envelope=None,
        registry_registrations={"baseline": object()},
        reviewed_runner=object(),
        verifier=object(),
        registry=SimpleNamespace(resolve=lambda _identity: object()),
        unit_dataset=SimpleNamespace(
            resolve=lambda _unit_ref: (
                {
                    "model_input": {"kind": "normalized_icp"},
                    "execution_mode": "intent_refresh",
                    "target_count": 1,
                    "evaluated_on": "2026-08-22",
                },
                _hash("f"),
            )
        ),
        evaluation_adapter=_EvaluationAdapter(),
        gold_labels={},
        authoritative_billing_rollup=None,
    )

    result = worker._run_exact_model(
        spec=spec,
        inputs=inputs,
        lease=service.lease,
    )

    assert result is not None
    assert len(service.store.decisions) == 1
    assert len(service.store.evaluations) == 1
    assert [item["event_type"] for item in service.store.events] == [
        "run_started",
        "run_completed",
    ]
    assert service.store.closed[0]["close_reason"] == "completed"


def test_exact_decision_set_fails_closed_before_persistence(monkeypatch):
    spec = _synthetic_intent_spec()
    monkeypatch.setattr(
        worker_module,
        "routing_experiment_v2_artifact_key",
        lambda _variant: _hash("c"),
    )
    empty_unit_results = {
        "baseline": {"unit-1": SimpleNamespace(provider_receipts=())}
    }
    valid = _synthetic_intent_decision()

    indexed = worker_module._index_exact_model_decision_receipts(
        spec=spec,
        decisions=(valid,),
        unit_results=empty_unit_results,
    )
    assert indexed == {("baseline", "unit-1"): (valid,)}

    invalid_sets = (
        ((), "coverage"),
        ((valid, valid), "duplicated"),
        (
            (
                _refinalize_decision(
                    valid,
                    stage="candidate_acquisition",
                ),
            ),
            "lineage",
        ),
        (
            (
                _refinalize_decision(
                    valid,
                    artifact_key=_hash("f"),
                ),
            ),
            "lineage",
        ),
        (
            (
                _refinalize_decision(
                    valid,
                    experiment_id="another-experiment",
                ),
            ),
            "lineage",
        ),
    )
    for decisions, message in invalid_sets:
        with pytest.raises(RoutingExperimentWorkerError, match=message):
            worker_module._index_exact_model_decision_receipts(
                spec=spec,
                decisions=decisions,
                unit_results=empty_unit_results,
            )

    missing_provider_decision = _synthetic_intent_decision(
        provider_receipt_refs=("provider_receipt:" + "a" * 16,)
    )
    with pytest.raises(RoutingExperimentWorkerError, match="provider receipt"):
        worker_module._index_exact_model_decision_receipts(
            spec=spec,
            decisions=(missing_provider_decision,),
            unit_results=empty_unit_results,
        )


def test_exact_evaluation_is_bound_to_validated_receipts(monkeypatch):
    spec = _synthetic_intent_spec()
    monkeypatch.setattr(
        worker_module,
        "routing_experiment_v2_artifact_key",
        lambda _variant: _hash("c"),
    )
    unit_results = {
        "baseline": {"unit-1": SimpleNamespace(provider_receipts=())}
    }
    decision = _synthetic_intent_decision()
    decisions_by_unit = {("baseline", "unit-1"): (decision,)}
    evaluation = _synthetic_intent_evaluation(decision=decision)

    worker_module._validate_exact_model_evaluation(
        spec=spec,
        evaluation=evaluation,
        decisions_by_unit=decisions_by_unit,
        unit_results=unit_results,
    )

    wrong_artifact = replace(
        evaluation,
        variants=(
            replace(evaluation.variants[0], artifact_key=_hash("f")),
        ),
    )
    missing_decisions = replace(
        evaluation,
        variants=(
            replace(evaluation.variants[0], decision_receipt_refs=()),
        ),
        decision_receipt_refs=(),
    )
    selected_failure = replace(
        evaluation,
        variants=(replace(evaluation.variants[0], passed=False),),
    )
    wrong_identity = replace(
        evaluation,
        receipt_id="routing_evaluation_v2:" + "f" * 16,
    )
    for forged, message in (
        (wrong_artifact, "lineage"),
        (missing_decisions, "lineage"),
        (selected_failure, "selection"),
        (wrong_identity, "identity"),
    ):
        with pytest.raises(RoutingExperimentWorkerError, match=message):
            worker_module._validate_exact_model_evaluation(
                spec=spec,
                evaluation=forged,
                decisions_by_unit=decisions_by_unit,
                unit_results=unit_results,
            )


def test_exact_evaluation_rejects_provider_receipt_reuse_across_variants(monkeypatch):
    variants = (
        SimpleNamespace(
            variant_id="baseline",
            stage="intent_evidence",
            artifact=SimpleNamespace(to_dict=lambda: {"artifact": "baseline"}),
        ),
        SimpleNamespace(
            variant_id="challenger",
            stage="intent_evidence",
            artifact=SimpleNamespace(to_dict=lambda: {"artifact": "challenger"}),
        ),
    )
    spec = SimpleNamespace(
        experiment_id="intent-experiment",
        baseline_variant_id="baseline",
        allow_live_credit_spend=False,
        input=SimpleNamespace(
            stage="intent_evidence",
            calibration_unit_refs=("unit-1",),
            holdout_unit_refs=(),
        ),
        variants=variants,
        experiment_hash=lambda: _hash("a"),
    )
    artifact_keys = {"baseline": _hash("c"), "challenger": _hash("f")}
    monkeypatch.setattr(
        worker_module,
        "routing_experiment_v2_artifact_key",
        lambda variant: artifact_keys[variant.variant_id],
    )
    shared_provider_ref = "provider_receipt:" + "a" * 16
    decisions = (
        _synthetic_intent_decision(
            variant_id="baseline",
            artifact_key=artifact_keys["baseline"],
            provider_receipt_refs=(shared_provider_ref,),
        ),
        _synthetic_intent_decision(
            variant_id="challenger",
            artifact_key=artifact_keys["challenger"],
            provider_receipt_refs=(shared_provider_ref,),
        ),
    )
    provider_receipt = SimpleNamespace(receipt_ref=shared_provider_ref)
    unit_results = {
        variant.variant_id: {
            "unit-1": SimpleNamespace(provider_receipts=(provider_receipt,))
        }
        for variant in variants
    }
    decisions_by_unit = {
        (decision.variant_id, decision.unit_ref): (decision,)
        for decision in decisions
    }
    variant_evaluations = tuple(
        RoutingExperimentV2VariantEvaluation(
            variant_id=decision.variant_id,
            artifact_key=artifact_keys[decision.variant_id],
            stage="intent_evidence",
            calibration=_SyntheticMetrics(),
            holdout=_SyntheticMetrics(),
            passed_precision_gate=True,
            passed_recall_gate=True,
            passed_cost_gate=True,
            passed_efficiency_gate=True,
            passed=True,
            decision_receipt_refs=(decision.receipt_id,),
            provider_receipt_refs=(shared_provider_ref,),
        )
        for decision in decisions
    )
    draft = RoutingExperimentV2Evaluation(
        receipt_id="routing_evaluation_v2:pending",
        experiment_id=spec.experiment_id,
        experiment_hash=spec.experiment_hash(),
        variants=variant_evaluations,
        baseline_variant_id=spec.baseline_variant_id,
        selected_variant_id="baseline",
        decision_receipt_refs=tuple(
            sorted(decision.receipt_id for decision in decisions)
        ),
        provider_receipt_refs=(shared_provider_ref,),
        provider_cache_hits=1,
        provider_cache_misses=1,
    )
    evaluation = replace(
        draft,
        receipt_id=(
            "routing_evaluation_v2:"
            + worker_module.sha256_json(draft.to_dict()).split(":", 1)[1][:16]
        ),
    )

    with pytest.raises(RoutingExperimentWorkerError, match="duplicated"):
        worker_module._validate_exact_model_evaluation(
            spec=spec,
            evaluation=evaluation,
            decisions_by_unit=decisions_by_unit,
            unit_results=unit_results,
        )


def test_claim_heartbeat_uses_sql_expiry_after_a_delayed_renewal():
    lease = RoutingExecutionRequestLease(
        request_hash=_hash("d"),
        experiment_hash=_hash("a"),
        lease_hash=_hash("e"),
        worker_ref="worker-1",
        lease_generation=1,
        lease_expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    claim = RoutingExperimentExecutionClaim(
        lease.experiment_hash,
        _hash("b"),
        1,
        routing_claim_fence_hash_v3(
            experiment_hash=lease.experiment_hash,
            claim_key=_hash("b"),
            claim_generation=1,
        ),
        lease.request_hash,
        lease.lease_hash,
        lease.lease_generation,
        lease.worker_ref,
        lease.lease_expires_at,
    )
    authoritative_expiry = datetime.now(timezone.utc) + timedelta(seconds=20)

    class _DelayedStore:
        def renew_claim(self, **kwargs):
            time.sleep(0.05)
            return {
                "renewed": True,
                "idempotent": False,
                "heartbeat_key": kwargs["heartbeat_key"],
                "lease_expires_at": authoritative_expiry.isoformat(),
            }

    heartbeat = _RoutingClaimHeartbeat(
        store=_DelayedStore(),
        claim=claim,
        lease_seconds=3600,
    )
    heartbeat._renew_once()
    assert heartbeat.deadline_monotonic <= time.monotonic() + 5.1


@pytest.mark.parametrize("lease_expires_at", ["", "later", "2099-01-01T00:00:00"])
def test_claim_identity_rejects_missing_malformed_or_naive_sql_expiry(
    lease_expires_at,
):
    lease = RoutingExecutionRequestLease(
        request_hash=_hash("d"),
        experiment_hash=_hash("a"),
        lease_hash=_hash("e"),
        worker_ref="worker-1",
        lease_generation=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    with pytest.raises(RoutingExperimentStoreError, match="claim lease expiry is invalid"):
        RoutingExperimentExecutionClaim(
            lease.experiment_hash,
            _hash("b"),
            1,
            routing_claim_fence_hash_v3(
                experiment_hash=lease.experiment_hash,
                claim_key=_hash("b"),
                claim_generation=1,
            ),
            lease.request_hash,
            lease.lease_hash,
            lease.lease_generation,
            lease.worker_ref,
            lease_expires_at,
        )


def test_coordinator_only_accepts_a_reviewed_named_factory():
    service = _Service()
    worker = RoutingExperimentWorker(service=service, worker_ref="worker-1")
    coordinator = RoutingExperimentCoordinator(worker=worker, factories={})
    with pytest.raises(RoutingExperimentWorkerError, match="factory is unavailable"):
        coordinator.resume(experiment_hash=_hash("a"), factory_name="attested_provider_broker_v2")


def test_named_factory_accepts_only_exact_variant_adapters_and_a_real_reviewed_runner():
    class _Variant:
        def __init__(self, variant_id):
            self.variant_id = variant_id

    spec = type("Spec", (), {"variants": (_Variant("baseline"), _Variant("candidate"))})()

    class _ReviewedExecutor(RoutingProviderDispatchExecutor):
        _routing_dispatch_executor_token = _ROUTING_DISPATCH_EXECUTOR_TOKEN

        def __call__(self, _request):
            raise AssertionError("factory validation must not execute a TEE job")

    protected_receipt = authority_fixture()["attempts"][0]["attempt_doc"][
        "protected_release_receipt"
    ]

    class _ArtifactAuthority:
        def verify(self, **_kwargs):
            return {"verified": True}

    runner = ReviewedProviderBrokerRoutingRunner(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=object(),
        artifact_lineage=object(),
        compiler=object(),
        model_binding_requirements=object(),
        authorization_authority=AttestedScoringV2RoutingProviderCallAuthority(
            executor=_ReviewedExecutor()
        ),
        dispatch_authority=AttestedScoringV2RoutingProviderDispatchAuthority(
            executor=_ReviewedExecutor(),
            protected_release_receipt=protected_receipt,
        ),
        authorization_parent_receipt_graphs=({"receipts": []},),
        dispatch_parent_receipt_graphs=({"receipts": []},),
    )
    envelope = RoutingExperimentExecutionEnvelopeV2.from_mapping(
        authority_fixture()["execution_envelope"]
    )
    def reviewed_runner_factory(_spec):
        return runner

    reviewed_runner_factory.validate_readiness = lambda: None
    factory = AttestedProviderBrokerRoutingRunFactory(
        adapter_factory=lambda _spec: {"baseline": object(), "candidate": object()},
        gold_label_loader=lambda _spec: {"company-1": True},
        reviewed_runner_factory=reviewed_runner_factory,
        artifact_authority=_ArtifactAuthority(),
        billing_rollup_factory=lambda _spec: (lambda _receipts: {}),
        execution_envelope_factory=lambda _spec: envelope,
    )
    factory.validate_readiness()
    inputs = factory.build(spec)
    assert inputs.runner is runner
    assert set(inputs.adapters) == {"baseline", "candidate"}

    with pytest.raises(RoutingExperimentWorkerError, match="artifact authority"):
        replace(factory, artifact_authority=object()).validate_readiness()

    with pytest.raises(RoutingExperimentWorkerError, match="runner readiness"):
        replace(factory, reviewed_runner_factory=lambda _spec: runner).validate_readiness()

    incomplete = AttestedProviderBrokerRoutingRunFactory(
        adapter_factory=lambda _spec: {"baseline": object()},
        gold_label_loader=lambda _spec: {},
        reviewed_runner_factory=lambda _spec: runner,
        artifact_authority=object(),
        billing_rollup_factory=lambda _spec: (lambda _receipts: {}),
        execution_envelope_factory=lambda _spec: envelope,
    )
    with pytest.raises(RoutingExperimentWorkerError, match="adapter map is incomplete"):
        incomplete.build(spec)

    generic = AttestedProviderBrokerRoutingRunFactory(
        adapter_factory=lambda _spec: {"baseline": object(), "candidate": object()},
        gold_label_loader=lambda _spec: {},
        reviewed_runner_factory=lambda _spec: object(),
        artifact_authority=object(),
        billing_rollup_factory=lambda _spec: (lambda _receipts: {}),
        execution_envelope_factory=lambda _spec: envelope,
    )
    with pytest.raises(RoutingExperimentWorkerError, match="routing runner is invalid"):
        generic.build(spec)


def test_reviewed_runner_rejects_direct_broker_or_generic_execute_at_construction():
    common = dict(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=object(),
        artifact_lineage=object(),
        compiler=object(),
        model_binding_requirements=object(),
        authorization_authority=AttestedScoringV2RoutingProviderCallAuthority(
            executor=None
        ),
        dispatch_parent_receipt_graphs=({"receipts": []},),
    )
    with pytest.raises(TypeError):
        ReviewedProviderBrokerRoutingRunner(
            **common,
            dispatch_authority=AttestedScoringV2RoutingProviderDispatchAuthority(
                executor=None, protected_release_receipt=None
            ),
            broker=object(),
        )
    with pytest.raises(RoutingExperimentRuntimeError, match="dispatch authority"):
        ReviewedProviderBrokerRoutingRunner(
            **common,
            dispatch_authority=type("GenericExecute", (), {"execute": lambda self, _: {}})(),
        )


def test_cli_check_config_and_run_fail_closed_without_a_reviewed_factory(capsys):
    class _Config:
        enabled = True
        live_execution_enabled = False
        attested_authority_mode = ""

    assert main(["--check-config"], config_factory=lambda: _Config()) == 0
    assert '"enabled": true' in capsys.readouterr().out

    # A run has a complete CLI path, but the default registry contains no
    # factory. It therefore cannot load arbitrary code or reach a provider.
    assert (
        main(
            ["--run", _hash("a")],
            config_factory=lambda: _Config(),
            store_factory=lambda: _Store(_Spec()),
        )
        == 2
    )
    assert "factory is unavailable" in capsys.readouterr().out
