from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import time

import pytest

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
from tests.routing_experiment_authority_fixture import authority_fixture


def _hash(char: str) -> str:
    return "sha256:" + char * 64


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
