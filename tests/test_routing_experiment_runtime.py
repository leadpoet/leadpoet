from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from gateway.research_lab.routing_experiment_runtime import (
    KmsRoutingExperimentArtifactAuthority,
    ProviderBrokerV2RoutingExecutor,
    ProviderBrokerRoutingRunner,
    ReviewedProviderBrokerRoutingRunner,
    RoutingExperimentDeferredRecoveryError,
    RoutingExperimentRuntimeConfig,
    RoutingExperimentRuntimeError,
    RoutingExperimentService,
    RoutingExperimentTerminalRecoveryError,
    _provider_attempt_replay_ref,
)
from gateway.research_lab.routing_admission import RoutingAdmissionBundleV2
from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    build_routing_model_completion_contract_v1,
)
from gateway.research_lab.routing_experiment_store import (
    RoutingExperimentExecutionClaim,
    RoutingExecutionRequestLease,
    SupabaseRoutingExperimentStore,
    routing_claim_fence_hash_v3,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    ReceiptExecutionMode,
    RoutingCallAuthorization,
)
from tests.routing_experiment_authority_fixture import authority_fixture


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _binding() -> ProviderBindingIdentity:
    return ProviderBindingIdentity(
        binding_id="deepline-jobs",
        provider_id="deepline",
        tool_id="intent.source_add.bloomberry_jobs",
        source_lineage_id="deepline.bloomberry.jobs",
        adapter_version="v1",
        manifest_hash=_hash("1"),
        capability_hash=_hash("2"),
        execution_contract_hash=_hash("3"),
        cost_model_hash=_hash("4"),
    )


def _authorization(*, credit: int = 10, timeout_ms: int = 100) -> RoutingCallAuthorization:
    return RoutingCallAuthorization(
        experiment_id="routing-experiment",
        variant_id="candidate",
        artifact_key=_hash("5"),
        stage="intent_evidence",
        unit_ref="company-1",
        tool_id="intent.source_add.bloomberry_jobs",
        attempt=0,
        request_fingerprint=_hash("6"),
        remaining_credit_microunits=credit,
        timeout_ceiling_ms=timeout_ms,
        execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )


@dataclass
class _Store:
    calls: list[tuple[str, dict]]

    def reserve_budget(self, **kwargs):
        self.calls.append(("reserve", kwargs))
        claim = kwargs["claim"]
        return {
            "schema_version": "leadpoet.research_lab.routing_budget_reservation_result.v3",
            "reserved": True,
            "idempotent": False,
            "reservation_id": kwargs["reservation_id"],
            "event_key": kwargs["event_key"],
            "experiment_hash": kwargs["experiment_hash"],
            "binding_id": kwargs["binding_id"],
            "claim_key": claim.claim_key,
            "claim_generation": claim.claim_generation,
            "credit_microunits": kwargs["credit_microunits"],
            "lease_expires_at": "2099-01-01T00:00:00+00:00",
        }

    def append_provider_attempt(self, **kwargs):
        self.calls.append(("attempt", kwargs))
        return {"attempt_key": kwargs["key"]}

    def settle_budget(self, **kwargs):
        self.calls.append(("settle", kwargs))
        return {"settled": True}

    def append_event(self, **kwargs):
        self.calls.append(("dispatch", kwargs))
        return {"event_hash": _hash("e")}

    def mark_budget_uncertain(self, **kwargs):
        self.calls.append(("uncertain", kwargs))
        return {"uncertain": True, "credit_microunits": 10}


class _Authority:
    def authorize(self, **kwargs):
        binding = kwargs["binding"]
        return {
            "attested": True,
            "operation": "routing_experiment_v2",
            "purpose": "research_lab.routing_provider_evidence.v2",
            "binding_id": binding.binding_id,
        }


def _runner(*, broker, store=None):
    config = RoutingExperimentRuntimeConfig(
        enabled=True,
        live_execution_enabled=True,
        worker_lease_seconds=60,
        evidence_proxy_url="http://proxy.invalid",
        attested_authority_mode="attested",
    )
    root = ProviderBrokerRoutingRunner(
        config=config,
        store=store or _Store([]),
        execution_authority=_Authority(),
        broker_executor=broker,
    )
    claim = RoutingExperimentExecutionClaim(
        _hash("a"),
        _hash("b"),
        1,
        routing_claim_fence_hash_v3(
            experiment_hash=_hash("a"),
            claim_key=_hash("b"),
            claim_generation=1,
        ),
    )
    return root, root.for_execution(_hash("a"), "routing-experiment", claim)


def _result(*, credit: int = 10, timeout_ms: int = 1):
    return {
        "outcome": "source_miss",
        "evidence_hash": _hash("d"),
        "credit_microunits": credit,
        "latency_ms": timeout_ms,
        "call_count": 1,
        "billing_state": "known",
        "binding_id": "deepline-jobs",
        "provider_id": "deepline",
        "tool_id": "intent.source_add.bloomberry_jobs",
        "request_fingerprint": _hash("6"),
    }


def test_measured_runner_binds_exact_execution_and_persists_only_valid_broker_result():
    root, bound = _runner(broker=lambda _request: _result())
    with pytest.raises(RoutingExperimentRuntimeError, match="not bound"):
        root(_binding(), "company-1", _hash("6"), _authorization())

    receipt = bound(_binding(), "company-1", _hash("6"), _authorization())
    assert receipt["tool_id"] == "intent.source_add.bloomberry_jobs"
    assert [name for name, _kwargs in bound.store.calls] == ["reserve", "dispatch", "attempt", "settle"]
    broker_request = bound.broker_executor  # The runner owns the route; no direct HTTP path exists.
    assert broker_request is not None


def test_legacy_runner_rejects_v3_durable_store_before_provider_dispatch():
    broker_calls = []
    durable_store = SupabaseRoutingExperimentStore(object())
    _root, bound = _runner(
        broker=lambda request: broker_calls.append(request) or _result(),
        store=durable_store,
    )
    with pytest.raises(
        RoutingExperimentRuntimeError,
        match="legacy provider broker runner is incompatible with V3 durable store",
    ):
        bound(_binding(), "company-1", _hash("6"), _authorization())
    assert broker_calls == []


def test_legacy_v2_executor_rejects_v3_durable_store_at_construction():
    durable_store = SupabaseRoutingExperimentStore(object())
    with pytest.raises(
        RoutingExperimentRuntimeError,
        match="legacy provider broker executor is incompatible with V3 durable store",
    ):
        ProviderBrokerV2RoutingExecutor(
            broker=object(),
            broker_request_factory=lambda request: request,
            result_projector=lambda request, response: response,
            store=durable_store,
        )


def test_invalid_broker_cost_never_becomes_a_provider_attempt_and_marks_budget_uncertain():
    root, bound = _runner(broker=lambda _request: _result(credit=11))
    with pytest.raises(
        RoutingExperimentDeferredRecoveryError,
        match="budget was marked uncertain",
    ):
        bound(_binding(), "company-1", _hash("6"), _authorization(credit=10))
    assert [name for name, _kwargs in bound.store.calls] == ["reserve", "dispatch", "uncertain"]


def test_reservation_failure_never_reaches_provider_and_conservatively_marks_uncertain():
    class _ExpiredReservationStore(_Store):
        def reserve_budget(self, **kwargs):
            self.calls.append(("reserve", kwargs))
            raise RuntimeError("reservation expired")

    broker_calls = []
    _root, bound = _runner(
        broker=lambda request: broker_calls.append(request) or _result(),
        store=_ExpiredReservationStore([]),
    )
    with pytest.raises(
        RoutingExperimentDeferredRecoveryError,
        match="budget was marked uncertain",
    ):
        bound(_binding(), "company-1", _hash("6"), _authorization())
    assert broker_calls == []
    assert [name for name, _kwargs in bound.store.calls] == ["reserve", "uncertain"]


def test_reservation_recovery_failure_is_deferred_and_never_reaches_provider():
    class _UnrecoverableReservationStore(_Store):
        def reserve_budget(self, **kwargs):
            self.calls.append(("reserve", kwargs))
            raise RuntimeError("reservation response lost")

        def mark_budget_uncertain(self, **kwargs):
            self.calls.append(("uncertain", kwargs))
            raise RuntimeError("authority unavailable")

    broker_calls = []
    _root, bound = _runner(
        broker=lambda request: broker_calls.append(request) or _result(),
        store=_UnrecoverableReservationStore([]),
    )
    with pytest.raises(RoutingExperimentDeferredRecoveryError, match="recovery could not be confirmed"):
        bound(_binding(), "company-1", _hash("6"), _authorization())
    assert broker_calls == []
    assert [name for name, _kwargs in bound.store.calls] == ["reserve", "uncertain"]


def test_malformed_host_reservation_result_never_reaches_provider():
    class _MalformedReservationStore(_Store):
        def reserve_budget(self, **kwargs):
            self.calls.append(("reserve", kwargs))
            return {}

    broker_calls = []
    _root, bound = _runner(
        broker=lambda request: broker_calls.append(request) or _result(),
        store=_MalformedReservationStore([]),
    )
    with pytest.raises(
        RoutingExperimentDeferredRecoveryError,
        match="budget was marked uncertain",
    ):
        bound(_binding(), "company-1", _hash("6"), _authorization())
    assert broker_calls == []
    assert [name for name, _kwargs in bound.store.calls] == ["reserve", "uncertain"]


def test_dispatch_marker_failure_never_reaches_provider_and_keeps_full_reservation_uncertain():
    class _DispatchFailureStore(_Store):
        def append_event(self, **kwargs):
            self.calls.append(("dispatch", kwargs))
            raise RuntimeError("dispatch marker unavailable")

    broker_calls = []
    _root, bound = _runner(
        broker=lambda request: broker_calls.append(request) or _result(),
        store=_DispatchFailureStore([]),
    )
    with pytest.raises(
        RoutingExperimentDeferredRecoveryError,
        match="budget was marked uncertain",
    ):
        bound(_binding(), "company-1", _hash("6"), _authorization())
    assert broker_calls == []
    assert [name for name, _kwargs in bound.store.calls] == ["reserve", "dispatch", "uncertain"]


def test_bounded_claim_lease_rejects_a_call_that_cannot_finish_before_the_fence():
    _root, bound = _runner(broker=lambda _request: _result())
    bound._execution = bound._execution.__class__(
        experiment_hash=bound._execution.experiment_hash,
        experiment_id=bound._execution.experiment_id,
        claim=bound._execution.claim,
        deadline_monotonic=0.0,
    )
    with pytest.raises(RoutingExperimentRuntimeError, match="lease is exhausted"):
        bound(_binding(), "company-1", _hash("6"), _authorization())
    assert bound.store.calls == []


def test_artifact_authority_requires_full_manifest_signature_binding():
    artifact = type(
        "Artifact",
        (),
        {
            "model_artifact_hash": _hash("1"),
            "manifest_hash": _hash("2"),
            "commit_sha": "3" * 40,
        },
    )()
    manifest = {"signature_ref": "s3://private/manifest.sig"}
    authority = KmsRoutingExperimentArtifactAuthority(
        verifier=lambda _manifest: {
            "verified": True,
            "manifest_hash": _hash("2"),
            "signature_ref": "s3://private/manifest.sig",
            "key_id": "kms-key",
            "signing_algorithm": "ECDSA_SHA_256",
            "consumer_contract_binding_mode": "semantic_v1_required",
        }
    )
    assert authority.verify(artifact=artifact, manifest=manifest)["verified"] is True

    with pytest.raises(RoutingExperimentRuntimeError, match="binding is incomplete"):
        KmsRoutingExperimentArtifactAuthority(
            verifier=lambda _manifest: {"verified": True, "manifest_hash": _hash("2")}
        ).verify(artifact=artifact, manifest=manifest)


def test_service_recovers_an_expired_claim_and_requires_a_new_experiment():
    lease = RoutingExecutionRequestLease(
        request_hash=_hash("a"),
        experiment_hash=_hash("b"),
        lease_hash=_hash("c"),
        worker_ref="worker-1",
        lease_generation=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    claim_key = sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_claim_key.v3",
            "experiment_hash": _hash("b"),
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "worker_ref": lease.worker_ref,
        }
    )

    class _RecoveryStore:
        def __init__(self):
            self.claim_calls = []
            self.recovery_calls = []

        def claim_execution(self, **kwargs):
            self.claim_calls.append(kwargs)
            return {
                "claimed": False,
                "recoverable": True,
                "claim_key": claim_key,
                "claim_generation": 1,
                "request_hash": lease.request_hash,
                "lease_hash": lease.lease_hash,
                "lease_generation": lease.lease_generation,
            }

        def recover_claim(self, **kwargs):
            self.recovery_calls.append(kwargs)
            return {"recovered": True, "claim_generation": 2}

        def unresolved_budget_reservations(self, **kwargs):
            return ()

    spec = type("Spec", (), {"experiment_hash": lambda self: _hash("b")})()
    store = _RecoveryStore()
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )
    with pytest.raises(
        RoutingExperimentRuntimeError,
        match="submit a new immutable experiment",
    ):
        service.claim_execution(spec=spec, worker_ref="worker-1", lease=lease)
    assert len(store.claim_calls) == 1
    assert len(store.recovery_calls) == 1
    assert store.recovery_calls[0]["recovery_doc"] == {
        "schema_version": "leadpoet.research_lab.routing_claim_recovery.v3",
        "worker_ref": "worker-1",
        "stale_claim_key": claim_key,
        "stale_claim_generation": 1,
    }


def test_service_handles_sql_stale_claim_key_before_prospective_key_check():
    lease = RoutingExecutionRequestLease(
        request_hash=_hash("a"),
        experiment_hash=_hash("b"),
        lease_hash=_hash("c"),
        worker_ref="worker-1",
        lease_generation=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    prospective_key = sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_claim_key.v3",
            "experiment_hash": _hash("b"),
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "worker_ref": lease.worker_ref,
        }
    )
    stale_key = _hash("d")

    class _SqlShapedRecoveryStore:
        def __init__(self):
            self.recovery_calls = []

        def claim_execution(self, **kwargs):
            assert kwargs["claim_key"] == prospective_key
            return {
                "claimed": False,
                "recoverable": True,
                "claim_key": stale_key,
                "claim_generation": 4,
            }

        def recover_claim(self, **kwargs):
            self.recovery_calls.append(kwargs)
            return {"recovered": True}

    spec = type("Spec", (), {"experiment_hash": lambda self: _hash("b")})()
    store = _SqlShapedRecoveryStore()
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )
    with pytest.raises(RoutingExperimentRuntimeError, match="submit a new immutable experiment"):
        service.claim_execution(spec=spec, worker_ref="worker-1", lease=lease)
    expected_recovery_key = sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_claim_recovery_key.v3",
            "experiment_hash": _hash("b"),
            "stale_claim_key": stale_key,
            "stale_claim_generation": 4,
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "worker_ref": lease.worker_ref,
        }
    )
    assert store.recovery_calls[0]["recovery_key"] == expected_recovery_key
    assert store.recovery_calls[0]["recovery_key"] != prospective_key
    assert store.recovery_calls[0]["recovery_doc"]["stale_claim_key"] == stale_key


def test_service_does_not_resume_or_mark_budget_after_terminal_recovery():
    lease = RoutingExecutionRequestLease(
        request_hash=_hash("a"),
        experiment_hash=_hash("b"),
        lease_hash=_hash("c"),
        worker_ref="worker-1",
        lease_generation=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    claim_key = sha256_json(
        {
            "schema_version": "leadpoet.research_lab.routing_claim_key.v3",
            "experiment_hash": _hash("b"),
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "worker_ref": lease.worker_ref,
        }
    )

    class _UncertainRecoveryStore:
        def __init__(self):
            self.claim_calls = []
            self.recovery_calls = []
            self.uncertain_calls = []

        def claim_execution(self, **kwargs):
            self.claim_calls.append(kwargs)
            return {
                "claimed": False,
                "recoverable": True,
                "claim_key": claim_key,
                "claim_generation": 1,
                "request_hash": lease.request_hash,
                "lease_hash": lease.lease_hash,
                "lease_generation": lease.lease_generation,
            }

        def recover_claim(self, **kwargs):
            self.recovery_calls.append(kwargs)
            return {"recovered": True, "claim_generation": 2}

        def unresolved_budget_reservations(self, **kwargs):
            raise AssertionError("terminal recovery must not inspect a fresh claim")

        def mark_budget_uncertain(self, **kwargs):
            self.uncertain_calls.append(kwargs)
            return {"uncertain": True}

    spec = type("Spec", (), {"experiment_hash": lambda self: _hash("b")})()
    store = _UncertainRecoveryStore()
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )
    with pytest.raises(RoutingExperimentRuntimeError, match="submit a new immutable experiment"):
        service.claim_execution(spec=spec, worker_ref="worker-1", lease=lease)
    assert len(store.claim_calls) == 1
    assert len(store.recovery_calls) == 1
    assert store.uncertain_calls == []


def _durable_model_attempt_fixture():
    fixture = authority_fixture()
    row = deepcopy(fixture["attempts"][0])
    document = deepcopy(row["attempt_doc"])
    grant = RoutingProviderCallAuthorizationV2.from_mapping(
        document["call_grant"]
    )
    admission = RoutingAdmissionBundleV2.from_mapping(
        document["admission_bundle"]
    )
    lineage = VerifiedRoutingArtifactLineage(**fixture["lineage"])
    action = {
        "action_sha256": "a" * 64,
        "response_schema_version": "model-provider-response:v1",
        "max_response_bytes": 1_000_000,
    }
    response = {
        "schema_version": "host-provider-response:v1",
        "provider": "fixture",
        "status_code": 200,
        "body": {},
    }
    terminal_result = {
        "provider_receipt": document["provider_receipt"],
        "model_provider_response": response,
        "model_provider_response_sha256": sha256_json(response),
        "model_completion_contract_hash": sha256_json(
            build_routing_model_completion_contract_v1(action)
        ),
    }
    terminal_receipt = {
        "job_id": "routing-dispatch:" + "1" * 32,
        "receipt_hash": _hash("2"),
        "input_root": _hash("3"),
    }
    document.pop("legacy_fixture", None)
    document.update(
        {
            "schema_version": (
                "leadpoet.research_lab.routing_provider_attempt.v3"
            ),
            "terminal_result": terminal_result,
            "terminal_execution_receipt": terminal_receipt,
            "terminal_request_hash": terminal_receipt["input_root"],
        }
    )
    row.update(
        {
            "attempt_doc": document,
            "terminal_result_hash": sha256_json(terminal_result),
            "terminal_receipt_hash": terminal_receipt["receipt_hash"],
        }
    )
    authorization = RoutingCallAuthorization(
        experiment_id=grant.experiment_id,
        variant_id=grant.variant_id,
        artifact_key=_hash("5"),
        stage=grant.stage,
        unit_ref=grant.unit_ref,
        tool_id=grant.binding.tool_id,
        attempt=grant.attempt,
        request_fingerprint=grant.core_request_fingerprint,
        remaining_credit_microunits=grant.credit_cap_microunits,
        timeout_ceiling_ms=grant.timeout_ms,
        execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    return {
        "row": row,
        "document": document,
        "grant": grant,
        "admission": admission,
        "lineage": lineage,
        "action": action,
        "authorization": authorization,
    }


def _resolve_durable_model_attempt(values):
    grant = values["grant"]
    return _provider_attempt_replay_ref(
        values["row"],
        attempt_key=values["row"]["attempt_key"],
        experiment_hash=values["row"]["experiment_hash"],
        experiment_id=grant.experiment_id,
        authorization=values["authorization"],
        binding=grant.binding,
        artifact_lineage=values["lineage"],
        unit_ref=grant.unit_ref,
        request_fingerprint=grant.core_request_fingerprint,
        action=values["action"],
        admission=values["admission"],
        protected_release_receipt=values["document"][
            "protected_release_receipt"
        ],
    )


def test_durable_model_attempt_derives_the_exact_protected_replay_reference():
    values = _durable_model_attempt_fixture()

    assert _resolve_durable_model_attempt(values) == {
        "schema_version": (
            "leadpoet.research_lab.protected_model_replay_ref.v1"
        ),
        "protected_dispatch_job_id": "routing-dispatch:" + "1" * 32,
        "terminal_receipt_hash": _hash("2"),
        "model_provider_response_sha256": values["document"][
            "terminal_result"
        ]["model_provider_response_sha256"],
        "model_completion_contract_hash": values["document"][
            "terminal_result"
        ]["model_completion_contract_hash"],
    }


@pytest.mark.parametrize("tamper", ["receipt", "hash", "identity"])
def test_durable_model_attempt_replay_rejects_receipt_hash_or_identity_tampering(
    tamper,
):
    values = _durable_model_attempt_fixture()
    if tamper == "receipt":
        values["document"]["provider_receipt"]["evidence_hash"] = _hash("f")
    elif tamper == "hash":
        values["document"]["terminal_result"][
            "model_provider_response_sha256"
        ] = _hash("f")
        values["row"]["terminal_result_hash"] = sha256_json(
            values["document"]["terminal_result"]
        )
    else:
        values["row"]["unit_ref"] = "forged-unit"

    with pytest.raises(
        RoutingExperimentTerminalRecoveryError,
        match="durable Model provider attempt",
    ):
        _resolve_durable_model_attempt(values)


def test_existing_model_attempt_replays_before_authorization_or_reservation():
    values = _durable_model_attempt_fixture()
    calls = []

    class _Config:
        def assert_live_enabled(self):
            calls.append("config")

    class _AttemptStore:
        def provider_attempt_row(self, key):
            calls.append(("attempt_lookup", key))
            return values["row"]

        def reserve_budget(self, **_values):
            raise AssertionError("durable replay must precede reservation")

    class _UnavailableCompiler:
        def prepare(self, **_values):
            raise AssertionError("durable replay must precede compilation")

    class _UnavailableAuthorization:
        def authorize_call(self, **_values):
            raise AssertionError("durable replay must precede authorization")

    runner = object.__new__(ReviewedProviderBrokerRoutingRunner)
    runner.config = _Config()
    runner.store = _AttemptStore()
    runner.artifact_lineage = values["lineage"]
    runner.artifact_lineages = {}
    runner.compiler = _UnavailableCompiler()
    runner.model_binding_requirements = object()
    runner.authorization_authority = _UnavailableAuthorization()
    runner.dispatch_authority = object()
    runner.execution_envelope = SimpleNamespace(
        envelope_hash=lambda: values["admission"].envelope_hash,
        experiment_hash=values["row"]["experiment_hash"],
    )
    runner.admission_bundle = values["admission"]
    runner.protected_release_receipt = values["document"][
        "protected_release_receipt"
    ]
    runner.authorization_parent_receipt_graphs = ()
    runner.dispatch_parent_receipt_graphs = ()
    runner.admission_validator = lambda *_values: None
    runner._execution = SimpleNamespace(
        experiment_hash=values["row"]["experiment_hash"],
        experiment_id=values["grant"].experiment_id,
        claim=object(),
        deadline_monotonic=0.0,
        deadline_supplier=None,
    )

    def replay_model_action(**kwargs):
        calls.append(("replay", kwargs["replay_ref"]))
        return {"replayed": True}

    runner.replay_model_action = replay_model_action

    assert runner._dispatch_call(
        values["grant"].binding,
        values["grant"].unit_ref,
        values["grant"].core_request_fingerprint,
        values["authorization"],
        model_action=values["action"],
    ) == {"replayed": True}
    assert [
        item if isinstance(item, str) else item[0] for item in calls
    ] == ["config", "attempt_lookup", "replay"]
