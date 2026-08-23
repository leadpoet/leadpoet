from __future__ import annotations

from dataclasses import replace

import pytest

from research_lab.common_model_runner_host import HostActionResult
from research_lab.model_runner_protocol import (
    ExactModelRunnerRegistration,
    ModelRunnerHostError,
    ResearchLabModelRunnerProtocol,
)
from research_lab.routing_experiments import ProviderReceipt
from research_lab.canonical import sha256_json
from gateway.research_lab.common_model_experiment import (
    CommonModelExperimentError,
    CommonModelExperimentRecoveryError,
    ExactModelExperimentCoordinator,
    FencedModelTransitionRepository,
    MODEL_TRANSITION_SCHEMA_VERSION,
    ProtectedModelActionResult,
    ReviewedProtectedModelActionDispatcher,
    _validate_variant_provider_binding,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    build_routing_model_completion_contract_v1,
)


HASHES = {name: char * 64 for name, char in zip(
    ("artifact", "manifest", "contract", "catalog", "policy", "feature", "binding", "release"),
    "abcdef12",
)}


def _artifact():
    return {
        "repository": "leadpoet/Sourcing_model",
        "branch": "leadpoet-lab",
        "commit_sha": "1" * 40,
        "artifact_uri": "s3://reviewed/model.tar.gz",
        "model_artifact_hash": "sha256:" + HASHES["artifact"],
        "manifest_hash": "sha256:" + HASHES["manifest"],
        "routing_contract_hash": "sha256:" + HASHES["contract"],
        "routing_catalog_hash": "sha256:" + HASHES["catalog"],
        "routing_policy_hash": "sha256:" + HASHES["policy"],
        "feature_schema_hash": "sha256:" + HASHES["feature"],
        "verifier_contract_hash": "sha256:" + "9" * 64,
    }


def _release():
    return {
        "source_commit": "1" * 40,
        "model_artifact_digest": "sha256:" + HASHES["artifact"],
        "consumer_contract_sha256": HASHES["contract"],
        "catalog_sha256": HASHES["catalog"],
        "policy_sha256": HASHES["policy"],
        "candidate_profiles_sha256": "2" * 64,
        "intent_profiles_sha256": "3" * 64,
        "feature_schema_sha256": HASHES["feature"],
        "candidate_waterfall_contract_sha256": "4" * 64,
        "tool_binding_manifest_sha256": HASHES["binding"],
        "release_identity_sha256": HASHES["release"],
    }


def _action():
    return {
        "action_type": "execute_candidate_tool",
        "tool_id": "candidate.reviewed",
        "binding_contract_sha256": HASHES["binding"],
        "action_sha256": "4" * 64,
        "idempotency_key": "5" * 64,
        "max_response_bytes": 1_000_000,
    }


class _Transport:
    def __init__(self, release=None):
        self.release = dict(_release() if release is None else release)

    def build_runner_start(self, **values):
        return {"start": True, **values}

    def runner_preflight(self, **_values):
        release = self.release
        return {
            "preflight_sha256": "6" * 64,
            "release_identity_sha256": release["release_identity_sha256"],
            "source_commit": release["source_commit"],
            "consumer_contract_sha256": release["consumer_contract_sha256"],
            "catalog_sha256": release["catalog_sha256"],
            "policy_sha256": release["policy_sha256"],
            "candidate_profiles_sha256": release["candidate_profiles_sha256"],
            "intent_profiles_sha256": release["intent_profiles_sha256"],
            "feature_schema_sha256": release["feature_schema_sha256"],
            "host_capability_manifest_sha256": HASHES["manifest"],
            "binding_contracts_sha256": release["tool_binding_manifest_sha256"],
            "candidate_waterfall_contract_sha256": release[
                "candidate_waterfall_contract_sha256"
            ],
        }

    def continue_runner(self, _start, *, continuation, completion, **_values):
        if continuation is None:
            return {
                "status": "action_required",
                "action": _action(),
                "continuation": {"pending": "4" * 64},
            }
        assert completion["completion_sha256"] == "7" * 64
        return {
            "status": "completed",
            "action": None,
            "continuation": {"terminal": True},
            "result": {"leads": []},
            "model_receipt": {"receipt_sha256": "8" * 64},
        }

    def validate_runner_result(self, value, **_values):
        return value

    def build_runner_completion(self, _action_value, result):
        return {
            "completion_sha256": "7" * 64,
            "provider_response": result["provider_response"],
        }


def _registration(*, commit_char: str = "1"):
    release = {**_release(), "source_commit": commit_char * 40}
    artifact = {**_artifact(), "commit_sha": commit_char * 40}
    manifest = {
        "manifest_sha256": HASHES["manifest"],
        "bindings": [{
            "action_type": "execute_candidate_tool",
            "tool_id": "candidate.reviewed",
            "binding_contract_sha256": HASHES["binding"],
            "available": True,
        }]
    }
    protocol = ResearchLabModelRunnerProtocol(
        transport=_Transport(release), expected_release_identity=release
    )
    return ExactModelRunnerRegistration(
        artifact_identity=artifact,
        protocol=protocol,
        host_capability_manifest=manifest,
    )


def test_exact_registration_accepts_site_main_champion_identity():
    registration = _registration()
    main_registration = replace(
        registration,
        artifact_identity={
            **registration.artifact_identity,
            "branch": "main",
        },
    )

    assert main_registration.preflight()[
        "release_identity_sha256"
    ] == HASHES["release"]


class _Transitions:
    def __init__(self):
        self.values = {}
        self.artifact_keys = {}

    def load_model_transition(self, **identity):
        idempotency_key = identity["idempotency_key"]
        value = self.values.get(idempotency_key)
        if value is not None and self.artifact_keys.get(idempotency_key) != (
            identity["artifact_key"]
        ):
            raise CommonModelExperimentRecoveryError(
                "stored Model transition artifact identity differs"
            )
        return value

    def append_model_transition(self, **value):
        idempotency_key = value["action"]["idempotency_key"]
        self.artifact_keys[idempotency_key] = value["artifact_key"]
        self.values[idempotency_key] = {
            "action": dict(value["action"]),
            "continuation": dict(value["continuation"]),
            "completion": dict(value["completion"]),
            "provider_receipt": dict(value["provider_receipt"]),
        }


class _Dispatcher:
    calls = 0

    def dispatch_provider_action(self, *, action, unit_ref, **_values):
        self.calls += 1
        receipt = ProviderReceipt(
            receipt_ref="provider_receipt:" + "a" * 16,
            binding_id="reviewed-binding",
            tool_id=action["tool_id"],
            binding_version="v1",
            source_lineage_id="reviewed-source",
            unit_ref=unit_ref,
            request_fingerprint="sha256:" + "b" * 64,
            outcome="verified",
            evidence_hash="sha256:" + "c" * 64,
            credit_microunits=10,
            latency_ms=20,
            execution_mode="measured_lab",
            call_count=1,
        )
        return ProtectedModelActionResult(
            host_result=HostActionResult(
                outcome="succeeded",
                reason_code="reviewed_provider",
                provider_response={"provider": "fixture"},
                calls=1,
                cost_credits=0.00001,
                latency_ms=20,
                provider_receipt_ref=receipt.receipt_ref,
            ),
            provider_receipt=receipt,
        )

    def verify_company_action(self, **_values):
        raise AssertionError("not called")

    def verify_intent_action(self, **_values):
        raise AssertionError("not called")

    def verify_contact_action(self, **_values):
        raise AssertionError("not called")


def test_exact_coordinator_replays_persisted_completion_without_paid_call():
    transitions = _Transitions()
    dispatcher = _Dispatcher()
    coordinator = ExactModelExperimentCoordinator(
        experiment_hash="sha256:" + "d" * 64,
        registration=_registration(),
        dispatcher=dispatcher,
        transitions=transitions,
    )
    values = dict(
        variant_id="baseline",
        unit_ref="unit-1",
        model_input={"kind": "normalized_icp", "normalized_icp": {}},
        execution_mode="full_company",
        target_count=1,
        evaluated_on="2026-08-20",
    )
    first = coordinator.run_unit(**values)
    second = coordinator.run_unit(**values)

    assert first.terminal_result == second.terminal_result
    assert dispatcher.calls == 1
    assert second.replayed_transition_count == 1


def _protected_action_payload(*, call_count):
    response = {"provider": "fixture"}
    receipt = ProviderReceipt(
        receipt_ref="provider_receipt:" + "a" * 16,
        binding_id="reviewed-binding",
        tool_id="candidate.reviewed",
        binding_version="v1",
        source_lineage_id="reviewed-source",
        unit_ref="unit-1",
        request_fingerprint="sha256:" + "b" * 64,
        outcome="verified",
        evidence_hash="sha256:" + "c" * 64,
        credit_microunits=10,
        latency_ms=20,
        execution_mode="measured_lab",
        call_count=call_count,
    )
    identity = receipt.to_dict()
    identity.pop("receipt_ref")
    receipt = replace(
        receipt,
        receipt_ref="provider_receipt:" + sha256_json(identity).split(":", 1)[1][:16],
    )
    action = {
        **_action(),
        "response_schema_version": "model-provider-response:v1",
    }
    return action, {
        "provider_receipt": receipt.to_dict(),
        "model_provider_response": response,
        "model_provider_response_sha256": sha256_json(response),
        "model_completion_contract_hash": sha256_json(
            build_routing_model_completion_contract_v1(action)
        ),
        "protected_dispatch_job_id": "dispatch-1",
        "terminal_receipt_hash": "sha256:" + "d" * 64,
    }


def test_protected_model_result_uses_authoritative_multi_call_receipt():
    action, protected = _protected_action_payload(call_count=3)

    result = ReviewedProtectedModelActionDispatcher._protected_action_result(
        action=action,
        protected=protected,
    )

    assert result.provider_receipt.call_count == 3
    assert result.host_result.calls == 3


def test_protected_model_result_rejects_missing_measured_call_count():
    action, protected = _protected_action_payload(call_count=None)

    with pytest.raises(CommonModelExperimentError, match="call count"):
        ReviewedProtectedModelActionDispatcher._protected_action_result(
            action=action,
            protected=protected,
        )


def test_fenced_restart_rejects_artifact_b_then_replays_artifact_a():
    class _Store:
        marker = None

        def load_model_transition_marker(self, **_identity):
            return self.marker

        def append_event(self, **value):
            self.marker = {
                "schema_version": "leadpoet.research_lab.routing_event.v2",
                **value["event_doc"],
            }
            return {"inserted": True}

    class _ReplayDispatcher(_Dispatcher):
        dispatch_calls = 0
        replay_calls = 0

        @staticmethod
        def _result(action, unit_ref):
            result = _Dispatcher().dispatch_provider_action(
                action=action, unit_ref=unit_ref
            )
            response = result.host_result.provider_response
            return ProtectedModelActionResult(
                host_result=result.host_result,
                provider_receipt=result.provider_receipt,
                replay_ref={
                    "schema_version": (
                        "leadpoet.research_lab.protected_model_replay_ref.v1"
                    ),
                    "protected_dispatch_job_id": "routing-dispatch:" + "1" * 32,
                    "terminal_receipt_hash": "sha256:" + "2" * 64,
                    "model_provider_response_sha256": sha256_json(response),
                    "model_completion_contract_hash": "sha256:" + "3" * 64,
                },
            )

        def dispatch_provider_action(self, *, action, unit_ref, **_values):
            self.dispatch_calls += 1
            return self._result(action, unit_ref)

        def replay_provider_action(self, *, action, unit_ref, **_values):
            self.replay_calls += 1
            return self._result(action, unit_ref)

    store = _Store()
    dispatcher = _ReplayDispatcher()
    values = dict(
        variant_id="baseline",
        unit_ref="unit-1",
        model_input={"kind": "normalized_icp", "normalized_icp": {}},
        execution_mode="full_company",
        target_count=1,
        evaluated_on="2026-08-20",
    )
    def coordinator(registration):
        return ExactModelExperimentCoordinator(
            experiment_hash="sha256:" + "d" * 64,
            registration=registration,
            dispatcher=dispatcher,
            transitions=FencedModelTransitionRepository(
                store=store, claim=object()
            ),
        )

    registration_a = _registration()
    result = coordinator(registration_a).run_unit(**values)
    assert store.marker["artifact_key"] == registration_a.key
    assert store.marker["event_schema_version"] == (
        MODEL_TRANSITION_SCHEMA_VERSION
    )

    with pytest.raises(
        CommonModelExperimentRecoveryError,
        match="artifact identity differs",
    ):
        coordinator(_registration(commit_char="9")).run_unit(**values)
    assert dispatcher.dispatch_calls == 1
    assert dispatcher.replay_calls == 0

    result = coordinator(registration_a).run_unit(**values)

    assert dispatcher.dispatch_calls == 1
    assert dispatcher.replay_calls == 1
    assert result.replayed_transition_count == 1


def test_fenced_repository_rejects_legacy_identityless_v1_marker():
    class _Store:
        def append_event(self, **_value):
            raise AssertionError("legacy marker must fail before append")

        def load_model_transition_marker(self, **_identity):
            return {
                "schema_version": "leadpoet.research_lab.routing_event.v2",
                "event_schema_version": (
                    "leadpoet.research_lab.model_transition.v1"
                ),
            }

    registration = _registration()
    repository = FencedModelTransitionRepository(
        store=_Store(), claim=object()
    )
    with pytest.raises(
        CommonModelExperimentRecoveryError,
        match="artifact identity differs",
    ):
        repository.load_model_transition(
            experiment_hash="sha256:" + "d" * 64,
            variant_id="baseline",
            unit_ref="unit-1",
            idempotency_key=_action()["idempotency_key"],
            artifact_key=registration.key,
        )


def test_fenced_transition_repository_persists_hashes_not_provider_body():
    class _Store:
        def __init__(self):
            self.document = None

        def load_model_transition_marker(self, **_identity):
            return None

        def append_event(self, **value):
            self.document = value["event_doc"]
            return {"inserted": True}

    store = _Store()
    repository = FencedModelTransitionRepository(store=store, claim=object())
    repository.append_model_transition(
        experiment_hash="sha256:" + "d" * 64,
        variant_id="baseline",
        unit_ref="unit-1",
        artifact_key=_registration().key,
        action=_action(),
        continuation={"private": "continuation"},
        completion={
            "completion_sha256": "7" * 64,
            "provider_response": {"private": "provider-value"},
        },
        provider_receipt=None,
    )

    assert "provider_response" not in store.document
    assert "completion" not in store.document
    assert "continuation" not in store.document
    assert "provider-value" not in str(store.document)


def test_stored_transition_action_substitution_fails_closed():
    transitions = _Transitions()
    transitions.values[_action()["idempotency_key"]] = {
        "action": {**_action(), "tool_id": "candidate.forged"},
        "continuation": {"pending": "4" * 64},
        "completion": {"completion_sha256": "7" * 64},
        "provider_receipt": None,
    }
    transitions.artifact_keys[_action()["idempotency_key"]] = (
        _registration().key
    )
    coordinator = ExactModelExperimentCoordinator(
        experiment_hash="sha256:" + "d" * 64,
        registration=_registration(),
        dispatcher=_Dispatcher(),
        transitions=transitions,
    )
    try:
        coordinator.run_unit(
            variant_id="baseline",
            unit_ref="unit-1",
            model_input={"kind": "normalized_icp", "normalized_icp": {}},
            execution_mode="full_company",
            target_count=1,
            evaluated_on="2026-08-20",
        )
    except CommonModelExperimentError as exc:
        assert "action differs" in str(exc)
    else:
        raise AssertionError("forged stored action must fail closed")


def test_exact_variant_payload_is_identity_only_and_tamper_evident():
    registration = _registration()
    payload = registration.variant_audit_payload()

    registration.validate_variant_audit_payload(payload)
    assert set(payload) == {"schema_version", "artifact_key"}

    for forged in (
        {**payload, "provider_order": ["candidate.forged"]},
        {**payload, "artifact_key": payload["artifact_key"] + "0"},
    ):
        try:
            registration.validate_variant_audit_payload(forged)
        except ModelRunnerHostError as exc:
            assert "exact Model artifact identity" in str(exc)
        else:
            raise AssertionError("forged variant payload must fail closed")


def test_variant_provider_binding_is_checked_against_its_own_manifest():
    registration = _registration()
    _validate_variant_provider_binding(
        registration=registration,
        action=_action(),
        provider_binding=type(
            "Binding",
            (),
            {
                "binding_id": "reviewed-binding",
                "execution_contract_hash": "sha256:" + HASHES["binding"],
            },
        )(),
        allowed_binding_ids=("reviewed-binding",),
    )

    forged_registration = replace(
        registration,
        host_capability_manifest={
            **registration.host_capability_manifest,
            "bindings": [],
        },
    )
    try:
        _validate_variant_provider_binding(
            registration=forged_registration,
            action=_action(),
            provider_binding=type(
                "Binding",
                (),
                {
                    "binding_id": "reviewed-binding",
                    "execution_contract_hash": "sha256:" + HASHES["binding"],
                },
            )(),
            allowed_binding_ids=("reviewed-binding",),
        )
    except CommonModelExperimentError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("variant binding substitution must fail closed")


def test_variant_provider_binding_must_be_declared_before_dispatch():
    registration = _registration()
    try:
        _validate_variant_provider_binding(
            registration=registration,
            action=_action(),
            provider_binding=type(
                "Binding",
                (),
                {
                    "binding_id": "manifest-authorized-but-undeclared",
                    "execution_contract_hash": "sha256:" + HASHES["binding"],
                },
            )(),
            allowed_binding_ids=("declared-only",),
        )
    except CommonModelExperimentError as exc:
        assert "did not declare" in str(exc)
    else:
        raise AssertionError(
            "a globally available binding excluded from the variant must fail closed"
        )
