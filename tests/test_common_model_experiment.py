from __future__ import annotations

from dataclasses import replace

import pytest

from research_lab.common_model_runner_host import HostActionResult
from research_lab.model_runner_protocol import (
    ExactModelRunnerRegistration,
    ExactModelRunnerRegistry,
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
    _action_budget,
    _bind_durable_provider_result,
    _validate_variant_provider_binding,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    build_routing_model_completion_contract_v1,
)
from gateway.research_lab.routing_experiment_worker import (
    RoutingExperimentWorkerError,
    preflight_exact_model_unit,
)
from tests.model_runner_protocol_fixtures import (
    runner_declaration,
    runner_release_identity,
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
        **runner_release_identity("v2"),
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
        "schema_version": "model-runner-action:v1",
        "action_type": "execute_candidate_tool",
        "tool_id": "candidate.reviewed",
        "binding_contract_sha256": HASHES["binding"],
        "action_sha256": "4" * 64,
        "idempotency_key": "5" * 64,
        "max_response_bytes": 1_000_000,
    }


def _append_event_ack(value):
    document = {
        "schema_version": "leadpoet.research_lab.routing_event.v2",
        **value["event_doc"],
    }
    return {
        "event_hash": sha256_json(
            {
                "schema_version": "leadpoet.research_lab.routing_event.v2",
                "event_type": value["event_type"],
                "document": document,
            }
        ),
        "idempotent": False,
    }


class _Transport:
    def __init__(self, release=None):
        self.release = dict(_release() if release is None else release)

    def runner_protocol_generation(self, *, release_identity):
        assert release_identity == self.release
        return runner_declaration("v2")

    def build_raw_runner_input(self, *_args, **_values):
        raise AssertionError("v2 has no raw ICP entrypoint")

    def build_runner_start(self, *, member_name, **values):
        assert member_name == "build_runner_start"
        return {
            "schema_version": "model-runner-start:v2",
            "host_capability_manifest": values["host_capability_manifest"],
            "start": True,
        }

    def runner_preflight(self, *, execution_mode, member_name, **_values):
        assert member_name == "runner_preflight"
        release = self.release
        return {
            "schema_version": "model-runner-preflight:v2",
            "execution_mode": execution_mode,
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

    def validate_runner_preflight(
        self, value, *, member_name, **_values
    ):
        assert member_name == "validate_runner_preflight"
        return value

    def continue_runner(
        self,
        _start,
        *,
        continuation,
        completion,
        member_name,
        **_values,
    ):
        assert member_name == "continue_runner"
        if continuation is None:
            return {
                "status": "action_required",
                "action": _action(),
                "continuation": {
                    "schema_version": "model-runner-continuation:v2",
                    "pending": "4" * 64,
                },
            }
        assert completion["completion_sha256"] == "7" * 64
        return {
            "status": "completed",
            "action": None,
            "continuation": {
                "schema_version": "model-runner-continuation:v2",
                "terminal": True,
            },
            "result": {
                "schema_version": "model-runner-result:v2",
                "leads": [],
            },
            "model_receipt": {
                "schema_version": "model-runner-receipt:v2",
                "receipt_sha256": "8" * 64,
            },
        }

    def validate_runner_result(self, value, *, member_name, **_values):
        assert member_name == "validate_runner_result"
        return value

    def build_runner_completion(
        self, _action_value, result, *, member_name
    ):
        assert member_name == "build_runner_completion"
        return {
            "schema_version": "model-runner-completion:v2",
            "completion_sha256": "7" * 64,
            "provider_response": result["provider_response"],
        }

    def build_runner_provider_receipt_binding(self, *_args, **_values):
        raise AssertionError("v2 has no provider receipt binding entrypoint")


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


def _v3_registration():
    release = {
        **_release(),
        **runner_release_identity(
            "v3", contract_hash=HASHES["contract"]
        ),
        "source_commit": "2" * 40,
        "model_artifact_digest": "sha256:" + "3" * 64,
    }

    class _V3Transport(_Transport):
        def runner_protocol_generation(self, *, release_identity):
            assert release_identity == release
            return runner_declaration(
                "v3", contract_hash=HASHES["contract"]
            )

        def build_raw_runner_input(
            self, payload, *, source_schema, member_name
        ):
            assert member_name == "build_raw_runner_input"
            return {
                "kind": "raw_icp",
                "raw_icp": {
                    "schema_version": "model-raw-icp-envelope:v1",
                    "source_schema": source_schema,
                    "payload": dict(payload),
                },
            }

        def runner_preflight(
            self, *, execution_mode, member_name, **_values
        ):
            assert member_name == "runner_preflight"
            return {
                "schema_version": "model-runner-preflight:v3",
                "execution_mode": execution_mode,
            }

        def build_runner_start(self, *, member_name, **values):
            assert member_name == "build_runner_start"
            return {
                "schema_version": "model-runner-start:v3",
                "host_capability_manifest": values[
                    "host_capability_manifest"
                ],
            }

        def continue_runner(
            self,
            _start,
            *,
            continuation,
            member_name,
            **_values,
        ):
            assert member_name == "continue_runner"
            assert continuation is None
            return {
                "status": "action_required",
                "action": _normalization_action(),
                "continuation": {
                    "schema_version": "model-runner-continuation:v3",
                },
            }

        def build_runner_provider_receipt_binding(
            self, _action_value, result, *, member_name
        ):
            assert member_name == "build_runner_provider_receipt_binding"
            return {
                "schema_version": "model-provider-receipt-binding:v1",
                "provider_receipt_ref": result["provider_receipt_ref"],
                "provider_identity_sha256": result[
                    "provider_identity_sha256"
                ],
                "receipt_sha256": "8" * 64,
            }

    artifact = {
        **_artifact(),
        "commit_sha": "2" * 40,
        "model_artifact_hash": "sha256:" + "3" * 64,
    }
    return ExactModelRunnerRegistration(
        artifact_identity=artifact,
        protocol=ResearchLabModelRunnerProtocol(
            transport=_V3Transport(),
            expected_release_identity=release,
        ),
        host_capability_manifest={
            "manifest_sha256": HASHES["manifest"],
            "bindings": [
                {
                    "action_type": "normalize_icp",
                    "tool_id": "normalization.openrouter_json_schema",
                    "binding_contract_sha256": HASHES["binding"],
                    "response_schema_version": (
                        "model-icp-normalization-provider-response:v1"
                    ),
                    "available": True,
                }
            ],
        },
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

    assert main_registration.preflight(execution_mode="full_company")[
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

    def resolve_run_protocol_generation(self, **_identity):
        return _registration().protocol_generation.protocol_generation_sha256

    def append_model_transition(self, **value):
        idempotency_key = value["action"]["idempotency_key"]
        self.artifact_keys[idempotency_key] = value["artifact_key"]
        self.values[idempotency_key] = {
            "action": dict(value["action"]),
            "continuation": dict(value["continuation"]),
            "completion": dict(value["completion"]),
            "provider_receipt": dict(value["provider_receipt"]),
            "protocol_generation_sha256": value[
                "protocol_generation_sha256"
            ],
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


def test_old_v2_run_drains_under_its_pin_with_new_v3_registry_present():
    old_registration = _registration()
    new_registration = _v3_registration()
    registry = ExactModelRunnerRegistry(
        (old_registration, new_registration)
    )
    resolved = registry.resolve_identity(old_registration.artifact_identity)
    dispatcher = _Dispatcher()
    result = ExactModelExperimentCoordinator(
        experiment_hash="sha256:" + "d" * 64,
        registration=resolved,
        dispatcher=dispatcher,
        transitions=_Transitions(),
    ).run_unit(
        variant_id="baseline",
        unit_ref="unit-1",
        model_input={"kind": "normalized_icp", "normalized_icp": {}},
        execution_mode="full_company",
        target_count=1,
        evaluated_on="2026-08-20",
    )

    assert resolved is old_registration
    assert result.protocol_generation_sha256 == (
        old_registration.protocol_generation.protocol_generation_sha256
    )
    assert result.protocol_generation_sha256 != (
        new_registration.protocol_generation.protocol_generation_sha256
    )


def test_v3_raw_run_does_not_require_a_host_predicted_normalization_action(
    monkeypatch,
):
    registration = _v3_registration()
    generation_sha256 = (
        registration.protocol_generation.protocol_generation_sha256
    )
    terminal = {
        "status": "completed",
        "action": None,
        "continuation": {
            "schema_version": "model-runner-continuation:v3",
        },
        "result": {
            "schema_version": "model-runner-result:v3",
            "leads": [],
        },
        "model_receipt": {
            "schema_version": "model-runner-receipt:v3",
            "receipt_sha256": "8" * 64,
        },
    }
    monkeypatch.setattr(
        registration.protocol,
        "advance",
        lambda *_args, **_values: terminal,
    )
    monkeypatch.setattr(
        registration.protocol,
        "validate_result",
        lambda value, **_values: value,
    )
    monkeypatch.setattr(
        registration.protocol,
        "validate_normalization_action",
        lambda *_args, **_values: (_ for _ in ()).throw(
            AssertionError(
                "the host must not require normalization when the artifact "
                "selects a deterministic path"
            )
        ),
    )

    class _V3Transitions(_Transitions):
        def resolve_run_protocol_generation(self, **_identity):
            return generation_sha256

    dispatcher = _Dispatcher()
    result = ExactModelExperimentCoordinator(
        experiment_hash="sha256:" + "d" * 64,
        registration=registration,
        dispatcher=dispatcher,
        transitions=_V3Transitions(),
    ).run_unit(
        variant_id="baseline",
        unit_ref="unit-1",
        model_input={"kind": "raw_icp", "raw_icp": {}},
        execution_mode="full_company",
        target_count=1,
        evaluated_on="2026-08-23",
    )

    assert result.terminal_result == terminal
    assert result.protocol_generation_sha256 == generation_sha256
    assert dispatcher.calls == 0


@pytest.mark.parametrize("pin_kind", ["missing", "artifact", "generation"])
def test_run_registration_failure_happens_before_marker_or_oci(pin_kind):
    registration = _registration()

    class _Store:
        def append_event(self, **_values):
            raise AssertionError("run registration failure must not append")

        def load_model_transition_marker(self, **_values):
            raise AssertionError("run registration failure must precede marker")

        def load_exact_model_run_registration(self, **_values):
            if pin_kind == "missing":
                return None
            return {
                "schema_version": (
                    "leadpoet.research_lab.routing_worker_event.v2"
                ),
                "worker_ref": "worker-1",
                "runner_contract": (
                    "exact_model_runner_generation_pinned_v1"
                ),
                "artifact_keys": {
                    "baseline": (
                        registration.key
                        if pin_kind != "artifact"
                        else "0" * 40
                        + ":sha256:"
                        + "0" * 64
                        + ":sha256:"
                        + "0" * 64
                    )
                },
                "protocol_generations": {
                    "baseline": (
                        registration.protocol_generation.protocol_generation_sha256
                        if pin_kind != "generation"
                        else "sha256:" + "0" * 64
                    )
                },
            }

    if pin_kind == "missing":
        registration.protocol._transport.runner_protocol_generation = (
            lambda **_values: (_ for _ in ()).throw(
                AssertionError("missing pin must precede OCI")
            )
        )
    coordinator = ExactModelExperimentCoordinator(
        experiment_hash="sha256:" + "d" * 64,
        registration=registration,
        dispatcher=_Dispatcher(),
        transitions=FencedModelTransitionRepository(
            store=_Store(), claim=object()
        ),
    )
    with pytest.raises(CommonModelExperimentRecoveryError):
        coordinator.run_unit(
            variant_id="baseline",
            unit_ref="unit-1",
            model_input={"kind": "normalized_icp", "normalized_icp": {}},
            execution_mode="full_company",
            target_count=1,
            evaluated_on="2026-08-20",
        )


def _normalization_action() -> dict:
    return {
        "schema_version": "model-runner-action:v2",
        "action_phase": "normalization",
        "action_type": "normalize_icp",
        "stage": "icp_normalization",
        "tool_id": "normalization.openrouter_json_schema",
        "binding_contract_sha256": HASHES["binding"],
        "action_sha256": "4" * 64,
        "idempotency_key": "5" * 64,
        "request_fingerprint_sha256": "6" * 64,
        "max_response_bytes": 1_000_000,
        "response_schema_version": (
            "model-icp-normalization-provider-response:v1"
        ),
        "arguments": {
            "schema_version": "model-normalization-action-arguments:v1",
            "phase": "infer",
            "call_cap": 1,
            "credit_cap": 1.0,
            "timeout_seconds": 120.0,
            "provider_request": {"model": "fixture"},
        },
    }


def test_normalization_budget_uses_only_exact_artifact_owned_top_level_bounds():
    registration = _v3_registration()
    action = _normalization_action()
    identity = registration.protocol_generation.champion_execution[
        "normalization_action"
    ]

    registration.protocol.validate_normalization_action(
        action,
        host_capability_manifest=registration.host_capability_manifest,
    )
    assert _action_budget(
        action, normalization_identity=identity
    ) == (1_000_000, 120_000)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.pop("call_cap"),
        lambda values: values.__setitem__("call_cap", 2),
        lambda values: values.__setitem__("credit_cap", 1),
        lambda values: values.__setitem__("credit_cap", 0.5),
        lambda values: values.__setitem__("timeout_seconds", 120),
        lambda values: values.__setitem__("timeout_seconds", 60.0),
        lambda values: values.__setitem__("step", {}),
    ],
)
def test_normalization_budget_rejects_missing_tampered_or_host_substituted_bounds(
    mutate,
):
    registration = _v3_registration()
    action = _normalization_action()
    mutate(action["arguments"])
    identity = registration.protocol_generation.champion_execution[
        "normalization_action"
    ]

    with pytest.raises(ModelRunnerHostError, match="normalization action"):
        registration.protocol.validate_normalization_action(
            action,
            host_capability_manifest=registration.host_capability_manifest,
        )
    with pytest.raises(CommonModelExperimentError, match="normalization budget"):
        _action_budget(action, normalization_identity=identity)


def test_orchestration_budget_still_requires_the_model_step_document():
    with pytest.raises(CommonModelExperimentError, match="budget is missing"):
        _action_budget({"action_type": "execute_candidate_tool", "arguments": {}})


def test_v3_dataset_preflight_rejects_missing_or_mismatched_normalization_binding_before_claim():
    unit = {
        "execution_mode": "full_company",
        "target_count": 1,
        "evaluated_on": "2026-08-23",
        "raw_icp_source_schema": (
            "leadpoet-research-lab-benchmark-icp:v1"
        ),
        "raw_icp_payload": {"prompt": "fixture ICP"},
    }
    registration = _v3_registration()
    assert preflight_exact_model_unit(
        registration=registration,
        unit_input=unit,
    )["kind"] == "raw_icp"

    for bindings in (
        [],
        [
            {
                **registration.host_capability_manifest["bindings"][0],
                "binding_contract_sha256": "0" * 64,
            }
        ],
    ):
        unavailable = replace(
            registration,
            host_capability_manifest={
                **registration.host_capability_manifest,
                "bindings": bindings,
            },
        )
        with pytest.raises(
            ModelRunnerHostError,
            match="normalization binding differs",
        ):
            preflight_exact_model_unit(
                registration=unavailable,
                unit_input=unit,
            )


def test_v3_dataset_preflight_does_not_predict_that_raw_input_needs_normalization(
    monkeypatch,
):
    registration = _v3_registration()
    state = {
        "status": "action_required",
        "action": {
            "schema_version": "model-runner-action:v2",
            "action_phase": "orchestration",
            "action_type": "execute_candidate_tool",
            "stage": "company_discovery",
            "tool_id": "candidate.fixture",
        },
        "continuation": {
            "schema_version": "model-runner-continuation:v3",
        },
    }
    monkeypatch.setattr(
        registration.protocol,
        "advance",
        lambda *_args, **_values: state,
    )
    monkeypatch.setattr(
        registration.protocol,
        "validate_result",
        lambda value, **_values: value,
    )
    monkeypatch.setattr(
        registration.protocol,
        "validate_normalization_action",
        lambda *_args, **_values: (_ for _ in ()).throw(
            AssertionError("an orchestration action is not normalization")
        ),
    )

    result = preflight_exact_model_unit(
        registration=registration,
        unit_input={
            "execution_mode": "full_company",
            "target_count": 1,
            "evaluated_on": "2026-08-23",
            "raw_icp_source_schema": (
                "leadpoet-research-lab-benchmark-icp:v1"
            ),
            "raw_icp_payload": {"prompt": "fully explicit fixture ICP"},
        },
    )

    assert result["kind"] == "raw_icp"


def test_v3_dataset_preflight_rejects_invalid_mode_before_artifact_oci():
    registration = _v3_registration()
    registration.protocol._transport.runner_preflight = (
        lambda **_values: (_ for _ in ()).throw(
            AssertionError("invalid mode must precede OCI")
        )
    )
    with pytest.raises(
        RoutingExperimentWorkerError,
        match="execution mode is invalid",
    ):
        preflight_exact_model_unit(
            registration=registration,
            unit_input={
                "execution_mode": "host-substituted",
                "target_count": 1,
                "evaluated_on": "2026-08-23",
            },
        )


def _durable_v3_provider_result(
    *, host_receipt_sha256: str | None = None
) -> tuple[ResearchLabModelRunnerProtocol, dict, ProtectedModelActionResult]:
    registration = _v3_registration()
    receipt = ProviderReceipt(
        receipt_ref="provider_receipt:" + "a" * 16,
        binding_id="normalization",
        tool_id="normalization.openrouter_json_schema",
        binding_version="v1",
        source_lineage_id="normalization.openrouter",
        unit_ref="unit-1",
        request_fingerprint="sha256:" + "b" * 64,
        outcome="verified",
        evidence_hash="sha256:" + "c" * 64,
        credit_microunits=10,
        latency_ms=20,
        execution_mode="measured_lab",
        call_count=1,
    )
    protected = ProtectedModelActionResult(
        host_result=HostActionResult(
            outcome="succeeded",
            reason_code="protected_provider_verified",
            provider_response={"provider": "fixture"},
            calls=1,
            cost_credits=0.00001,
            latency_ms=20,
            provider_receipt_ref=receipt.receipt_ref,
            provider_receipt_sha256=host_receipt_sha256,
            provider_identity_sha256="7" * 64,
        ),
        provider_receipt=receipt,
    )
    return registration.protocol, _normalization_action(), protected


def test_artifact_owned_receipt_binding_is_read_back_before_completion():
    protocol, action, protected = _durable_v3_provider_result()

    bound = _bind_durable_provider_result(
        protocol=protocol,
        action=action,
        protected=protected,
    )

    assert bound.provider_receipt_ref == protected.provider_receipt.receipt_ref
    assert bound.provider_receipt_sha256 == "8" * 64
    assert bound.provider_identity_sha256 == "7" * 64


@pytest.mark.parametrize("tamper", ["receipt", "hash", "identity"])
def test_artifact_owned_receipt_binding_rejects_durable_custody_tampering(
    tamper,
):
    protocol, action, protected = _durable_v3_provider_result(
        host_receipt_sha256=("9" * 64 if tamper == "hash" else None)
    )
    if tamper == "receipt":
        protected = replace(
            protected,
            provider_receipt=replace(
                protected.provider_receipt,
                receipt_ref="provider_receipt:" + "f" * 16,
            ),
        )
    elif tamper == "identity":
        original = protocol._transport.build_runner_provider_receipt_binding

        def forged_identity(*args, **kwargs):
            return {
                **original(*args, **kwargs),
                "provider_identity_sha256": "6" * 64,
            }

        protocol._transport.build_runner_provider_receipt_binding = forged_identity

    with pytest.raises(CommonModelExperimentError, match="custody differs"):
        _bind_durable_provider_result(
            protocol=protocol,
            action=action,
            protected=protected,
        )


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
        binding=type("Binding", (), {"provider_id": "fixture"})(),
        protected=protected,
    )

    assert result.provider_receipt.call_count == 3
    assert result.host_result.calls == 3


def test_protected_model_result_rejects_missing_measured_call_count():
    action, protected = _protected_action_payload(call_count=None)

    with pytest.raises(CommonModelExperimentError, match="call count"):
        ReviewedProtectedModelActionDispatcher._protected_action_result(
            action=action,
            binding=type("Binding", (), {"provider_id": "fixture"})(),
            protected=protected,
        )


def test_fenced_restart_rejects_artifact_b_then_replays_artifact_a():
    class _Store:
        marker = None

        def load_exact_model_run_registration(self, **_identity):
            registration = _registration()
            return {
                "schema_version": (
                    "leadpoet.research_lab.routing_worker_event.v2"
                ),
                "worker_ref": "worker-1",
                "runner_contract": (
                    "exact_model_runner_generation_pinned_v1"
                ),
                "artifact_keys": {"baseline": registration.key},
                "protocol_generations": {
                    "baseline": registration.protocol_generation.protocol_generation_sha256
                },
            }

        def load_model_transition_marker(self, **_identity):
            return self.marker

        def append_event(self, **value):
            self.marker = {
                "schema_version": "leadpoet.research_lab.routing_event.v2",
                **value["event_doc"],
            }
            return _append_event_ack(value)

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


def test_fenced_restart_rejects_tampered_durable_completion_hash():
    class _Store:
        marker = None

        def load_exact_model_run_registration(self, **_identity):
            registration = _registration()
            return {
                "schema_version": (
                    "leadpoet.research_lab.routing_worker_event.v2"
                ),
                "worker_ref": "worker-1",
                "runner_contract": (
                    "exact_model_runner_generation_pinned_v1"
                ),
                "artifact_keys": {"baseline": registration.key},
                "protocol_generations": {
                    "baseline": registration.protocol_generation.protocol_generation_sha256
                },
            }

        def load_model_transition_marker(self, **_identity):
            return self.marker

        def append_event(self, **value):
            self.marker = {
                "schema_version": "leadpoet.research_lab.routing_event.v2",
                **value["event_doc"],
            }
            return _append_event_ack(value)

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
    ExactModelExperimentCoordinator(
        experiment_hash="sha256:" + "d" * 64,
        registration=_registration(),
        dispatcher=dispatcher,
        transitions=FencedModelTransitionRepository(
            store=store, claim=object()
        ),
    ).run_unit(**values)
    store.marker["completion_sha256"] = "9" * 64

    with pytest.raises(
        CommonModelExperimentError,
        match="replayed Model completion differs",
    ):
        ExactModelExperimentCoordinator(
            experiment_hash="sha256:" + "d" * 64,
            registration=_registration(),
            dispatcher=dispatcher,
            transitions=FencedModelTransitionRepository(
                store=store, claim=object()
            ),
        ).run_unit(**values)

    assert dispatcher.dispatch_calls == 1
    assert dispatcher.replay_calls == 1


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

        def load_exact_model_run_registration(self, **_identity):
            return None

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

        def load_exact_model_run_registration(self, **_identity):
            return None

        def append_event(self, **value):
            self.document = value["event_doc"]
            return _append_event_ack(value)

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
        protocol_generation_sha256="sha256:" + "f" * 64,
    )

    assert "provider_response" not in store.document
    assert "completion" not in store.document
    assert "continuation" not in store.document
    assert "provider-value" not in str(store.document)


def test_fenced_transition_repository_rejects_unconfirmed_durable_append():
    class _Store:
        def load_model_transition_marker(self, **_identity):
            return None

        def load_exact_model_run_registration(self, **_identity):
            return None

        def append_event(self, **_value):
            return {}

    repository = FencedModelTransitionRepository(
        store=_Store(), claim=object()
    )
    with pytest.raises(
        CommonModelExperimentError,
        match="durable Model transition result is invalid",
    ):
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
            protocol_generation_sha256=(
                _registration().protocol_generation.protocol_generation_sha256
            ),
        )


def test_stored_transition_action_substitution_fails_closed():
    generation_sha256 = (
        _registration().protocol_generation.protocol_generation_sha256
    )
    transitions = _Transitions()
    transitions.values[_action()["idempotency_key"]] = {
        "action": {**_action(), "tool_id": "candidate.forged"},
        "continuation": {"pending": "4" * 64},
        "completion": {"completion_sha256": "7" * 64},
        "provider_receipt": None,
        "protocol_generation_sha256": generation_sha256,
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
