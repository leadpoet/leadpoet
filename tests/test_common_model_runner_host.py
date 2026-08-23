from __future__ import annotations

from pathlib import Path

import pytest

from research_lab.common_model_runner_host import (
    CommonModelRunnerHost,
    HostActionBinding,
    HostActionResult,
    ModelRunnerHostError,
)
import research_lab.champion_model_runner as champion_model_runner
from research_lab.deepline_model_runner_bindings import (
    DEEPLINE_RUNNER_REQUEST_SCHEMA_VERSION,
    DeeplineCallReceipt,
    build_deepline_runner_binding,
)
from research_lab.model_runner_protocol import ResearchLabModelRunnerProtocol
from research_lab.docker_model_runner_transport import (
    DockerModelRunnerTransport,
)
from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
)


CONTRACT_HASH = "a" * 64


def _start_request():
    return {
        "host_capability_manifest": {
            "bindings": [{
                "action_type": "execute_intent_tool",
                "tool_id": "intent.source_add.predictleads_financing",
                "binding_contract_sha256": CONTRACT_HASH,
                "available": True,
            }]
        }
    }


def _action():
    return {
        "action_type": "execute_intent_tool",
        "tool_id": "intent.source_add.predictleads_financing",
        "binding_contract_sha256": CONTRACT_HASH,
        "action_sha256": "b" * 64,
        "idempotency_key": "c" * 64,
        "arguments": {"candidate": {"official_domain": "acme.test"}},
    }


class FakeArtifactTransport:
    def __init__(self):
        self.completion_inputs = []

    def continue_runner(
        self,
        _start,
        *,
        expected_release_identity,
        continuation,
        completion,
    ):
        assert expected_release_identity == {"release": "exact"}
        if continuation is None:
            return {
                "status": "action_required",
                "action": _action(),
                "continuation": {"step": 1},
            }
        assert completion["completion_sha256"] == "e" * 64
        return {
            "status": "completed",
            "continuation": {"step": 2},
            "result": {"leads": []},
            "model_receipt": {"receipt_sha256": "f" * 64},
        }

    def build_runner_start(self, **_values):
        return {"start": True}

    def build_runner_completion(self, _action_value, result):
        self.completion_inputs.append(result)
        return {
            "outcome": result["outcome"],
            "calls": result["calls"],
            "cost_credits": result["cost_credits"],
            "latency_ms": result["latency_ms"],
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_response_sha256": "d" * 64,
            "completion_sha256": "e" * 64,
        }

    def runner_preflight(self, **_values):
        return {"preflight": True}

    def validate_runner_result(self, value, **_values):
        return value


class FakeDeeplineClient:
    def __init__(self, *, cost=0.56):
        self.cost = cost
        self.requests = []

    def execute_tool(self, **request):
        self.requests.append(request)
        return DeeplineCallReceipt(
            status_code=200,
            body={
                "run": {"status": "completed"},
                "outputs": {"model_provider_records": []},
            },
            calls=1,
            cost_credits=self.cost,
            latency_ms=20,
            provider_request_id="private-provider-id",
            provider_receipt_ref="provider_receipt:" + "a" * 16,
        )


def test_lab_dispatches_deepline_through_the_artifact_protocol():
    artifact = FakeArtifactTransport()
    protocol = ResearchLabModelRunnerProtocol(
        transport=artifact,
        expected_release_identity={"release": "exact"},
    )
    client = FakeDeeplineClient()
    binding = build_deepline_runner_binding(
        model_tool_id="intent.source_add.predictleads_financing",
        binding_contract_sha256=CONTRACT_HASH,
        client=client,
    )
    persisted = []
    result = CommonModelRunnerHost(
        consumer_id="research-lab-champion",
        protocol=protocol,
        bindings=[binding],
        persist_transition=lambda **value: persisted.append(value),
    ).run(_start_request())

    assert result["status"] == "completed"
    assert client.requests[0]["tool_id"] == (
        "predictleads_company_financing_events"
    )
    assert client.requests[0]["payload"]["schema_version"] == (
        DEEPLINE_RUNNER_REQUEST_SCHEMA_VERSION
    )
    assert artifact.completion_inputs[0]["provider_response"]["provider"] == (
        "deepline"
    )
    assert artifact.completion_inputs[0]["provider_receipt_ref"] == (
        "provider_receipt:" + "a" * 16
    )
    assert "private-provider-id" not in str(persisted[0]["host_receipt"])


def test_deepline_binding_enforces_reviewed_cost_limit():
    binding = build_deepline_runner_binding(
        model_tool_id="intent.source_add.predictleads_financing",
        binding_contract_sha256=CONTRACT_HASH,
        client=FakeDeeplineClient(cost=0.57),
    )
    assert not hasattr(binding, "execute")
    try:
        binding.dispatch(_action())
    except ModelRunnerHostError as exc:
        assert "cost limit" in str(exc)
    else:
        raise AssertionError("over-budget Deepline call must fail closed")


def test_artifact_protocol_rejects_incomplete_transport():
    class IncompleteTransport:
        def continue_runner(self, *_args, **_kwargs):
            return {}

    try:
        ResearchLabModelRunnerProtocol(
            transport=IncompleteTransport(),
            expected_release_identity={"release": "exact"},
        )
    except ModelRunnerHostError as exc:
        assert "build_runner_completion" in str(exc)
    else:
        raise AssertionError("incomplete artifact transport must fail closed")


def test_artifact_protocol_rejects_invalid_action_state_before_host_dispatch():
    class InvalidStateTransport(FakeArtifactTransport):
        def continue_runner(self, *args, **kwargs):
            return {
                "status": "action_required",
                "action": {"action_type": "execute_intent_tool"},
                "continuation": {"step": 1},
            }

    protocol = ResearchLabModelRunnerProtocol(
        transport=InvalidStateTransport(),
        expected_release_identity={"release": "exact"},
    )
    try:
        protocol.advance(
            _start_request(), continuation=None, completion=None
        )
    except ModelRunnerHostError as exc:
        assert "action identity" in str(exc)
    else:
        raise AssertionError("invalid action state must fail closed")


def test_artifact_protocol_rejects_invalid_terminal_result():
    class InvalidResultTransport(FakeArtifactTransport):
        def validate_runner_result(self, _value, **_kwargs):
            return {"status": "completed", "result": {}}

    protocol = ResearchLabModelRunnerProtocol(
        transport=InvalidResultTransport(),
        expected_release_identity={"release": "exact"},
    )
    try:
        protocol.validate_result(
            {"status": "completed", "result": {}},
            start_request=_start_request(),
        )
    except ModelRunnerHostError as exc:
        assert "terminal result" in str(exc)
    else:
        raise AssertionError("invalid terminal result must fail closed")


def test_champion_preflights_and_validates_the_complete_protocol(monkeypatch):
    calls = []

    class FakeProtocol:
        def __init__(self, *, transport, expected_release_identity):
            calls.append(("protocol", transport, expected_release_identity))

        def preflight(self, *, host_capability_manifest):
            calls.append(("preflight", host_capability_manifest))
            return {"preflight": True}

        def build_start(self, **values):
            calls.append(("build_start", values))
            return {"start": True}

        def validate_result(self, value, *, start_request):
            calls.append(("validate", value, start_request))
            return {"validated": True}

    class FakeHost:
        def __init__(self, **values):
            calls.append(("host", values))

        def run(self, start_request, *, continuation):
            calls.append(("run", start_request, continuation))
            return {"status": "completed"}

    transport = object()
    monkeypatch.setattr(
        champion_model_runner,
        "DockerModelRunnerTransport",
        lambda _runner: transport,
    )
    monkeypatch.setattr(
        champion_model_runner,
        "ResearchLabModelRunnerProtocol",
        FakeProtocol,
    )
    monkeypatch.setattr(champion_model_runner, "CommonModelRunnerHost", FakeHost)

    result = champion_model_runner.run_common_champion(
        runner=object(),
        input={"icp": "test"},
        execution_mode="full_company",
        target_count=1,
        evaluated_on="2026-08-21",
        host_capability_manifest={"bindings": []},
        release_identity={"source_commit": "a" * 40},
        bindings=(),
        persist_transition=lambda **_values: None,
    )

    assert result == {"validated": True}
    assert [item[0] for item in calls] == [
        "protocol",
        "preflight",
        "build_start",
        "host",
        "run",
        "validate",
    ]


def test_registered_model_unit_is_the_live_lab_protocol_caller():
    from research_lab.champion_model_runner import run_registered_model_unit

    registration = _registration_for_live_caller()
    persisted = []
    loaded = []
    result = run_registered_model_unit(
        registration=registration,
        input={"icp": "test"},
        execution_mode="full_company",
        target_count=1,
        evaluated_on="2026-08-21",
        bindings=[
            HostActionBinding(
                action_type="execute_candidate_tool",
                tool_id="candidate.reviewed",
                binding_contract_sha256=CONTRACT_HASH,
                dispatch=lambda _action: HostActionResult(
                    outcome="succeeded",
                    reason_code="fixture",
                    provider_response={"records": []},
                    calls=1,
                    cost_credits=0,
                    latency_ms=1,
                ),
            )
        ],
        persist_transition=lambda **value: persisted.append(value),
        load_completion=lambda **value: loaded.append(value) or None,
    )
    assert result["status"] == "completed"
    assert len(persisted) == 1
    assert persisted[0]["artifact_key"] == registration.key
    assert loaded == [{
        "artifact_key": registration.key,
        "idempotency_key": "c" * 64,
    }]


def test_registered_model_continuation_rejects_b_and_a_can_drain():
    from research_lab.champion_model_runner import run_registered_model_unit

    registration_a = _registration_for_live_caller()
    binding_calls = []
    binding = HostActionBinding(
        action_type="execute_candidate_tool",
        tool_id="candidate.reviewed",
        binding_contract_sha256=CONTRACT_HASH,
        dispatch=lambda action: binding_calls.append(dict(action)) or (
            HostActionResult(
                outcome="succeeded",
                reason_code="fixture",
                provider_response={"records": []},
                calls=1,
                cost_credits=0,
                latency_ms=1,
            )
        ),
    )
    persisted = []

    class SimulatedPostCommitCrash(RuntimeError):
        pass

    def persist_then_crash(**value):
        persisted.append(value)
        raise SimulatedPostCommitCrash

    with pytest.raises(SimulatedPostCommitCrash):
        run_registered_model_unit(
            registration=registration_a,
            input={"icp": "test"},
            execution_mode="full_company",
            target_count=1,
            evaluated_on="2026-08-21",
            bindings=[binding],
            persist_transition=persist_then_crash,
        )

    assert len(binding_calls) == 1
    durable = persisted[0]
    registration_b = _registration_for_live_caller(commit_char="9")
    with pytest.raises(ModelRunnerHostError, match="artifact identity differs"):
        run_registered_model_unit(
            registration=registration_b,
            input={"icp": "test"},
            execution_mode="full_company",
            target_count=1,
            evaluated_on="2026-08-21",
            bindings=[binding],
            persist_transition=lambda **_value: None,
            continuation=durable["continuation"],
            continuation_artifact_key=durable["artifact_key"],
        )
    assert len(binding_calls) == 1

    result = run_registered_model_unit(
        registration=registration_a,
        input={"icp": "test"},
        execution_mode="full_company",
        target_count=1,
        evaluated_on="2026-08-21",
        bindings=[binding],
        persist_transition=lambda **_value: None,
        continuation=durable["continuation"],
        continuation_artifact_key=durable["artifact_key"],
    )
    assert result["status"] == "completed"
    assert len(binding_calls) == 1


def test_model_runner_hosts_are_default_off_with_no_production_caller():
    from gateway.research_lab.routing_execution_consumer import (
        REVIEWED_ROUTING_FACTORY_REGISTRY,
        RoutingExecutionConsumerConfig,
    )

    assert RoutingExecutionConsumerConfig.from_env({}).enabled is False
    assert not REVIEWED_ROUTING_FACTORY_REGISTRY

    root = Path(__file__).resolve().parents[1]
    callers = []
    entrypoints = (
        "run_common_champion(",
        "run_registered_model_unit(",
    )
    for package in (root / "gateway", root / "research_lab"):
        for path in package.rglob("*.py"):
            if path == root / "research_lab" / "champion_model_runner.py":
                continue
            source = path.read_text(encoding="utf-8")
            for entrypoint in entrypoints:
                if entrypoint in source:
                    callers.append(
                        (path.relative_to(root).as_posix(), entrypoint)
                    )
    assert callers == []


def _registration_for_live_caller(*, commit_char: str = "1"):
    from research_lab.model_runner_protocol import ExactModelRunnerRegistration

    release = {
        "source_commit": commit_char * 40,
        "model_artifact_digest": "sha256:" + "a" * 64,
        "consumer_contract_sha256": "c" * 64,
        "catalog_sha256": "d" * 64,
        "policy_sha256": "e" * 64,
        "candidate_profiles_sha256": "2" * 64,
        "intent_profiles_sha256": "3" * 64,
        "feature_schema_sha256": "f" * 64,
        "candidate_waterfall_contract_sha256": "4" * 64,
        "tool_binding_manifest_sha256": "5" * 64,
        "release_identity_sha256": "6" * 64,
    }
    live_action = {
        "action_type": "execute_candidate_tool",
        "tool_id": "candidate.reviewed",
        "binding_contract_sha256": CONTRACT_HASH,
        "action_sha256": "b" * 64,
        "idempotency_key": "c" * 64,
    }

    class _Transport(FakeArtifactTransport):
        def build_runner_start(self, **values):
            return {"start": True, **values}

        def runner_preflight(self, **_values):
            return {
                "release_identity_sha256": release["release_identity_sha256"],
                "source_commit": release["source_commit"],
                "consumer_contract_sha256": release["consumer_contract_sha256"],
                "catalog_sha256": release["catalog_sha256"],
                "policy_sha256": release["policy_sha256"],
                "candidate_profiles_sha256": release["candidate_profiles_sha256"],
                "intent_profiles_sha256": release["intent_profiles_sha256"],
                "feature_schema_sha256": release["feature_schema_sha256"],
                "host_capability_manifest_sha256": "b" * 64,
                "binding_contracts_sha256": release["tool_binding_manifest_sha256"],
                "candidate_waterfall_contract_sha256": release[
                    "candidate_waterfall_contract_sha256"
                ],
            }

        def continue_runner(
            self,
            _start,
            *,
            expected_release_identity,
            continuation,
            completion,
        ):
            assert expected_release_identity == release
            if continuation is None:
                return {
                    "status": "action_required",
                    "action": live_action,
                    "continuation": {"step": 1},
                }
            if continuation == {"step": 2} and completion is None:
                return {
                    "status": "completed",
                    "continuation": {"step": 2},
                    "result": {"leads": []},
                    "model_receipt": {"receipt_sha256": "f" * 64},
                }
            assert completion["completion_sha256"] == "e" * 64
            return {
                "status": "completed",
                "continuation": {"step": 2},
                "result": {"leads": []},
                "model_receipt": {"receipt_sha256": "f" * 64},
            }

    protocol = ResearchLabModelRunnerProtocol(
        transport=_Transport(), expected_release_identity=release
    )
    return ExactModelRunnerRegistration(
        artifact_identity={
            "repository": "leadpoet/Sourcing_model",
            "branch": "main",
            "commit_sha": commit_char * 40,
            "model_artifact_hash": "sha256:" + "a" * 64,
            "manifest_hash": "sha256:" + "b" * 64,
            "routing_contract_hash": "sha256:" + "c" * 64,
            "routing_catalog_hash": "sha256:" + "d" * 64,
            "routing_policy_hash": "sha256:" + "e" * 64,
            "feature_schema_hash": "sha256:" + "f" * 64,
        },
        protocol=protocol,
        host_capability_manifest={
            "manifest_sha256": "b" * 64,
            "bindings": [{
                "action_type": "execute_candidate_tool",
                "tool_id": "candidate.reviewed",
                "binding_contract_sha256": CONTRACT_HASH,
                "available": True,
            }],
        },
    )


def test_oci_runner_transport_does_not_forward_provider_credentials(
    monkeypatch,
):
    source = object.__new__(DockerPrivateModelRunner)
    source.spec = DockerPrivateModelSpec(
        image_digest="model@sha256:" + "1" * 64,
        env_passthrough=("DEEPLINE_API_KEY", "EXA_API_KEY"),
        extra_env={"DEEPLINE_API_KEY": "must-not-cross"},
        pull_before_run=False,
    )

    def capture_spec(instance, spec):
        instance.spec = spec

    monkeypatch.setattr(DockerPrivateModelRunner, "__init__", capture_spec)
    transport = DockerModelRunnerTransport(source)

    assert transport._runner.spec.env_passthrough == ()
    assert transport._runner.spec.extra_env == {}
    assert transport._runner.spec.pull_before_run is False


def test_oci_runner_transport_exposes_the_complete_artifact_protocol(monkeypatch):
    transport = object.__new__(DockerModelRunnerTransport)
    calls = []

    def capture(operation, payload):
        calls.append((operation, payload))
        return {"operation": operation}

    monkeypatch.setattr(transport, "_call", capture)
    assert transport.runner_preflight(
        host_capability_manifest={"bindings": []},
        release_identity={"source_commit": "a" * 40},
    ) == {"operation": "runner_preflight"}
    assert transport.validate_runner_result(
        {"status": "completed"},
        start_request={"input": {}},
        expected_release_identity={"source_commit": "a" * 40},
    ) == {"operation": "validate_runner_result"}
    assert calls == [
        (
            "runner_preflight",
            {
                "host_capability_manifest": {"bindings": []},
                "release_identity": {"source_commit": "a" * 40},
            },
        ),
        (
            "validate_runner_result",
            {
                "value": {"status": "completed"},
                "start_request": {"input": {}},
                "expected_release_identity": {"source_commit": "a" * 40},
            },
        ),
    ]
