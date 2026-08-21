from __future__ import annotations

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
