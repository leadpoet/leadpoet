from __future__ import annotations

from copy import deepcopy
import hashlib
from io import StringIO
import json
import sys

import pytest

from research_lab.common_model_runner_host import (
    CommonModelRunnerHost,
    HostActionBinding,
    HostActionResult,
    ModelRunnerHostError,
    ProviderReceiptCustodyRecord,
)
import research_lab.champion_model_runner as champion_model_runner
from research_lab.deepline_model_runner_bindings import (
    DEEPLINE_RUNNER_REQUEST_SCHEMA_VERSION,
    DeeplineCallReceipt,
    build_deepline_runner_binding,
)
from research_lab.model_runner_protocol import (
    ArtifactRunnerProtocolGeneration,
    ResearchLabModelRunnerProtocol,
)
from research_lab.docker_model_runner_transport import (
    _COMMON_RUNNER_BOOTSTRAP,
    DockerModelRunnerTransport,
)
from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
)
from tests.model_runner_protocol_fixtures import (
    runner_declaration,
    runner_release_identity,
)


CONTRACT_HASH = "a" * 64
PROVIDER_RECEIPT_REF = "provider_receipt:" + "a" * 16
PROVIDER_RECEIPT_SHA256 = "7" * 64
PROVIDER_IDENTITY_SHA256 = hashlib.sha256(b"deepline").hexdigest()
RELEASE = runner_release_identity("v3", release="exact")


class DurableCustody:
    durable = True

    def __init__(self, record=None):
        self.record = record or ProviderReceiptCustodyRecord(
            provider_receipt_ref=PROVIDER_RECEIPT_REF,
            provider_receipt_sha256=PROVIDER_RECEIPT_SHA256,
            provider_identity_sha256=PROVIDER_IDENTITY_SHA256,
        )
        self.calls = []

    def resolve_provider_receipt(self, provider_receipt_ref):
        self.calls.append(provider_receipt_ref)
        return self.record


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
        "schema_version": "model-runner-action:v2",
        "action_phase": "orchestration",
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

    def runner_protocol_generation(self, *, release_identity):
        assert release_identity == RELEASE
        return runner_declaration("v3")

    def build_raw_runner_input(self, payload, *, source_schema, member_name):
        assert member_name == "build_raw_runner_input"
        return {
            "kind": "raw_icp",
            "raw_icp": {
                "schema_version": "model-raw-icp-envelope:v1",
                "source_schema": source_schema,
                "payload": dict(payload),
            },
        }

    def continue_runner(
        self,
        _start,
        *,
        expected_release_identity,
        continuation,
        completion,
        member_name,
    ):
        assert expected_release_identity == RELEASE
        assert member_name == "continue_runner"
        if continuation is None:
            return {
                "status": "action_required",
                "action": _action(),
                "continuation": {
                    "schema_version": "model-runner-continuation:v3",
                    "step": 1,
                },
            }
        assert completion["completion_sha256"] == "e" * 64
        return {
            "status": "completed",
            "continuation": {
                "schema_version": "model-runner-continuation:v3",
                "step": 2,
            },
            "result": {
                "schema_version": "model-runner-result:v3",
                "leads": [],
            },
            "model_receipt": {
                "schema_version": "model-runner-receipt:v3",
                "receipt_sha256": "f" * 64,
            },
        }

    def build_runner_start(self, *, member_name, **values):
        assert member_name == "build_runner_start"
        return {
            "schema_version": "model-runner-start:v3",
            "host_capability_manifest": values["host_capability_manifest"],
            "start": True,
        }

    def build_runner_completion(
        self, _action_value, result, *, member_name
    ):
        assert member_name == "build_runner_completion"
        self.completion_inputs.append(result)
        return {
            "schema_version": "model-runner-completion:v3",
            "outcome": result["outcome"],
            "calls": result["calls"],
            "cost_credits": result["cost_credits"],
            "latency_ms": result["latency_ms"],
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_receipt_sha256": result[
                "provider_receipt_sha256"
            ],
            "provider_identity_sha256": result[
                "provider_identity_sha256"
            ],
            "provider_response_sha256": "d" * 64,
            "completion_sha256": "e" * 64,
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
            "receipt_sha256": PROVIDER_RECEIPT_SHA256,
        }

    def runner_preflight(self, *, execution_mode, member_name, **_values):
        assert member_name == "runner_preflight"
        return {
            "schema_version": "model-runner-preflight:v3",
            "execution_mode": execution_mode,
        }

    def validate_runner_preflight(
        self, value, *, member_name, **_values
    ):
        assert member_name == "validate_runner_preflight"
        return value

    def validate_runner_result(self, value, *, member_name, **_values):
        assert member_name == "validate_runner_result"
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
            provider_receipt_ref=PROVIDER_RECEIPT_REF,
            provider_receipt_sha256=PROVIDER_RECEIPT_SHA256,
            provider_identity_sha256=PROVIDER_IDENTITY_SHA256,
        )


def test_lab_dispatches_deepline_through_the_artifact_protocol():
    artifact = FakeArtifactTransport()
    protocol = ResearchLabModelRunnerProtocol(
        transport=artifact,
        expected_release_identity=RELEASE,
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
        provider_receipt_custody=DurableCustody(),
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
    assert set(artifact.completion_inputs[0]) == {
        "outcome",
        "reason_code",
        "provider_response",
        "calls",
        "cost_credits",
        "latency_ms",
        "provider_receipt_ref",
        "provider_receipt_sha256",
        "provider_identity_sha256",
    }
    assert artifact.completion_inputs[0]["provider_receipt_ref"] == (
        PROVIDER_RECEIPT_REF
    )
    assert artifact.completion_inputs[0]["provider_receipt_sha256"] == (
        PROVIDER_RECEIPT_SHA256
    )
    assert artifact.completion_inputs[0]["provider_identity_sha256"] == (
        PROVIDER_IDENTITY_SHA256
    )
    assert persisted[0]["host_receipt"]["provider_receipt_sha256"] == (
        PROVIDER_RECEIPT_SHA256
    )
    assert persisted[0]["host_receipt"]["schema_version"] == (
        "model-runner-host-receipt:v2"
    )
    assert "private-provider-id" not in str(persisted[0]["host_receipt"])


def test_host_rejects_missing_custody_before_provider_dispatch():
    artifact = FakeArtifactTransport()
    protocol = ResearchLabModelRunnerProtocol(
        transport=artifact,
        expected_release_identity=RELEASE,
    )
    dispatches = []
    binding = HostActionBinding(
        action_type="execute_intent_tool",
        tool_id="intent.source_add.predictleads_financing",
        binding_contract_sha256=CONTRACT_HASH,
        dispatch=lambda action: dispatches.append(action),
    )

    with pytest.raises(
        ModelRunnerHostError,
        match="durable provider receipt custody is required",
    ):
        CommonModelRunnerHost(
            consumer_id="research-lab-champion",
            protocol=protocol,
            bindings=[binding],
            persist_transition=lambda **_value: None,
        ).run(_start_request())

    assert dispatches == []
    assert artifact.completion_inputs == []


def test_host_rejects_non_durable_custody_before_provider_dispatch():
    class NonDurableCustody(DurableCustody):
        durable = False

    artifact = FakeArtifactTransport()
    protocol = ResearchLabModelRunnerProtocol(
        transport=artifact,
        expected_release_identity=RELEASE,
    )
    dispatches = []
    binding = HostActionBinding(
        action_type="execute_intent_tool",
        tool_id="intent.source_add.predictleads_financing",
        binding_contract_sha256=CONTRACT_HASH,
        dispatch=lambda action: dispatches.append(action),
    )

    with pytest.raises(
        ModelRunnerHostError,
        match="durable provider receipt custody is required",
    ):
        CommonModelRunnerHost(
            consumer_id="research-lab-champion",
            protocol=protocol,
            bindings=[binding],
            persist_transition=lambda **_value: None,
            provider_receipt_custody=NonDurableCustody(),
        ).run(_start_request())

    assert dispatches == []
    assert artifact.completion_inputs == []


def test_host_rejects_custody_hash_mismatch_before_model_completion():
    artifact = FakeArtifactTransport()
    protocol = ResearchLabModelRunnerProtocol(
        transport=artifact,
        expected_release_identity=RELEASE,
    )
    custody = DurableCustody(
        ProviderReceiptCustodyRecord(
            provider_receipt_ref=PROVIDER_RECEIPT_REF,
            provider_receipt_sha256="9" * 64,
            provider_identity_sha256=PROVIDER_IDENTITY_SHA256,
        )
    )
    binding = build_deepline_runner_binding(
        model_tool_id="intent.source_add.predictleads_financing",
        binding_contract_sha256=CONTRACT_HASH,
        client=FakeDeeplineClient(),
    )

    with pytest.raises(
        ModelRunnerHostError,
        match="custody differs from the host result",
    ):
        CommonModelRunnerHost(
            consumer_id="research-lab-champion",
            protocol=protocol,
            bindings=[binding],
            persist_transition=lambda **_value: None,
            provider_receipt_custody=custody,
        ).run(_start_request())

    assert custody.calls == [PROVIDER_RECEIPT_REF]
    assert artifact.completion_inputs == []


def test_host_rejects_artifact_completion_that_changes_custody():
    class MutatingArtifact(FakeArtifactTransport):
        def build_runner_completion(self, action, result, *, member_name):
            completion = super().build_runner_completion(
                action, result, member_name=member_name
            )
            return {**completion, "provider_receipt_sha256": "9" * 64}

    artifact = MutatingArtifact()
    protocol = ResearchLabModelRunnerProtocol(
        transport=artifact,
        expected_release_identity=RELEASE,
    )
    binding = build_deepline_runner_binding(
        model_tool_id="intent.source_add.predictleads_financing",
        binding_contract_sha256=CONTRACT_HASH,
        client=FakeDeeplineClient(),
    )

    with pytest.raises(
        ModelRunnerHostError,
        match="model completion changed provider receipt custody",
    ):
        CommonModelRunnerHost(
            consumer_id="research-lab-champion",
            protocol=protocol,
            bindings=[binding],
            persist_transition=lambda **_value: None,
            provider_receipt_custody=DurableCustody(),
        ).run(_start_request())


def test_host_revalidates_durable_custody_for_cached_completion():
    artifact = FakeArtifactTransport()
    protocol = ResearchLabModelRunnerProtocol(
        transport=artifact,
        expected_release_identity=RELEASE,
    )
    persisted = []
    cached = {
        "outcome": "succeeded",
        "calls": 1,
        "cost_credits": 0.56,
        "latency_ms": 20,
        "provider_receipt_ref": PROVIDER_RECEIPT_REF,
        "provider_receipt_sha256": PROVIDER_RECEIPT_SHA256,
        "provider_identity_sha256": PROVIDER_IDENTITY_SHA256,
        "provider_response_sha256": "d" * 64,
        "completion_sha256": "e" * 64,
    }

    def must_not_dispatch(_action):
        raise AssertionError("cached completion must not dispatch")

    binding = HostActionBinding(
        action_type="execute_intent_tool",
        tool_id="intent.source_add.predictleads_financing",
        binding_contract_sha256=CONTRACT_HASH,
        dispatch=must_not_dispatch,
    )
    custody = DurableCustody()

    result = CommonModelRunnerHost(
        consumer_id="research-lab-champion",
        protocol=protocol,
        bindings=[binding],
        persist_transition=lambda **value: persisted.append(value),
        provider_receipt_custody=custody,
        load_completion=lambda _key: cached,
    ).run(_start_request())

    assert result["status"] == "completed"
    assert custody.calls == [PROVIDER_RECEIPT_REF]
    assert persisted[0]["host_receipt"]["replayed"] is True


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
            expected_release_identity=RELEASE,
        )
    except ModelRunnerHostError as exc:
        assert "runner_protocol_generation" in str(exc)
    else:
        raise AssertionError("incomplete artifact transport must fail closed")


def test_artifact_protocol_generation_is_loaded_once_and_immutable():
    class ChangingDeclarationTransport(FakeArtifactTransport):
        declaration_calls = 0

        def runner_protocol_generation(self, *, release_identity):
            assert release_identity == RELEASE
            self.declaration_calls += 1
            if self.declaration_calls == 1:
                return runner_declaration("v3")
            return runner_declaration("v2")

    transport = ChangingDeclarationTransport()
    protocol = ResearchLabModelRunnerProtocol(
        transport=transport,
        expected_release_identity=RELEASE,
    )

    first = protocol.protocol_generation
    second = protocol.protocol_generation

    assert second is first
    assert first.family == "model-runner-protocol:v3"
    assert transport.declaration_calls == 1


@pytest.mark.parametrize(
    ("family", "wrong_continuation"),
    [("v3", "model-runner-continuation:v2"),
     ("v2", "model-runner-continuation:v3")],
)
def test_runner_generation_rejects_cross_generation_continuation(
    family,
    wrong_continuation,
):
    release = runner_release_identity(family)

    class CrossGenerationTransport(FakeArtifactTransport):
        def runner_protocol_generation(self, *, release_identity):
            assert release_identity == release
            return runner_declaration(family)

        def continue_runner(self, *_args, **_kwargs):
            return {
                "status": "action_required",
                "action": {
                    **_action(),
                    "schema_version": (
                        "model-runner-action:v2"
                        if family == "v3"
                        else "model-runner-action:v1"
                    ),
                },
                "continuation": {
                    "schema_version": wrong_continuation,
                },
            }

    protocol = ResearchLabModelRunnerProtocol(
        transport=CrossGenerationTransport(),
        expected_release_identity=release,
    )

    with pytest.raises(ModelRunnerHostError, match="action identity"):
        protocol.advance({}, continuation=None, completion=None)


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_v3_declaration_requires_exact_provider_receipt_member_constant(
    mutation,
):
    declaration = runner_declaration("v3")
    constants = declaration["consumer_contract"]["exact_constants"][
        "sourcing_model/model_runner.py"
    ]
    if mutation == "missing":
        constants.pop("MODEL_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION")
    else:
        constants["MODEL_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION"] = (
            "model-provider-receipt-binding:forged"
        )

    with pytest.raises(ModelRunnerHostError, match="normalization identities"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            declaration,
            expected_consumer_contract_sha256=("c" * 64),
        )


def test_v2_declaration_cannot_claim_the_v3_receipt_member():
    declaration = deepcopy(runner_declaration("v2"))
    declaration["champion_execution"][
        "provider_receipt_binding_entrypoint"
    ] = "build_runner_provider_receipt_binding"

    with pytest.raises(ModelRunnerHostError, match="metadata differs"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            declaration,
            expected_consumer_contract_sha256=("c" * 64),
        )


def test_artifact_protocol_rejects_preflight_for_a_different_mode():
    class MismatchedPreflightTransport(FakeArtifactTransport):
        def runner_preflight(self, **_values):
            return {
                "schema_version": "model-runner-preflight:v3",
                "execution_mode": "intent_refresh",
            }

    protocol = ResearchLabModelRunnerProtocol(
        transport=MismatchedPreflightTransport(),
        expected_release_identity=RELEASE,
    )

    with pytest.raises(ModelRunnerHostError, match="preflight is invalid"):
        protocol.preflight(
            host_capability_manifest={"bindings": []},
            execution_mode="full_company",
        )


def test_artifact_protocol_rejects_invalid_action_state_before_host_dispatch():
    class InvalidStateTransport(FakeArtifactTransport):
        def continue_runner(self, *args, **kwargs):
            return {
                "status": "action_required",
                "action": {"action_type": "execute_intent_tool"},
                "continuation": {
                    "schema_version": "model-runner-continuation:v3",
                    "step": 1,
                },
            }

    protocol = ResearchLabModelRunnerProtocol(
        transport=InvalidStateTransport(),
        expected_release_identity=RELEASE,
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
        expected_release_identity=RELEASE,
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

        def preflight(self, *, host_capability_manifest, execution_mode):
            calls.append(
                ("preflight", host_capability_manifest, execution_mode)
            )
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
        provider_receipt_custody=DurableCustody(),
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
                    provider_receipt_ref=PROVIDER_RECEIPT_REF,
                    provider_receipt_sha256=PROVIDER_RECEIPT_SHA256,
                    provider_identity_sha256=PROVIDER_IDENTITY_SHA256,
                ),
            )
        ],
        persist_transition=lambda **value: persisted.append(value),
        provider_receipt_custody=DurableCustody(),
    )
    assert result["status"] == "completed"
    assert len(persisted) == 1


def _registration_for_live_caller():
    from research_lab.model_runner_protocol import ExactModelRunnerRegistration

    release = {
        **runner_release_identity("v2"),
        "source_commit": "1" * 40,
        "model_artifact_digest": "sha256:" + "a" * 64,
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
        "schema_version": "model-runner-action:v1",
        "action_type": "execute_candidate_tool",
        "tool_id": "candidate.reviewed",
        "binding_contract_sha256": CONTRACT_HASH,
        "action_sha256": "b" * 64,
        "idempotency_key": "c" * 64,
    }

    class _Transport(FakeArtifactTransport):
        def runner_protocol_generation(self, *, release_identity):
            assert release_identity == release
            return runner_declaration("v2")

        def build_runner_start(self, *, member_name, **values):
            assert member_name == "build_runner_start"
            return {
                "schema_version": "model-runner-start:v2",
                "host_capability_manifest": values[
                    "host_capability_manifest"
                ],
                "start": True,
            }

        def runner_preflight(
            self, *, execution_mode, member_name, **_values
        ):
            assert member_name == "runner_preflight"
            return {
                "schema_version": "model-runner-preflight:v2",
                "execution_mode": execution_mode,
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

        def build_runner_completion(
            self, _action_value, result, *, member_name
        ):
            assert member_name == "build_runner_completion"
            self.completion_inputs.append(result)
            return {
                "schema_version": "model-runner-completion:v2",
                "outcome": result["outcome"],
                "calls": result["calls"],
                "cost_credits": result["cost_credits"],
                "latency_ms": result["latency_ms"],
                "provider_receipt_ref": result[
                    "provider_receipt_ref"
                ],
                "provider_receipt_sha256": result[
                    "provider_receipt_sha256"
                ],
                "provider_identity_sha256": result[
                    "provider_identity_sha256"
                ],
                "provider_response_sha256": "d" * 64,
                "completion_sha256": "e" * 64,
            }

        def continue_runner(
            self,
            _start,
            *,
            expected_release_identity,
            continuation,
            completion,
            member_name,
        ):
            assert expected_release_identity == release
            assert member_name == "continue_runner"
            if continuation is None:
                return {
                    "status": "action_required",
                    "action": live_action,
                    "continuation": {
                        "schema_version": "model-runner-continuation:v2",
                        "step": 1,
                    },
                }
            assert completion["completion_sha256"] == "e" * 64
            return {
                "status": "completed",
                "continuation": {
                    "schema_version": "model-runner-continuation:v2",
                    "step": 2,
                },
                "result": {
                    "schema_version": "model-runner-result:v2",
                    "leads": [],
                },
                "model_receipt": {
                    "schema_version": "model-runner-receipt:v2",
                    "receipt_sha256": "f" * 64,
                },
            }

    protocol = ResearchLabModelRunnerProtocol(
        transport=_Transport(), expected_release_identity=release
    )
    return ExactModelRunnerRegistration(
        artifact_identity={
            "repository": "leadpoet/Sourcing_model",
            "branch": "main",
            "commit_sha": "1" * 40,
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


def test_oci_generation_discovers_signed_role_path_and_extensions(
    tmp_path,
    monkeypatch,
    capsys,
):
    member = "artifact_provider_prepare"
    consumer_path = "compat/research_lab_adapter.py:" + member
    contract = {
        "schema_version": 73,
        "contract_id": "fixture-v73",
        "functions": {
            "compat/research_lab_adapter.py": {member: ["action"]}
        },
        "exact_signatures": [consumer_path],
        "full_parameters": {
            consumer_path: ["action", "future_optional"]
        },
        "required_keyword_only": {},
        "frozen_asyncness": {consumer_path: False},
        "exact_constants": {},
        "extensions": {
            "leadpoet.fixture": {"hash_bound": True}
        },
    }
    model_dir = tmp_path / "sourcing_model"
    model_dir.mkdir()
    contract_bytes = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (model_dir / "consumer_contract.json").write_bytes(contract_bytes)
    metadata = {
        "champion_execution": {
            "runner_role_contract": {
                "roles": {
                    "provider_prepare": {
                        "adapter_member": member,
                        "consumer_signature": {
                            "consumer_contract_path": consumer_path
                        },
                    }
                }
            }
        }
    }
    (tmp_path / "fixture_adapter.py").write_text(
        "METADATA = "
        + repr(metadata)
        + "\n\ndef adapter_metadata():\n    return METADATA\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["bootstrap", "fixture_adapter", "runner_protocol_generation"],
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO(json.dumps({
            "release_identity": {
                "consumer_contract_sha256": hashlib.sha256(
                    contract_bytes
                ).hexdigest()
            }
        })),
    )

    exec(_COMMON_RUNNER_BOOTSTRAP, {"__name__": "bootstrap_fixture"})

    result = json.loads(capsys.readouterr().out)
    consumer = result["consumer_contract"]
    assert consumer["functions"] == {member: ["action"]}
    assert consumer["exact_signatures"] == [consumer_path]
    assert consumer["full_parameters"] == {
        member: ["action", "future_optional"]
    }
    assert consumer["frozen_asyncness"] == {member: False}
    assert consumer["extensions"] == {
        "leadpoet.fixture": {"hash_bound": True}
    }


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
        execution_mode="full_company",
        member_name="runner_preflight",
    ) == {"operation": "runner_preflight"}
    completion_result = {
        "provider_receipt_ref": PROVIDER_RECEIPT_REF,
        "provider_receipt_sha256": PROVIDER_RECEIPT_SHA256,
        "provider_identity_sha256": PROVIDER_IDENTITY_SHA256,
    }
    assert transport.build_runner_completion(
        {"action_sha256": "a" * 64},
        completion_result,
        member_name="build_runner_completion",
    ) == {"operation": "build_runner_completion"}
    assert transport.validate_runner_result(
        {"status": "completed"},
        start_request={"input": {}},
        expected_release_identity={"source_commit": "a" * 40},
        member_name="validate_runner_result",
    ) == {"operation": "validate_runner_result"}
    assert calls == [
        (
            "runner_preflight",
            {
                "host_capability_manifest": {"bindings": []},
                "release_identity": {"source_commit": "a" * 40},
                "execution_mode": "full_company",
                "member_name": "runner_preflight",
            },
        ),
        (
            "build_runner_completion",
            {
                "action": {"action_sha256": "a" * 64},
                "result": completion_result,
                "member_name": "build_runner_completion",
            },
        ),
        (
            "validate_runner_result",
            {
                "value": {"status": "completed"},
                "start_request": {"input": {}},
                "expected_release_identity": {"source_commit": "a" * 40},
                "member_name": "validate_runner_result",
            },
        ),
    ]
