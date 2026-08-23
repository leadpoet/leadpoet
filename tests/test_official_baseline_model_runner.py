from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import concurrent.futures
from types import SimpleNamespace

import pytest

from gateway.research_lab.common_model_experiment import ProtectedModelActionResult
from gateway.research_lab import scoring_worker as scoring_worker_module
from gateway.research_lab.official_baseline_model_runner import (
    ArtifactBenchmarkProjection,
    EXACT_MODEL_RUNNER_FAMILY,
    ExactOfficialBaselineRunner,
    OFFICIAL_BASELINE_AUTHORITY_PREFLIGHT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_PROJECTION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
    OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION,
    OfficialBaselineAuthorityUnavailable,
    OfficialBaselineExactDependencies,
    OfficialBaselineModelError,
    OfficialBaselineReleaseSelectionError,
    select_official_baseline_release,
    validate_official_baseline_checkpoint,
)
from research_lab.canonical import sha256_json
from research_lab.common_model_runner_host import HostActionResult
from research_lab.eval import DockerPrivateModelSpec, PrivateModelArtifactManifest
from research_lab.model_runner_protocol import (
    ArtifactRunnerProtocolGeneration,
    ExactModelRunnerRegistration,
    ModelRunnerHostError,
    ResearchLabModelRunnerProtocol,
)
from research_lab.routing_experiments import ProviderReceipt
from tests.model_runner_protocol_fixtures import (
    runner_declaration,
    runner_release_identity,
)


HASH = {
    name: character * 64
    for name, character in zip(
        (
            "artifact",
            "manifest",
            "contract",
            "catalog",
            "policy",
            "feature",
            "binding",
            "projection",
            "authority",
        ),
        "abcdef123",
    )
}


def _company(name: str = "Acme") -> dict:
    return {
        "company_name": name,
        "company_website": "https://acme.example",
        "industry": "Software",
        "employee_count": "51-200",
        "country": "United States",
        "intent_signals": [
            {
                "source": "news",
                "description": "Expanded its revenue team",
                "url": "https://news.example/acme",
                "date": "2026-08-20",
                "snippet": "The company announced a revenue-team expansion.",
                "matched_icp_signal": 0,
            }
        ],
    }


def _release() -> dict:
    return {
        **runner_release_identity("v3", contract_hash=HASH["contract"]),
        "source_commit": "1" * 40,
        "model_artifact_digest": "sha256:" + HASH["artifact"],
        "consumer_contract_sha256": HASH["contract"],
        "catalog_sha256": HASH["catalog"],
        "policy_sha256": HASH["policy"],
        "candidate_profiles_sha256": "4" * 64,
        "intent_profiles_sha256": "5" * 64,
        "feature_schema_sha256": HASH["feature"],
        "candidate_waterfall_contract_sha256": "6" * 64,
        "tool_binding_manifest_sha256": HASH["binding"],
        "release_identity_sha256": "7" * 64,
    }


def _provider_action(start: dict) -> dict:
    action = {
        "schema_version": "model-runner-action:v2",
        "action_type": "execute_candidate_tool",
        "action_phase": "candidate_acquisition",
        "stage": "candidate_acquisition",
        "tool_id": "candidate.fixture",
        "binding_contract_sha256": HASH["binding"],
        "request_fingerprint_sha256": "8" * 64,
        "idempotency_key": "9" * 64,
        "sequence": 0,
        "max_response_bytes": 100_000,
        "arguments": {"step": {"credit_cap": 0.01, "timeout_seconds": 30}},
    }
    action["action_sha256"] = sha256_json(action).removeprefix("sha256:")
    return action


class _Transport:
    def runner_protocol_generation(self, *, release_identity):
        assert release_identity == _release()
        return runner_declaration("v3", contract_hash=HASH["contract"])

    def build_raw_runner_input(self, payload, *, source_schema, member_name):
        assert member_name == "build_raw_runner_input"
        return {
            "kind": "raw_icp",
            "raw_icp": {
                "schema_version": "model-raw-icp-envelope:v1",
                "source_schema": source_schema,
                "payload": deepcopy(dict(payload)),
            },
        }

    def build_runner_start(self, *, member_name, **values):
        assert member_name == "build_runner_start"
        return {
            "schema_version": "model-runner-start:v3",
            "input": deepcopy(values["input"]),
            "host_capability_manifest": deepcopy(values["host_capability_manifest"]),
        }

    def runner_preflight(self, *, execution_mode, member_name, **_values):
        assert member_name == "runner_preflight"
        return {
            "schema_version": "model-runner-preflight:v3",
            "execution_mode": execution_mode,
        }

    def validate_runner_preflight(self, value, *, member_name, **_values):
        assert member_name == "validate_runner_preflight"
        return value

    def continue_runner(
        self,
        start,
        *,
        continuation,
        completion,
        member_name,
        **_values,
    ):
        assert member_name == "continue_runner"
        payload = start["input"]["raw_icp"]["payload"]
        if payload.get("requires_action") and continuation is None:
            return {
                "status": "action_required",
                "action": _provider_action(start),
                "continuation": {
                    "schema_version": "model-runner-continuation:v3",
                    "pending": True,
                },
            }
        if payload.get("requires_action"):
            assert completion is not None
        outputs = deepcopy(payload.get("outputs") or [])
        return {
            "status": "completed",
            "action": None,
            "continuation": {
                "schema_version": "model-runner-continuation:v3",
                "terminal": True,
            },
            "result": {
                "schema_version": "model-runner-result:v3",
                "outputs": outputs,
            },
            "model_receipt": {
                "schema_version": "model-runner-receipt:v3",
                "outputs_sha256": sha256_json(outputs),
            },
        }

    def validate_runner_result(self, value, *, member_name, **_values):
        assert member_name == "validate_runner_result"
        return value

    def build_runner_completion(self, action, result, *, member_name):
        assert member_name == "build_runner_completion"
        body = {
            "schema_version": "model-runner-completion:v3",
            "action_sha256": action["action_sha256"],
            "provider_response": deepcopy(result["provider_response"]),
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_receipt_sha256": result["provider_receipt_sha256"],
            "provider_identity_sha256": result["provider_identity_sha256"],
        }
        return {**body, "completion_sha256": sha256_json(body).removeprefix("sha256:")}

    def build_runner_provider_receipt_binding(self, action, result, *, member_name):
        assert member_name == "build_runner_provider_receipt_binding"
        body = {
            "action_sha256": action["action_sha256"],
            "provider_response": result["provider_response"],
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_identity_sha256": result["provider_identity_sha256"],
        }
        return {
            "schema_version": "model-provider-receipt-binding:v1",
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_identity_sha256": result["provider_identity_sha256"],
            "receipt_sha256": sha256_json(body).removeprefix("sha256:"),
        }


def _registration() -> ExactModelRunnerRegistration:
    identity = {
        "repository": "leadpoet/Sourcing_model",
        "branch": "main",
        "commit_sha": "1" * 40,
        "model_artifact_hash": "sha256:" + HASH["artifact"],
        "manifest_hash": "sha256:" + HASH["manifest"],
        "routing_contract_hash": "sha256:" + HASH["contract"],
        "routing_catalog_hash": "sha256:" + HASH["catalog"],
        "routing_policy_hash": "sha256:" + HASH["policy"],
        "feature_schema_hash": "sha256:" + HASH["feature"],
    }
    return ExactModelRunnerRegistration(
        artifact_identity=identity,
        protocol=ResearchLabModelRunnerProtocol(
            transport=_Transport(), expected_release_identity=_release()
        ),
        host_capability_manifest={
            "manifest_sha256": "sha256:" + "0" * 64,
            "bindings": [
                {
                    "action_type": "execute_candidate_tool",
                    "tool_id": "candidate.fixture",
                    "binding_contract_sha256": HASH["binding"],
                    "available": True,
                }
            ],
        },
    )


class _Projector:
    def __init__(self, registration):
        self.artifact_key = registration.key
        self.protocol_generation_sha256 = (
            registration.protocol_generation.protocol_generation_sha256
        )
        self.projection_identity_sha256 = "sha256:" + HASH["projection"]
        self.drift = False

    def project_company_outputs(self, *, start_request, terminal_result):
        outputs = deepcopy(terminal_result["result"]["outputs"])
        if self.drift:
            outputs = [*outputs, _company("Drifted")]
        body = {
            "schema_version": OFFICIAL_BASELINE_PROJECTION_SCHEMA_VERSION,
            "projection_identity_sha256": self.projection_identity_sha256,
            "source_result_sha256": sha256_json(dict(terminal_result)),
            "outputs_sha256": sha256_json(outputs),
        }
        return ArtifactBenchmarkProjection(
            outputs=tuple(outputs),
            projection_receipt={
                **body,
                "projection_sha256": sha256_json(body),
            },
        )


class _Transitions:
    def __init__(self, generation):
        self.generation = generation
        self.values = {}

    def resolve_run_protocol_generation(self, **_identity):
        return self.generation

    def load_model_transition(self, **identity):
        return self.values.get(identity["idempotency_key"])

    def append_model_transition(self, **value):
        self.values[value["action"]["idempotency_key"]] = {
            "action": deepcopy(value["action"]),
            "continuation": deepcopy(value["continuation"]),
            "completion": deepcopy(value["completion"]),
            "provider_receipt": deepcopy(value["provider_receipt"]),
            "protocol_generation_sha256": value["protocol_generation_sha256"],
        }


class _Dispatcher:
    def __init__(self):
        self.provider_calls = 0

    def dispatch_provider_action(self, *, action, unit_ref, **_values):
        self.provider_calls += 1
        receipt = ProviderReceipt(
            receipt_ref="provider_receipt:" + "a" * 16,
            binding_id="fixture-binding",
            tool_id=action["tool_id"],
            binding_version="v1",
            source_lineage_id="fixture-source",
            unit_ref=unit_ref,
            request_fingerprint="sha256:" + "8" * 64,
            outcome="verified",
            evidence_hash="sha256:" + "b" * 64,
            credit_microunits=10,
            latency_ms=20,
            execution_mode="measured_lab",
            call_count=1,
        )
        return ProtectedModelActionResult(
            host_result=HostActionResult(
                outcome="succeeded",
                reason_code="fixture",
                provider_response={"companies": []},
                calls=1,
                cost_credits=0.00001,
                latency_ms=20,
                provider_receipt_ref=receipt.receipt_ref,
                provider_identity_sha256="c" * 64,
            ),
            provider_receipt=receipt,
            replay_ref={"protected_job_ref": "fixture-job"},
        )

    def replay_provider_action(self, **_values):
        raise AssertionError("full transition replay does not redispatch")

    def verify_company_action(self, **_values):
        raise AssertionError("not called")

    verify_intent_action = verify_company_action
    verify_contact_action = verify_company_action


class _ProtectedAuthority:
    authority_identity_sha256 = "sha256:" + HASH["authority"]

    def __init__(self, registration):
        generation = registration.protocol_generation.protocol_generation_sha256
        self.transitions = _Transitions(generation)
        self.dispatcher = _Dispatcher()
        self.closures = {}

    def preflight_run(self, *, run_identity, registration):
        return {
            "schema_version": OFFICIAL_BASELINE_AUTHORITY_PREFLIGHT_SCHEMA_VERSION,
            "run_sha256": sha256_json(dict(run_identity)),
            "artifact_key_sha256": sha256_json({"artifact_key": registration.key}),
            "protocol_generation_sha256": (
                registration.protocol_generation.protocol_generation_sha256
            ),
            "authority_identity_sha256": self.authority_identity_sha256,
            "ready": True,
        }

    def dispatcher_for_unit(self, **_values):
        return self.dispatcher

    def transition_repository_for_unit(self, **_values):
        return self.transitions

    def close_unit(self, *, completion):
        ordered_keys = []
        ordered_hashes = []
        for stored in self.transitions.values.values():
            action = stored["action"]
            attempt_key = sha256_json(
                {
                    "run_sha256": completion["run_sha256"],
                    "unit_ref": completion["unit_ref"],
                    "action_idempotency_sha256": (
                        "sha256:" + action["idempotency_key"]
                    ),
                }
            )
            authorization_sha = sha256_json(action)
            terminal_doc_sha = sha256_json(stored["completion"])
            ordered_keys.append(attempt_key)
            ordered_hashes.append(
                sha256_json(
                    {
                        "authorization_sha256": authorization_sha,
                        "terminal_state": "terminal_known",
                        "terminal_doc_sha256": terminal_doc_sha,
                    }
                )
            )
        frontier = {
            "schema_version": OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
            "ordered_attempt_keys": ordered_keys,
            "ordered_attempt_sha256s": ordered_hashes,
        }
        body = {
            "schema_version": OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION,
            **{
                key: value
                for key, value in completion.items()
                if key != "schema_version"
            },
            "ordered_attempt_keys": ordered_keys,
            "ordered_attempt_sha256s": ordered_hashes,
            "provider_frontier_sha256": sha256_json(frontier),
        }
        closure_sha = sha256_json(body)
        result = {
            **body,
            "closure_ref": "baseline_closure:" + closure_sha.removeprefix("sha256:"),
            "closure_sha256": closure_sha,
        }
        existing = self.closures.get(completion["unit_ref"])
        if existing is not None and existing != result:
            raise OfficialBaselineModelError("provider closure conflict")
        self.closures[completion["unit_ref"]] = deepcopy(result)
        return {**result, "idempotent": existing is not None}

    def load_frontier(self, *, run_sha256, unit_ref):
        result = deepcopy(self.closures[unit_ref])
        assert result["run_sha256"] == run_sha256
        return {**result, "idempotent": True}


class _TerminalAuthority:
    def __init__(self):
        self.records = {}

    def persist_terminal_record(self, *, record_identity_sha256, record):
        existing = self.records.get(record_identity_sha256)
        if existing is not None and existing != dict(record):
            raise OfficialBaselineModelError("terminal record conflict")
        self.records[record_identity_sha256] = deepcopy(dict(record))
        return {
            "terminal_record_ref": "baseline_terminal:"
            + record_identity_sha256.removeprefix("sha256:"),
            "terminal_record_sha256": sha256_json(dict(record)),
        }

    def load_terminal_record(self, *, terminal_record_ref):
        identity = "sha256:" + terminal_record_ref.split(":", 1)[1]
        return deepcopy(self.records[identity])


def _exact_fixture():
    registration = _registration()
    projector = _Projector(registration)
    authority = _ProtectedAuthority(registration)
    terminal = _TerminalAuthority()
    execution = {
        "schema_version": OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
        "runner_family": EXACT_MODEL_RUNNER_FAMILY,
        "execution_mode": "measured_lab",
        "release_identity_sha256": sha256_json(_release()),
        "protocol_generation_sha256": (
            registration.protocol_generation.protocol_generation_sha256
        ),
        "benchmark_projection_sha256": projector.projection_identity_sha256,
        "protected_action_authority_sha256": authority.authority_identity_sha256,
    }
    artifact = PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + HASH["artifact"],
        git_commit_sha="1" * 40,
        image_digest="example.invalid/model@sha256:" + "d" * 64,
        config_hash="sha256:" + "e" * 64,
        component_registry_version="components:v3",
        scoring_adapter_version="scoring:v1",
        manifest_uri="s3://fixture/model/manifest.json",
        manifest_hash="sha256:" + HASH["manifest"],
        signature_ref="kms://fixture",
        signed_extensions={
            "model_release_identity": _release(),
            "official_baseline_execution": execution,
        },
    )
    dependencies = OfficialBaselineExactDependencies(
        registration=registration,
        projector=projector,
        protected_authority=authority,
        terminal_authority=terminal,
    )
    selection = select_official_baseline_release(artifact)
    runner = ExactOfficialBaselineRunner(
        artifact=artifact,
        selection=selection,
        dependencies=dependencies,
        spec=DockerPrivateModelSpec(image_digest=artifact.image_digest),
        benchmark_date="2026-08-23",
        rolling_window_hash="sha256:" + "f" * 64,
    )
    return runner, projector, authority, terminal


def test_exact_official_baseline_positive_and_empty_retry_zero():
    runner, _projector, _authority, _terminal = _exact_fixture()
    positive = runner.run_icp(
        raw_icp={"outputs": [_company()]},
        icp_ref="icp-positive",
        target_count=1,
    )
    empty = runner.run_icp(
        raw_icp={"outputs": []},
        icp_ref="icp-empty",
        target_count=1,
    )

    assert positive.company_outputs == (_company(),)
    assert empty.company_outputs == ()
    assert validate_official_baseline_checkpoint(empty.checkpoint) == empty.checkpoint


@pytest.mark.asyncio
async def test_scoring_worker_accepts_exact_empty_on_retry_zero(monkeypatch):
    runner, _projector, _authority, _terminal = _exact_fixture()
    worker = object.__new__(scoring_worker_module.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "test-worker"
    worker.config = SimpleNamespace(private_baseline_provider_retry_rounds=2)
    worker._active_baseline_context = {}

    async def unchanged(**_values):
        return None

    async def no_traces(**_values):
        return None

    worker._ensure_private_baseline_repo_head_unchanged = unchanged
    worker._record_baseline_icp_traces = no_traces
    monkeypatch.setattr(
        scoring_worker_module,
        "_apply_provider_cost_baseline_outcome",
        lambda _row: None,
    )
    item = {
        "icp_ref": "icp-empty-worker",
        "icp_hash": "sha256:" + "1" * 64,
        "set_id": "set-1",
        "day_index": 0,
        "day_rank": 1,
        "icp": {"outputs": [], "max_companies": 1},
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        row = await worker._run_baseline_icp(
            runner=runner,
            scorer=object(),
            item=item,
            item_index=1,
            total_icps=1,
            run_start=0.0,
            executor=executor,
            benchmark_date="2026-08-23",
            retry_round=0,
        )

    assert row["_retryable"] is False
    assert row["_nonempty"] is False
    assert row["diagnostics"]["sourcing_failed"] is False
    assert row["diagnostics"]["empty_result_provider_evidence_validated"] is True
    assert scoring_worker_module._OFFICIAL_BASELINE_CHECKPOINT_FIELD in row


def test_restart_reconstructs_and_does_not_duplicate_provider_call():
    runner, _projector, authority, terminal = _exact_fixture()
    raw = {"requires_action": True, "outputs": [_company()]}
    first = runner.run_icp(raw_icp=raw, icp_ref="icp-restart", target_count=1)
    second = runner.run_icp(
        raw_icp=raw,
        icp_ref="icp-restart",
        target_count=1,
        expected_checkpoint=first.checkpoint,
    )

    assert second.checkpoint == first.checkpoint
    assert second.replayed_transition_count == 1
    assert authority.dispatcher.provider_calls == 1
    assert len(terminal.records) == 1


@pytest.mark.parametrize(
    "field",
    ["terminal_result_sha256", "provider_frontier_sha256", "checkpoint_sha256"],
)
def test_checkpoint_tamper_fails_closed(field):
    runner, _projector, _authority, _terminal = _exact_fixture()
    result = runner.run_icp(
        raw_icp={"outputs": []}, icp_ref="icp-tamper", target_count=1
    )
    tampered = dict(result.checkpoint)
    tampered[field] = "sha256:" + "0" * 64

    with pytest.raises(OfficialBaselineModelError):
        runner.run_icp(
            raw_icp={"outputs": []},
            icp_ref="icp-tamper",
            target_count=1,
            expected_checkpoint=tampered,
        )


def test_generation_and_projection_drift_fail_closed():
    runner, projector, _authority, _terminal = _exact_fixture()
    bad_selection_doc = dict(runner.selection.selection_document)
    bad_selection_doc["protocol_generation_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(OfficialBaselineModelError, match="registration differs"):
        ExactOfficialBaselineRunner(
            artifact=runner.artifact,
            selection=replace(runner.selection, selection_document=bad_selection_doc),
            dependencies=runner.dependencies,
            spec=runner.spec,
            benchmark_date="2026-08-23",
            rolling_window_hash="sha256:" + "f" * 64,
        )

    original = runner.run_icp(
        raw_icp={"outputs": [_company()]},
        icp_ref="icp-projection",
        target_count=1,
    )
    projector.drift = True
    with pytest.raises(OfficialBaselineModelError):
        runner.run_icp(
            raw_icp={"outputs": [_company()]},
            icp_ref="icp-projection",
            target_count=1,
            expected_checkpoint=original.checkpoint,
        )


def test_exact_release_requires_protected_authority_at_startup():
    runner, _projector, _authority, _terminal = _exact_fixture()
    with pytest.raises(OfficialBaselineAuthorityUnavailable):
        ExactOfficialBaselineRunner(
            artifact=runner.artifact,
            selection=runner.selection,
            dependencies=None,
            spec=runner.spec,
            benchmark_date="2026-08-23",
            rolling_window_hash="sha256:" + "f" * 64,
        )


def test_old_drain_is_selected_only_by_exact_signed_legacy_contract():
    artifact = PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + HASH["artifact"],
        git_commit_sha="1" * 40,
        image_digest="example.invalid/model@sha256:" + "d" * 64,
        config_hash="sha256:" + "e" * 64,
        component_registry_version="components:v2",
        scoring_adapter_version="scoring:v1",
        manifest_uri="s3://fixture/model/manifest.json",
        manifest_hash="sha256:" + HASH["manifest"],
        signature_ref="kms://fixture",
        compatibility_contract={
            "contract_id": "legacy-qualify-v2",
            "path": "contracts/qualify.json",
            "sha256": "sha256:" + "1" * 64,
        },
        consumer_parity_fixtures={
            "path": "contracts/qualify-fixtures.json",
            "sha256": "sha256:" + "2" * 64,
        },
    )
    assert select_official_baseline_release(artifact).runner_family == (
        "attested_private_model:v2"
    )

    mixed = replace(
        artifact,
        signed_extensions={"model_release_identity": _release()},
    )
    with pytest.raises(OfficialBaselineReleaseSelectionError):
        select_official_baseline_release(mixed)


def test_completion_accounting_identity_creates_a_distinct_exact_generation():
    old_declaration = runner_declaration("v3", contract_hash=HASH["contract"])
    new_declaration = deepcopy(old_declaration)
    new_declaration["champion_execution"][
        "completion_accounting_schema_version"
    ] = "model-runner-completion-accounting:v2"
    new_declaration["consumer_contract"]["exact_constants"][
        "sourcing_model/model_runner.py"
    ][
        "MODEL_RUNNER_COMPLETION_ACCOUNTING_SCHEMA_VERSION"
    ] = "model-runner-completion-accounting:v2"

    old = ArtifactRunnerProtocolGeneration.from_declaration(
        old_declaration,
        expected_consumer_contract_sha256=HASH["contract"],
    )
    new = ArtifactRunnerProtocolGeneration.from_declaration(
        new_declaration,
        expected_consumer_contract_sha256=HASH["contract"],
    )

    assert old.protocol_generation_sha256 != new.protocol_generation_sha256
    assert (
        new.version("MODEL_RUNNER_COMPLETION_ACCOUNTING_SCHEMA_VERSION")
        == "model-runner-completion-accounting:v2"
    )

    tampered = deepcopy(new_declaration)
    tampered["champion_execution"][
        "completion_accounting_schema_version"
    ] = "model-runner-completion-accounting:v1"
    with pytest.raises(ModelRunnerHostError, match="schema tuple differs"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            tampered,
            expected_consumer_contract_sha256=HASH["contract"],
        )
