from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import concurrent.futures
import hashlib
import json
from types import SimpleNamespace

import pytest

from gateway.research_lab.common_model_experiment import (
    CommonModelExperimentError,
    ProtectedModelActionResult,
    _bind_durable_provider_result,
)
from gateway.research_lab import scoring_worker as scoring_worker_module
from gateway.research_lab.official_baseline_authority import (
    OfficialBaselineTerminalUncertainError,
)
from gateway.research_lab.official_baseline_model_runner import (
    ArtifactBenchmarkProjection,
    ArtifactProtocolBenchmarkProjector,
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
from research_lab.eval import (
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
)
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
HASH["authority"] = (
    "7f93061601526ce3d14b8555fefe388a1fd7322b565a748f06e232e2cb5c1b7a"
)


def _bare_wire_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rehash_role_contract(declaration: dict) -> None:
    contract = declaration["champion_execution"]["runner_role_contract"]
    contract["contract_sha256"] = _bare_wire_hash({
        key: value for key, value in contract.items()
        if key != "contract_sha256"
    })


def _upgrade_role_contract_v2(declaration: dict) -> None:
    contract = declaration["champion_execution"]["runner_role_contract"]
    contract["schema_version"] = "model-runner-role-contract:v2"
    for entry in contract["roles"].values():
        signature = entry["consumer_signature"]
        signature["required_positional_parameters"] = list(
            signature["positional_parameters"]
        )
        signature["defaulted_positional_parameters"] = []
        entry["consumer_signature_sha256"] = _bare_wire_hash(signature)
    _rehash_role_contract(declaration)


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
            "execution_mode": values["execution_mode"],
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


class _OfficialTransport(_Transport):
    def runner_protocol_generation(self, *, release_identity):
        assert release_identity == _release()
        return runner_declaration(
            "v3",
            contract_hash=HASH["contract"],
            official_baseline=True,
        )

    def project_runner_result_for_benchmark(
        self,
        value,
        *,
        start_request,
        expected_release_identity,
        member_name,
    ):
        assert member_name == "project_runner_result_for_benchmark"
        assert expected_release_identity == _release()

        def wire_hash(payload):
            return hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()

        companies = deepcopy(value["result"]["outputs"])
        body = {
            "schema_version": OFFICIAL_BASELINE_PROJECTION_SCHEMA_VERSION,
            "start_request_sha256": wire_hash(dict(start_request)),
            "release_identity_sha256": wire_hash(_release()),
            "model_receipt_sha256": wire_hash(value["model_receipt"]),
            "companies": companies,
            "companies_sha256": wire_hash(companies),
        }
        return {**body, "projection_sha256": wire_hash(body)}

    @staticmethod
    def _wire_hash(value):
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _catalog_bindings(cls):
        return [{
            "schema_version": "host-capability-binding:v1",
            "action_type": "verify_company",
            "tool_id": "verifier.company",
            "binding_contract_sha256": HASH["binding"],
            "response_schema_version": "company-verifier-result:v1",
            "idempotency": "idempotent",
            "max_response_bytes": 100_000,
        }]

    def runner_official_host_binding_catalog(self, *, member_name):
        assert member_name == "runner_official_host_binding_catalog"
        bindings = self._catalog_bindings()
        body = {
            "schema_version": (
                "model-runner-official-host-binding-catalog:v1"
            ),
            "bindings": bindings,
            "binding_contracts_sha256": self._wire_hash(bindings),
        }
        return {**body, "catalog_sha256": self._wire_hash(body)}

    def build_runner_official_host_capability_manifest(
        self, availability, *, member_name
    ):
        assert member_name == "build_runner_official_host_capability_manifest"
        assert availability == {"verifier.company": True}
        bindings = [
            {**binding, "available": availability[binding["tool_id"]]}
            for binding in self._catalog_bindings()
        ]
        body = {
            "schema_version": "host-capability-manifest:v1",
            "binding_contracts_sha256": self._wire_hash(
                self._catalog_bindings()
            ),
            "bindings": bindings,
        }
        return {**body, "manifest_sha256": self._wire_hash(body)}

    def execute_runner_verifier_action(
        self, action, *, member_name
    ):
        assert member_name == "execute_runner_verifier_action"
        result = {"status": "accepted", "reason_code": "fixture"}
        body = {
            "schema_version": "model-runner-verifier-execution:v1",
            "action_sha256": action["action_sha256"],
            "action_type": action["action_type"],
            "calls": 0,
            "cost_credits": 0.0,
            "provider_receipt_allowed": False,
            "result": result,
            "result_sha256": self._wire_hash(result),
        }
        return {**body, "execution_sha256": self._wire_hash(body)}

    def model_runner_provider_compiler_inventory(self, *, member_name):
        assert member_name == "model_runner_provider_compiler_inventory"
        return {
            "schema_version": "model-runner-provider-compiler-inventory:v1",
            "entries": [],
            "inventory_sha256": "f" * 64,
        }

    def prepare_runner_provider_request(self, action, *, member_name):
        assert member_name == "prepare_runner_provider_request"
        request = {"credential_binding": {"source": "fixture"}}
        body = {
            "schema_version": "model-runner-provider-dispatch:v1",
            "action_sha256": action["action_sha256"],
            "action_type": action["action_type"],
            "tool_id": action["tool_id"],
            "compiler_id": "fixture.compiler:v1",
            "compiler_contract_sha256": action[
                "binding_contract_sha256"
            ],
            "provider": "fixture",
            "request": request,
            "request_sha256": self._wire_hash(request),
            "response_contract": {},
            "budgets": {},
            "idempotency_key": action["idempotency_key"],
        }
        return {**body, "dispatch_sha256": self._wire_hash(body)}

    def ingest_runner_provider_response(
        self, action, host_response, *, member_name
    ):
        assert member_name == "ingest_runner_provider_response"
        dispatch = self.prepare_runner_provider_request(
            action,
            member_name="prepare_runner_provider_request",
        )
        parsed = {
            "schema_version": "model-provider-response:v3",
            "records": [],
            "freshness_context": {},
            "extensions": {},
            "records_sha256": self._wire_hash([]),
        }
        body = {
            "schema_version": "model-runner-provider-response-ingestion:v1",
            "action_sha256": action["action_sha256"],
            "dispatch_sha256": dispatch["dispatch_sha256"],
            "compiler_id": dispatch["compiler_id"],
            "compiler_contract_sha256": dispatch[
                "compiler_contract_sha256"
            ],
            "request_sha256": dispatch["request_sha256"],
            "host_response_schema_version": "host-provider-response:v1",
            "host_response_sha256": self._wire_hash(host_response),
            "provider": dispatch["provider"],
            "parsed_response_schema_version": (
                "model-provider-response:v3"
            ),
            "parsed_response": parsed,
            "parsed_response_sha256": self._wire_hash(parsed),
        }
        return {**body, "ingestion_sha256": self._wire_hash(body)}

    def build_runner_completion(self, action, result, *, member_name):
        if not isinstance(result.get("provider_response"), dict) or (
            result["provider_response"].get("schema_version")
            != "host-provider-response:v1"
        ):
            return super().build_runner_completion(
                action, result, member_name=member_name
            )
        ingestion = self.ingest_runner_provider_response(
            action,
            result["provider_response"],
            member_name="ingest_runner_provider_response",
        )
        body = {
            "schema_version": "model-runner-completion:v3",
            "action_sha256": action["action_sha256"],
            "provider_response": ingestion["parsed_response"],
            "provider_response_sha256": ingestion[
                "parsed_response_sha256"
            ],
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_receipt_sha256": result[
                "provider_receipt_sha256"
            ],
            "provider_identity_sha256": result[
                "provider_identity_sha256"
            ],
        }
        return {
            **body,
            "completion_sha256": self._wire_hash(body),
        }

    def build_runner_provider_receipt_binding(
        self, action, result, *, member_name
    ):
        binding = super().build_runner_provider_receipt_binding(
            action,
            result,
            member_name=member_name,
        )
        ingestion = self.ingest_runner_provider_response(
            action,
            result["provider_response"],
            member_name="ingest_runner_provider_response",
        )
        return {
            **binding,
            "provider_response_sha256": ingestion[
                "parsed_response_sha256"
            ],
        }

    def prepare_runner_normalization_request(
        self, action, *, member_name
    ):
        assert member_name == "prepare_runner_normalization_request"
        request_body = {
            "model": "fixture",
            "messages": [],
            "response_format": {},
            "temperature": 0,
        }
        body = {
            "schema_version": "model-runner-normalization-dispatch:v1",
            "action_sha256": action["action_sha256"],
            "provider": "openrouter",
            "method": "POST",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "credential_binding": {
                "header_name": "Authorization",
                "scheme": "Bearer",
                "source": "host_openrouter_credential",
                "persist": False,
            },
            "static_headers": {"Content-Type": "application/json"},
            "body": request_body,
            "body_sha256": self._wire_hash(request_body),
            "call_cap": 1,
            "credit_cap": 1.0,
            "timeout_seconds": 120.0,
            "max_response_bytes": 100_000,
        }
        return {**body, "request_sha256": self._wire_hash(body)}


class _CurrentOfficialTransport(_OfficialTransport):
    def runner_protocol_generation(self, *, release_identity):
        assert release_identity == _release()
        return runner_declaration(
            "v4",
            contract_hash=HASH["contract"],
            official_baseline=True,
        )

    def prepare_runner_provider_request(self, action, *, member_name):
        assert member_name == "prepare_runner_provider_request"
        request = {
            "credential_binding": {
                "source": "host_fixture_credential",
                "persist": False,
            }
        }
        body = {
            "schema_version": "model-runner-provider-dispatch:v1",
            "action_sha256": action["action_sha256"],
            "action_type": action["action_type"],
            "tool_id": action["tool_id"],
            "compiler_id": "fixture.compiler:v1",
            "compiler_contract_sha256": action[
                "binding_contract_sha256"
            ],
            "provider": "fixture",
            "request": request,
            "request_sha256": self._wire_hash(request),
            "response_contract": {},
            "budgets": {},
            "idempotency_key": "model-action:" + action["action_sha256"],
        }
        return {**body, "dispatch_sha256": self._wire_hash(body)}

    def build_runner_completion(self, action, result, *, member_name):
        assert member_name == "build_runner_completion"
        ingestion = self.ingest_runner_provider_response(
            action,
            result["provider_response"],
            member_name="ingest_runner_provider_response",
        )
        assert result["provider_response_ingestion"] == ingestion
        body = {
            "schema_version": "model-runner-completion:v4",
            "action_sha256": action["action_sha256"],
            "outcome": result["outcome"],
            "reason_code": result["reason_code"],
            "provider_response": ingestion["parsed_response"],
            "provider_response_sha256": ingestion[
                "parsed_response_sha256"
            ],
            "provider_response_ingestion_sha256": ingestion[
                "ingestion_sha256"
            ],
            "provider_dispatch_sha256": ingestion["dispatch_sha256"],
            "provider_request_sha256": ingestion["request_sha256"],
            "host_provider_response_sha256": ingestion[
                "host_response_sha256"
            ],
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_receipt_sha256": result[
                "provider_receipt_sha256"
            ],
            "provider_identity_sha256": result[
                "provider_identity_sha256"
            ],
            "calls": result["calls"],
            "cost_credits": result["cost_credits"],
            "latency_ms": result["latency_ms"],
        }
        return {**body, "completion_sha256": self._wire_hash(body)}

    def build_runner_provider_receipt_binding(
        self, action, result, *, member_name
    ):
        assert member_name == "build_runner_provider_receipt_binding"
        ingestion = self.ingest_runner_provider_response(
            action,
            result["provider_response"],
            member_name="ingest_runner_provider_response",
        )
        assert result["provider_response_ingestion"] == ingestion
        body = {
            "schema_version": "model-provider-receipt-binding:v2",
            "action_sha256": action["action_sha256"],
            "provider_response_ingestion_sha256": ingestion[
                "ingestion_sha256"
            ],
            "provider_response_sha256": ingestion[
                "parsed_response_sha256"
            ],
            "provider_dispatch_sha256": ingestion["dispatch_sha256"],
            "provider_request_sha256": ingestion["request_sha256"],
            "host_provider_response_sha256": ingestion[
                "host_response_sha256"
            ],
            "provider_receipt_ref": result["provider_receipt_ref"],
            "provider_identity_sha256": result[
                "provider_identity_sha256"
            ],
        }
        return {**body, "receipt_sha256": self._wire_hash(body)}


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


def _official_registration() -> ExactModelRunnerRegistration:
    registration = _registration()
    return ExactModelRunnerRegistration(
        artifact_identity=registration.artifact_identity,
        protocol=ResearchLabModelRunnerProtocol(
            transport=_OfficialTransport(),
            expected_release_identity=_release(),
        ),
        host_capability_manifest=registration.host_capability_manifest,
    )


def _current_official_registration() -> ExactModelRunnerRegistration:
    registration = _registration()
    return ExactModelRunnerRegistration(
        artifact_identity=registration.artifact_identity,
        protocol=ResearchLabModelRunnerProtocol(
            transport=_CurrentOfficialTransport(),
            expected_release_identity=_release(),
        ),
        host_capability_manifest=registration.host_capability_manifest,
    )


class _Projector:
    def __init__(self, registration):
        self._release_identity = dict(registration.protocol.release_identity)
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

        def wire_hash(value):
            return hashlib.sha256(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()

        body = {
            "schema_version": OFFICIAL_BASELINE_PROJECTION_SCHEMA_VERSION,
            "start_request_sha256": wire_hash(dict(start_request)),
            "release_identity_sha256": wire_hash(self._release_identity),
            "model_receipt_sha256": wire_hash(
                dict(terminal_result["model_receipt"])
            ),
            "companies": outputs,
            "companies_sha256": wire_hash(outputs),
        }
        return ArtifactBenchmarkProjection(
            outputs=tuple(outputs),
            projection_receipt={
                **body,
                "projection_sha256": wire_hash(body),
            },
        )


class _Transitions:
    def __init__(self, generation):
        self.generation = generation
        self.values = {}

    def resolve_run_protocol_generation(self, **_identity):
        return self.generation

    def load_model_transition(self, **identity):
        return self.values.get(
            (identity["unit_ref"], identity["idempotency_key"])
        )

    def append_model_transition(self, **value):
        self.values[(value["unit_ref"], value["action"]["idempotency_key"])] = {
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
        host_response = {
            "schema_version": "host-provider-response:v1",
            "provider": "fixture",
            "status_code": 200,
            "body": {"records": []},
        }
        ingestion = _OfficialTransport().ingest_runner_provider_response(
            action,
            host_response,
            member_name="ingest_runner_provider_response",
        )
        return ProtectedModelActionResult(
            host_result=HostActionResult(
                outcome="succeeded",
                reason_code="fixture",
                provider_response=host_response,
                calls=1,
                cost_credits=0.00001,
                latency_ms=20,
                provider_receipt_ref=receipt.receipt_ref,
                provider_identity_sha256="c" * 64,
            ),
            provider_receipt=receipt,
            replay_ref={"protected_job_ref": "fixture-job"},
            model_provider_response_ingestion=ingestion,
        )

    def replay_provider_action(self, **_values):
        raise AssertionError("full transition replay does not redispatch")

    def verify_company_action(self, **_values):
        raise AssertionError("not called")

    verify_intent_action = verify_company_action
    verify_contact_action = verify_company_action


class _CurrentDispatcher(_Dispatcher):
    def __init__(self):
        super().__init__()
        self.compiled_dispatches = []

    def dispatch_provider_action(
        self, *, action, unit_ref, compiled_dispatch=None, **values
    ):
        assert isinstance(compiled_dispatch, dict)
        self.compiled_dispatches.append(deepcopy(compiled_dispatch))
        protected = super().dispatch_provider_action(
            action=action,
            unit_ref=unit_ref,
            **values,
        )
        ingestion = _CurrentOfficialTransport().ingest_runner_provider_response(
            action,
            protected.host_result.provider_response,
            member_name="ingest_runner_provider_response",
        )
        return replace(
            protected,
            model_provider_response_ingestion=ingestion,
            host_result=replace(
                protected.host_result,
                model_provider_response_ingestion=ingestion,
                provider_action_receipt_sha256="d" * 64,
            ),
        )


class _ProtectedAuthority:
    authority_identity_sha256 = "sha256:" + HASH["authority"]

    def __init__(self, registration, *, current=False):
        generation = registration.protocol_generation.protocol_generation_sha256
        self.transitions = _Transitions(generation)
        self.dispatcher = _CurrentDispatcher() if current else _Dispatcher()
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
        for (unit_ref, _idempotency_key), stored in self.transitions.values.items():
            if unit_ref != completion["unit_ref"]:
                continue
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


def _exact_fixture(*, current=False):
    registration = (
        _current_official_registration()
        if current
        else _official_registration()
    )
    projector = _Projector(registration)
    authority = _ProtectedAuthority(registration, current=current)
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
        manifest_uri="s3://fixture/model/" + "1" * 40 + ".json",
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
    runner, _projector, _authority, terminal = _exact_fixture()
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
    assert {
        value["start_request"]["execution_mode"]
        for value in terminal.records.values()
    } == {"full_company"}
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


@pytest.mark.parametrize(
    "incomplete_marker",
    (
        "model_runner_incomplete:run_budget_exhausted",
        "model_runner_incomplete:required_provider_failure",
    ),
)
@pytest.mark.asyncio
async def test_scoring_worker_retries_exact_model_incomplete_outcome(
    monkeypatch,
    incomplete_marker: str,
):
    runner, _projector, _authority, _terminal = _exact_fixture()
    worker = object.__new__(scoring_worker_module.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "test-worker"
    worker.config = SimpleNamespace(private_baseline_provider_retry_rounds=2)
    worker._active_baseline_context = {}

    async def unchanged(**_values):
        return None

    async def no_traces(**_values):
        return None

    attempt_ordinals = []

    def incomplete_outcome(*_args, **kwargs):
        attempt_ordinals.append(kwargs.get("attempt_ordinal"))
        raise PrivateModelRuntimeError(incomplete_marker)

    worker._ensure_private_baseline_repo_head_unchanged = unchanged
    worker._record_baseline_icp_traces = no_traces
    monkeypatch.setattr(
        ExactOfficialBaselineRunner,
        "run_icp",
        incomplete_outcome,
    )
    item = {
        "icp_ref": "icp-incomplete-budget",
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
            retry_round=1,
        )

    assert attempt_ordinals == [1]
    assert row["_retryable"] is True
    assert row["_nonempty"] is False
    assert row["diagnostics"]["sourcing_failed"] is True
    assert scoring_worker_module._OFFICIAL_BASELINE_CHECKPOINT_FIELD not in row


@pytest.mark.asyncio
async def test_scoring_worker_retries_terminal_uncertain_with_fresh_attempt(
    monkeypatch,
):
    runner, _projector, _authority, _terminal = _exact_fixture()
    worker = object.__new__(scoring_worker_module.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "test-worker"
    worker.config = SimpleNamespace(private_baseline_provider_retry_rounds=2)
    worker._active_baseline_context = {}

    async def unchanged(**_values):
        return None

    async def no_traces(**_values):
        return None

    attempt_ordinals = []

    def terminal_uncertain(*_args, **kwargs):
        attempt_ordinals.append(kwargs.get("attempt_ordinal"))
        raise OfficialBaselineTerminalUncertainError(
            "official baseline protected call is terminal uncertain"
        )

    worker._ensure_private_baseline_repo_head_unchanged = unchanged
    worker._record_baseline_icp_traces = no_traces
    monkeypatch.setattr(
        ExactOfficialBaselineRunner,
        "run_icp",
        terminal_uncertain,
    )
    item = {
        "icp_ref": "icp-terminal-uncertain",
        "icp_hash": "sha256:" + "2" * 64,
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
            benchmark_date="2026-08-27",
            retry_round=0,
        )

    assert attempt_ordinals == [0]
    assert row["_retryable"] is True
    assert row["_nonempty"] is False
    assert row["diagnostics"]["sourcing_failed"] is True
    assert row["_runtime_error"].startswith(
        "OfficialBaselineTerminalUncertainError:"
    )
    assert scoring_worker_module._OFFICIAL_BASELINE_CHECKPOINT_FIELD not in row


@pytest.mark.asyncio
async def test_terminal_uncertain_advances_to_fresh_bounded_attempt(
    monkeypatch,
):
    runner, _projector, _authority, _terminal = _exact_fixture()
    runner = runner.with_spec(
        replace(
            runner.spec,
            extra_env={
                scoring_worker_module.PROVIDER_COST_EVALUATION_SCOPE_ENV: (
                    "sha256:" + "7" * 64
                )
            },
        )
    )
    worker = object.__new__(scoring_worker_module.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "test-worker"
    worker.config = SimpleNamespace(
        private_baseline_concurrency=1,
        private_baseline_retry_concurrency=1,
        private_baseline_provider_retry_rounds=1,
        scoring_worker_total_workers=1,
    )
    worker._active_baseline_context = {}

    async def unchanged(**_values):
        return None

    async def no_traces(**_values):
        return None

    async def maintenance_state():
        return {"paused": False}

    original_run_icp = ExactOfficialBaselineRunner.run_icp
    attempt_ordinals = []

    def recover_on_fresh_attempt(self, *_args, **kwargs):
        attempt_ordinal = kwargs.get("attempt_ordinal")
        attempt_ordinals.append(attempt_ordinal)
        if attempt_ordinal == 0:
            raise OfficialBaselineTerminalUncertainError(
                "official baseline protected call is terminal uncertain"
            )
        return original_run_icp(self, *_args, **kwargs)

    worker._ensure_private_baseline_repo_head_unchanged = unchanged
    worker._record_baseline_icp_traces = no_traces
    monkeypatch.setattr(
        ExactOfficialBaselineRunner,
        "run_icp",
        recover_on_fresh_attempt,
    )
    monkeypatch.setattr(
        scoring_worker_module,
        "get_scoring_maintenance_state",
        maintenance_state,
    )
    monkeypatch.setattr(
        scoring_worker_module,
        "_apply_provider_cost_baseline_outcome",
        lambda _row: None,
    )
    window = SimpleNamespace(
        benchmark_items=[
            {
                "icp_ref": "icp-terminal-uncertain",
                "icp_hash": "sha256:" + "2" * 64,
                "set_id": "set-1",
                "day_index": 0,
                "day_rank": 1,
                "icp": {"outputs": [], "max_companies": 1},
            }
        ]
    )

    rows, stats = await worker._run_baseline_batch_inner(
        runner=runner,
        retry_runner=runner,
        scorer=object(),
        window=window,
        run_start=0.0,
        benchmark_date="2026-08-27",
    )

    assert attempt_ordinals == [0, 1]
    assert stats == {"retried": 1, "recovered": 1, "unresolved": 0}
    assert rows[0]["_retryable"] is False
    assert rows[0]["diagnostics"]["sourcing_failed"] is False
    assert scoring_worker_module._OFFICIAL_BASELINE_CHECKPOINT_FIELD in rows[0]


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

    expected_round_zero_ref = "baseline_icp:" + sha256_json(
        {
            "run_sha256": runner.run_sha256,
            "icp_ref": "icp-restart",
            "raw_icp_sha256": sha256_json(raw),
        }
    ).removeprefix("sha256:")
    assert first.checkpoint["unit_ref"] == expected_round_zero_ref
    assert second.checkpoint == first.checkpoint
    assert second.replayed_transition_count == 1
    assert authority.dispatcher.provider_calls == 1
    assert len(terminal.records) == 1

    retry = runner.run_icp(
        raw_icp=raw,
        icp_ref="icp-restart",
        target_count=1,
        attempt_ordinal=1,
    )
    retry_after_restart = runner.run_icp(
        raw_icp=raw,
        icp_ref="icp-restart",
        target_count=1,
        attempt_ordinal=1,
        expected_checkpoint=retry.checkpoint,
    )

    assert retry.checkpoint != first.checkpoint
    assert retry_after_restart.checkpoint == retry.checkpoint
    assert retry_after_restart.replayed_transition_count == 1
    assert authority.dispatcher.provider_calls == 2
    assert len(terminal.records) == 2


@pytest.mark.parametrize("attempt_ordinal", (-1, True, 1.0, "1"))
def test_exact_official_baseline_rejects_invalid_attempt_ordinal(attempt_ordinal):
    runner, _projector, _authority, _terminal = _exact_fixture()

    with pytest.raises(
        OfficialBaselineModelError,
        match="attempt ordinal is invalid",
    ):
        runner.run_icp(
            raw_icp={"outputs": []},
            icp_ref="icp-invalid-attempt",
            target_count=1,
            attempt_ordinal=attempt_ordinal,
        )


def test_current_official_baseline_uses_compiled_dispatch_and_v4_custody():
    runner, _projector, authority, terminal = _exact_fixture(current=True)
    raw = {"requires_action": True, "outputs": [_company()]}

    first = runner.run_icp(
        raw_icp=raw,
        icp_ref="icp-current-custody",
        target_count=1,
    )
    second = runner.run_icp(
        raw_icp=raw,
        icp_ref="icp-current-custody",
        target_count=1,
        expected_checkpoint=first.checkpoint,
    )

    assert second.checkpoint == first.checkpoint
    assert second.replayed_transition_count == 1
    assert authority.dispatcher.provider_calls == 1
    assert len(authority.dispatcher.compiled_dispatches) == 1
    compiled = authority.dispatcher.compiled_dispatches[0]
    assert compiled["schema_version"] == "model-runner-provider-dispatch:v1"
    assert compiled["request"]["credential_binding"]["persist"] is False
    transition = next(iter(authority.transitions.values.values()))
    assert transition["completion"]["schema_version"] == (
        "model-runner-completion:v4"
    )
    assert transition["completion"][
        "provider_response_ingestion_sha256"
    ]
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
        manifest_uri="s3://fixture/model/" + "1" * 40 + ".json",
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


def test_official_baseline_generation_bundle_is_atomic_and_pinned():
    old = ArtifactRunnerProtocolGeneration.from_declaration(
        runner_declaration("v3", contract_hash=HASH["contract"]),
        expected_consumer_contract_sha256=HASH["contract"],
    )
    declaration = runner_declaration(
        "v3",
        contract_hash=HASH["contract"],
        official_baseline=True,
    )
    official = ArtifactRunnerProtocolGeneration.from_declaration(
        declaration,
        expected_consumer_contract_sha256=HASH["contract"],
    )

    assert old.supports_official_baseline is False
    assert official.supports_official_baseline is True
    assert old.protocol_generation_sha256 != (
        official.protocol_generation_sha256
    )
    assert official.member("provider_prepare") == (
        "prepare_runner_provider_request"
    )
    assert official.member("provider_response_ingestion") == (
        "ingest_runner_provider_response"
    )
    assert official.member("verifier_execution") == (
        "execute_runner_verifier_action"
    )
    assert official.member("official_host_binding_catalog") == (
        "runner_official_host_binding_catalog"
    )
    assert official.official_contract_sha256(
        "benchmark_projection_contract"
    ).startswith("sha256:")

    partial = deepcopy(declaration)
    partial["champion_execution"].pop("verifier_execution_entrypoint")
    without_legacy_entrypoint = (
        ArtifactRunnerProtocolGeneration.from_declaration(
            partial,
            expected_consumer_contract_sha256=HASH["contract"],
        )
    )
    assert without_legacy_entrypoint.member("verifier_execution") == (
        "execute_runner_verifier_action"
    )
    assert without_legacy_entrypoint.protocol_generation_sha256 != (
        official.protocol_generation_sha256
    )

    tampered = deepcopy(declaration)
    tampered["champion_execution"]["benchmark_projection_contract"][
        "contract_fields"
    ] = ["tampered"]
    with pytest.raises(ModelRunnerHostError, match="hash differs"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            tampered,
            expected_consumer_contract_sha256=HASH["contract"],
        )

    missing_normalization_member = deepcopy(declaration)
    missing_normalization_member["champion_execution"][
        "normalization_action"
    ].pop("dispatch_entrypoint")
    with pytest.raises(
        ModelRunnerHostError,
        match="normalization action identity differs",
    ):
        ArtifactRunnerProtocolGeneration.from_declaration(
            missing_normalization_member,
            expected_consumer_contract_sha256=HASH["contract"],
        )

    tampered_proof_identity = deepcopy(declaration)
    tampered_proof_identity["champion_execution"][
        "company_fit_proof_contract_sha256"
    ] = "sha256:" + "f" * 64
    with pytest.raises(ModelRunnerHostError, match="is invalid"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            tampered_proof_identity,
            expected_consumer_contract_sha256=HASH["contract"],
        )


def test_provider_response_ingestion_frozen_interface_identity_matches_model():
    declaration = runner_declaration(
        "v3",
        contract_hash=HASH["contract"],
        official_baseline=True,
    )
    role = declaration["champion_execution"]["runner_role_contract"][
        "roles"
    ]["provider_response_ingestion"]
    signature = {
        "consumer_contract_id": "leadpoet-sourcing-wrapper-contract-v73",
        "consumer_contract_path": (
            "research_lab_adapter.py:ingest_runner_provider_response"
        ),
        "positional_parameters": ["action", "host_response"],
        "full_parameters": ["action", "host_response"],
        "required_keyword_only": [],
        "is_async": False,
    }

    assert role["interface_contract_sha256"] == (
        "41708b700cbb8af11c28cdd9556ff70f7eed6676cf30a0d79cba3c0e6c424162"
    )
    assert _bare_wire_hash(signature) == (
        "356052075d056fa1f14eb168a52c1ba80a2db203946edbf5c8656adaa7cafc4f"
    )


def test_missing_or_incompatible_provider_ingestion_fails_generation_admission():
    declaration = runner_declaration(
        "v3",
        contract_hash=HASH["contract"],
        official_baseline=True,
    )
    missing = deepcopy(declaration)
    role_contract = missing["champion_execution"]["runner_role_contract"]
    role_contract["roles"].pop("provider_response_ingestion")
    profile = role_contract["activation_profiles"]["full_company"]
    profile["required_roles"].remove("provider_response_ingestion")
    profile["minimum_interface_major"].pop(
        "provider_response_ingestion"
    )
    _rehash_role_contract(missing)

    with pytest.raises(ModelRunnerHostError, match="requirements differ"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            missing,
            expected_consumer_contract_sha256=HASH["contract"],
        )

    incompatible = deepcopy(declaration)
    contract = incompatible["champion_execution"][
        "provider_response_ingestion_contract"
    ]
    contract["completion_input"] = "consumer_projected_response"
    contract["contract_sha256"] = _bare_wire_hash({
        key: value
        for key, value in contract.items()
        if key != "contract_sha256"
    })
    with pytest.raises(ModelRunnerHostError, match="contract differs"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            incompatible,
            expected_consumer_contract_sha256=HASH["contract"],
        )


def test_current_ingestion_custody_generation_is_admitted_without_commit_allowlist():
    declaration = runner_declaration(
        "v4",
        contract_hash=HASH["contract"],
        official_baseline=True,
    )
    generation = ArtifactRunnerProtocolGeneration.from_declaration(
        declaration,
        expected_consumer_contract_sha256=HASH["contract"],
    )

    assert generation.family == "model-runner-protocol:v3"
    assert generation.requires_raw_provider_response_custody is True
    assert generation.version("MODEL_RUNNER_COMPLETION_SCHEMA_VERSION") == (
        "model-runner-completion:v4"
    )
    assert generation.member("completion") == "build_runner_completion"
    assert generation.member("provider_receipt_binding") == (
        "build_runner_provider_receipt_binding"
    )

    additive = deepcopy(declaration)
    ingestion_contract = additive["champion_execution"][
        "provider_response_ingestion_contract"
    ]
    ingestion_contract["leadpoet.future_optional"] = {"enabled": True}
    ingestion_contract["contract_sha256"] = _bare_wire_hash({
        key: value
        for key, value in ingestion_contract.items()
        if key != "contract_sha256"
    })
    additive_generation = ArtifactRunnerProtocolGeneration.from_declaration(
        additive,
        expected_consumer_contract_sha256=HASH["contract"],
    )

    assert additive_generation.requires_raw_provider_response_custody is True
    assert additive_generation.protocol_generation_sha256 != (
        generation.protocol_generation_sha256
    )


@pytest.mark.parametrize(
    "failure",
    (
        "compatibility_major",
        "completion_major",
        "receipt_binding_major",
        "ingestion_custody_contract",
    ),
)
def test_current_role_or_custody_downgrade_fails_before_provider_spend(failure):
    declaration = runner_declaration(
        "v4",
        contract_hash=HASH["contract"],
        official_baseline=True,
    )
    role_contract = declaration["champion_execution"][
        "runner_role_contract"
    ]
    if failure == "compatibility_major":
        role_contract["compatibility_major"] = 1
        _rehash_role_contract(declaration)
    elif failure in {"completion_major", "receipt_binding_major"}:
        role = (
            "completion"
            if failure == "completion_major"
            else "provider_receipt_binding"
        )
        entry = role_contract["roles"][role]
        entry["interface_major"] = 1
        entry["interface_contract"]["interface_major"] = 1
        entry["interface_contract_sha256"] = _bare_wire_hash(
            entry["interface_contract"]
        )
        role_contract["activation_profiles"]["full_company"][
            "minimum_interface_major"
        ][role] = 1
        _rehash_role_contract(declaration)
    else:
        contract = declaration["champion_execution"][
            "provider_response_ingestion_contract"
        ]
        contract.pop("ingestion_receipt_required_for_response")
        contract["completion_input"] = "original_unchanged_host_response"
        contract["provider_receipt_binding_input"] = (
            "original_unchanged_host_response"
        )
        contract["custody_join"] = [
            "action_sha256",
            "host_response_sha256",
            "parsed_response_sha256",
        ]
        contract["contract_sha256"] = _bare_wire_hash({
            key: value
            for key, value in contract.items()
            if key != "contract_sha256"
        })

    with pytest.raises(ModelRunnerHostError):
        ArtifactRunnerProtocolGeneration.from_declaration(
            declaration,
            expected_consumer_contract_sha256=HASH["contract"],
        )


def test_durable_provider_ingestion_must_replay_byte_identically():
    registration = _official_registration()
    action = _provider_action({})
    protected = _Dispatcher().dispatch_provider_action(
        action=action,
        unit_ref="baseline_icp:" + "a" * 64,
    )
    ingestion = dict(protected.model_provider_response_ingestion)
    ingestion["ingestion_sha256"] = "0" * 64

    with pytest.raises(
        CommonModelExperimentError,
        match="differs from replay",
    ):
        _bind_durable_provider_result(
            protocol=registration.protocol,
            action=action,
            protected=replace(
                protected,
                model_provider_response_ingestion=ingestion,
            ),
        )


def test_role_map_discovers_renamed_member_without_commit_allowlist():
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    role = "provider_compiler_inventory"
    old_member = "model_runner_provider_compiler_inventory"
    new_member = "artifact_inventory_v2"
    consumer = declaration["consumer_contract"]
    consumer["functions"][new_member] = consumer["functions"].pop(old_member)
    consumer["full_parameters"][new_member] = consumer[
        "full_parameters"
    ].pop(old_member)
    consumer["frozen_asyncness"][new_member] = consumer[
        "frozen_asyncness"
    ].pop(old_member)
    old_path = "research_lab_adapter.py:" + old_member
    # The exact path/member tuple is owned by this signed consumer contract;
    # compatibility is not tied to a Leadpoet hard-coded adapter path.
    new_path = "compat/research_lab_adapter.py:" + new_member
    consumer["exact_signatures"] = [
        new_path if item == old_path else item
        for item in consumer["exact_signatures"]
    ]
    entry = declaration["champion_execution"]["runner_role_contract"][
        "roles"
    ][role]
    entry["adapter_member"] = new_member
    entry["consumer_signature"]["consumer_contract_path"] = new_path
    entry["consumer_signature_sha256"] = _bare_wire_hash(
        entry["consumer_signature"]
    )
    _rehash_role_contract(declaration)

    generation = ArtifactRunnerProtocolGeneration.from_declaration(
        declaration,
        expected_consumer_contract_sha256="e" * 64,
    )

    assert generation.member(role) == new_member


def test_compatible_role_interfaces_admit_distinct_artifact_identities():
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )

    class _ArtifactTransport(_OfficialTransport):
        def __init__(self, release):
            self.release = deepcopy(release)

        def runner_protocol_generation(self, *, release_identity):
            assert release_identity == self.release
            return deepcopy(declaration)

    generations = []
    keys = []
    for commit_marker, artifact_marker, manifest_marker in (
        ("1", "a", "8"),
        ("2", "b", "9"),
    ):
        release = {
            **_release(),
            "source_commit": commit_marker * 40,
            "model_artifact_digest": "sha256:" + artifact_marker * 64,
            "consumer_contract_sha256": "e" * 64,
        }
        identity = {
            "repository": "leadpoet/Sourcing_model",
            "branch": "main",
            "commit_sha": release["source_commit"],
            "model_artifact_hash": release["model_artifact_digest"],
            "manifest_hash": "sha256:" + manifest_marker * 64,
            "routing_contract_hash": "sha256:" + "e" * 64,
            "routing_catalog_hash": "sha256:" + HASH["catalog"],
            "routing_policy_hash": "sha256:" + HASH["policy"],
            "feature_schema_hash": "sha256:" + HASH["feature"],
        }
        registration = ExactModelRunnerRegistration(
            artifact_identity=identity,
            protocol=ResearchLabModelRunnerProtocol(
                transport=_ArtifactTransport(release),
                expected_release_identity=release,
            ),
            host_capability_manifest={"bindings": []},
        )
        generations.append(
            registration.protocol_generation.protocol_generation_sha256
        )
        keys.append(registration.key)

    assert len(set(keys)) == 2
    assert len(set(generations)) == 1


def test_additive_optional_role_profile_metadata_and_source_are_hash_bound():
    baseline = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    declaration = deepcopy(baseline)
    role = "future_optional_probe"
    member = "future_optional_probe_member"
    consumer = declaration["consumer_contract"]
    consumer["functions"][member] = []
    consumer["full_parameters"][member] = []
    consumer["frozen_asyncness"][member] = True
    consumer["exact_signatures"].append(
        "research_lab_adapter.py:" + member
    )
    interface = {
        "interface_id": "leadpoet.model_runner." + role,
        "interface_major": 2,
        "positional_parameters": [],
        "host_keyword_parameters": [],
        "required_keyword_only": [],
        "is_async": True,
    }
    signature = {
        "consumer_contract_id": consumer["contract_id"],
        "consumer_contract_path": "research_lab_adapter.py:" + member,
        "positional_parameters": [],
        "full_parameters": [],
        "required_keyword_only": [],
        "is_async": True,
    }
    role_contract = declaration["champion_execution"][
        "runner_role_contract"
    ]
    role_contract["roles"][role] = {
        "interface_id": interface["interface_id"],
        "interface_major": 2,
        "interface_contract": interface,
        "interface_contract_sha256": _bare_wire_hash(interface),
        "adapter_member": member,
        "consumer_signature": signature,
        "consumer_signature_sha256": _bare_wire_hash(signature),
        "required_for_profiles": [],
    }
    role_contract["activation_profiles"]["future.optional"] = {
        "required_roles": [role]
    }
    role_contract["extensions"] = {
        "leadpoet.future": {"enabled": True}
    }
    consumer["extensions"] = {
        "leadpoet.future": {"metadata": "ignored-and-hash-bound"}
    }
    declaration["champion_execution"]["future_optional_metadata"] = {
        "schema_version": "future:v1"
    }
    declaration["champion_execution"]["raw_icp_source_schemas"].append(
        "leadpoet-future-source:v1"
    )
    _rehash_role_contract(declaration)

    old = ArtifactRunnerProtocolGeneration.from_declaration(
        baseline,
        expected_consumer_contract_sha256="e" * 64,
    )
    new = ArtifactRunnerProtocolGeneration.from_declaration(
        declaration,
        expected_consumer_contract_sha256="e" * 64,
    )

    assert role not in new.members
    assert new.protocol_generation_sha256 != old.protocol_generation_sha256
    assert "leadpoet-future-source:v1" in new.raw_source_schemas


def test_signed_optional_consumer_parameter_preserves_stable_interface():
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    role = "provider_prepare"
    entry = declaration["champion_execution"]["runner_role_contract"][
        "roles"
    ][role]
    member = entry["adapter_member"]
    declaration["consumer_contract"]["full_parameters"][member].append(
        "future_optional"
    )
    entry["consumer_signature"]["full_parameters"].append(
        "future_optional"
    )
    entry["consumer_signature_sha256"] = _bare_wire_hash(
        entry["consumer_signature"]
    )
    _rehash_role_contract(declaration)

    generation = ArtifactRunnerProtocolGeneration.from_declaration(
        declaration,
        expected_consumer_contract_sha256="e" * 64,
    )

    assert generation.member(role) == member


def test_signed_defaulted_positional_parameter_preserves_stable_interface():
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    _upgrade_role_contract_v2(declaration)
    role = "provider_prepare"
    entry = declaration["champion_execution"]["runner_role_contract"][
        "roles"
    ][role]
    member = entry["adapter_member"]
    # The stable interface still requires only ``action``. The exact signed
    # artifact signature adds a trailing positional parameter, so the model's
    # role contract declares that parameter defaulted for the stable host call.
    declaration["consumer_contract"]["functions"][member].append(
        "future_optional"
    )
    declaration["consumer_contract"]["full_parameters"][member].append(
        "future_optional"
    )
    entry["consumer_signature"]["positional_parameters"].append(
        "future_optional"
    )
    entry["consumer_signature"]["defaulted_positional_parameters"].append(
        "future_optional"
    )
    entry["consumer_signature"]["full_parameters"].append(
        "future_optional"
    )
    entry["consumer_signature_sha256"] = _bare_wire_hash(
        entry["consumer_signature"]
    )
    _rehash_role_contract(declaration)

    generation = ArtifactRunnerProtocolGeneration.from_declaration(
        declaration,
        expected_consumer_contract_sha256="e" * 64,
    )

    assert generation.member(role) == member


def test_legacy_role_contract_rejects_unsigned_trailing_positional_parameter():
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    role = "provider_prepare"
    entry = declaration["champion_execution"]["runner_role_contract"][
        "roles"
    ][role]
    member = entry["adapter_member"]
    declaration["consumer_contract"]["functions"][member].append(
        "future_optional"
    )
    declaration["consumer_contract"]["full_parameters"][member].append(
        "future_optional"
    )
    entry["consumer_signature"]["positional_parameters"].append(
        "future_optional"
    )
    entry["consumer_signature"]["full_parameters"].append(
        "future_optional"
    )
    entry["consumer_signature_sha256"] = _bare_wire_hash(
        entry["consumer_signature"]
    )
    _rehash_role_contract(declaration)

    with pytest.raises(ModelRunnerHostError, match="host call differs"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            declaration,
            expected_consumer_contract_sha256="e" * 64,
        )


def test_signed_required_trailing_positional_fails_before_provider_spend():
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    _upgrade_role_contract_v2(declaration)
    role = "provider_response_ingestion"
    entry = declaration["champion_execution"]["runner_role_contract"][
        "roles"
    ][role]
    member = entry["adapter_member"]
    for collection in (
        declaration["consumer_contract"]["functions"][member],
        declaration["consumer_contract"]["full_parameters"][member],
        entry["consumer_signature"]["positional_parameters"],
        entry["consumer_signature"]["required_positional_parameters"],
        entry["consumer_signature"]["full_parameters"],
    ):
        collection.append("required_after_spend")
    entry["consumer_signature_sha256"] = _bare_wire_hash(
        entry["consumer_signature"]
    )
    _rehash_role_contract(declaration)

    with pytest.raises(ModelRunnerHostError, match="host call differs"):
        ArtifactRunnerProtocolGeneration.from_declaration(
            declaration,
            expected_consumer_contract_sha256="e" * 64,
        )


@pytest.mark.parametrize("failure", ("top_level", "unnamespaced_extension"))
def test_consumer_contract_additions_are_bounded_to_namespaced_extensions(
    failure,
):
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    consumer = declaration["consumer_contract"]
    if failure == "top_level":
        consumer["future_semantics"] = {"enabled": True}
    else:
        consumer["extensions"]["future"] = {"enabled": True}

    with pytest.raises(ModelRunnerHostError):
        ArtifactRunnerProtocolGeneration.from_declaration(
            declaration,
            expected_consumer_contract_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    "failure",
    (
        "missing",
        "major",
        "unknown_required",
        "renamed_host_parameter",
        "new_required_positional_parameter",
        "new_required_parameter",
        "async_kind",
    ),
)
def test_required_role_drift_fails_closed(failure):
    declaration = runner_declaration(
        "v3", contract_hash="e" * 64, official_baseline=True
    )
    role_contract = declaration["champion_execution"][
        "runner_role_contract"
    ]
    if failure == "missing":
        role_contract["roles"].pop("provider_prepare")
    elif failure == "major":
        entry = role_contract["roles"]["provider_prepare"]
        entry["interface_major"] = 2
        entry["interface_contract"]["interface_major"] = 2
        entry["interface_contract_sha256"] = _bare_wire_hash(
            entry["interface_contract"]
        )
        role_contract["activation_profiles"]["full_company"][
            "minimum_interface_major"
        ]["provider_prepare"] = 2
    elif failure == "unknown_required":
        source = deepcopy(role_contract["roles"]["provider_prepare"])
        source["interface_id"] = "leadpoet.model_runner.future_required"
        source["interface_contract"]["interface_id"] = source[
            "interface_id"
        ]
        source["interface_contract_sha256"] = _bare_wire_hash(
            source["interface_contract"]
        )
        source["required_for_profiles"] = ["full_company"]
        role_contract["roles"]["future_required"] = source
        profile = role_contract["activation_profiles"]["full_company"]
        profile["required_roles"] = sorted(
            [*profile["required_roles"], "future_required"]
        )
        profile["minimum_interface_major"]["future_required"] = 1
    else:
        entry = role_contract["roles"]["provider_prepare"]
        interface = entry["interface_contract"]
        signature = entry["consumer_signature"]
        member = entry["adapter_member"]
        consumer = declaration["consumer_contract"]
        if failure == "renamed_host_parameter":
            signature["positional_parameters"] = ["request"]
            signature["full_parameters"] = ["request"]
            consumer["functions"][member] = ["request"]
            consumer["full_parameters"][member] = ["request"]
        elif failure == "new_required_positional_parameter":
            interface["positional_parameters"].append("required_option")
            signature["positional_parameters"].append("required_option")
            signature["full_parameters"].append("required_option")
            consumer["functions"][member].append("required_option")
            consumer["full_parameters"][member].append("required_option")
        elif failure == "new_required_parameter":
            signature["full_parameters"].append("required_option")
            signature["required_keyword_only"] = ["required_option"]
            consumer["full_parameters"][member].append("required_option")
            consumer["required_keyword_only"][member] = [
                "required_option"
            ]
        else:
            interface["is_async"] = True
            signature["is_async"] = True
            consumer["frozen_asyncness"][member] = True
        entry["interface_contract_sha256"] = _bare_wire_hash(interface)
        entry["consumer_signature_sha256"] = _bare_wire_hash(signature)
    _rehash_role_contract(declaration)

    with pytest.raises(ModelRunnerHostError):
        ArtifactRunnerProtocolGeneration.from_declaration(
            declaration,
            expected_consumer_contract_sha256="e" * 64,
        )


def test_artifact_protocol_projector_uses_exact_generation_member():
    registration = _official_registration()
    projector = ArtifactProtocolBenchmarkProjector(registration)
    start = {"schema_version": "fixture-start:v1"}
    terminal = {
        "result": {"outputs": [_company()]},
        "model_receipt": {"schema_version": "fixture-receipt:v1"},
    }

    projected = projector.project_company_outputs(
        start_request=start,
        terminal_result=terminal,
    )

    assert projected.outputs == (_company(),)
    assert projected.projection_receipt["companies"] == [_company()]
    assert projector.projection_identity_sha256 == (
        registration.protocol_generation.official_contract_sha256(
            "benchmark_projection_contract"
        )
    )


def test_old_v3_drain_rejects_new_official_member_use():
    protocol = ResearchLabModelRunnerProtocol(
        transport=_Transport(),
        expected_release_identity=_release(),
    )
    assert protocol.artifact_official_baseline_supported is False
    with pytest.raises(
        ModelRunnerHostError,
        match="no official baseline bundle",
    ):
        protocol.provider_compiler_inventory()
    with pytest.raises(
        ModelRunnerHostError,
        match="no official baseline bundle",
    ):
        protocol.execute_verifier_action({
            "action_type": "verify_company",
            "action_sha256": "a" * 64,
        })


def test_official_catalog_manifest_and_verifier_are_artifact_owned():
    protocol = _official_registration().protocol

    catalog = protocol.official_host_binding_catalog()
    manifest = protocol.build_official_host_capability_manifest(
        {"verifier.company": True}
    )
    execution = protocol.execute_verifier_action({
        "action_type": "verify_company",
        "action_sha256": "a" * 64,
    })
    normalization = protocol.prepare_normalization_request({
        "action_type": "normalize_icp",
        "action_sha256": "b" * 64,
    })

    assert catalog["bindings"][0]["tool_id"] == "verifier.company"
    assert manifest["bindings"][0]["available"] is True
    assert execution["calls"] == 0
    assert execution["cost_credits"] == 0.0
    assert execution["provider_receipt_allowed"] is False
    assert normalization["provider"] == "openrouter"
    assert normalization["call_cap"] == 1


def test_official_verifier_rejects_tampered_artifact_execution():
    class _TamperedVerifierTransport(_OfficialTransport):
        def execute_runner_verifier_action(self, action, *, member_name):
            value = super().execute_runner_verifier_action(
                action, member_name=member_name
            )
            return {**value, "calls": 1}

    protocol = ResearchLabModelRunnerProtocol(
        transport=_TamperedVerifierTransport(),
        expected_release_identity=_release(),
    )
    with pytest.raises(ModelRunnerHostError, match="execution is invalid"):
        protocol.execute_verifier_action({
            "action_type": "verify_company",
            "action_sha256": "a" * 64,
        })


def test_exact_official_model_spec_exposes_no_proxy_or_credentials():
    spec = DockerPrivateModelSpec(
        image_digest="example.invalid/model@sha256:" + "a" * 64,
        env_passthrough=(
            "RESEARCH_LAB_OPENROUTER_API_KEY",
            "RESEARCH_LAB_EVIDENCE_PROXY_URL",
        ),
        extra_env={
            "RESEARCH_LAB_EVIDENCE_PROXY_URL": "http://127.0.0.1:8765",
            "RESEARCH_LAB_OPENROUTER_API_KEY": "never-forward",
            "HTTPS_PROXY": "http://127.0.0.1:9999",
            "MODEL_OPERATIONAL_SCOPE": "sha256:" + "b" * 64,
        },
    )

    exact = scoring_worker_module._exact_official_baseline_model_spec(spec)

    assert exact.env_passthrough == ()
    assert exact.extra_env == {
        "MODEL_OPERATIONAL_SCOPE": "sha256:" + "b" * 64
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", (False, True))
async def test_exact_proxy_lease_closes_on_success_and_failure(fail):
    class _Lease:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    worker = object.__new__(
        scoring_worker_module.ResearchLabGatewayScoringWorker
    )
    worker._active_official_baseline_evidence_proxy = None
    lease = _Lease()

    async def implementation():
        worker._active_official_baseline_evidence_proxy = lease
        if fail:
            raise RuntimeError("fixture failure")
        return {"status": "fixture success"}

    worker._maybe_run_private_baseline_impl = implementation
    if fail:
        with pytest.raises(RuntimeError, match="fixture failure"):
            await worker._maybe_run_private_baseline()
    else:
        assert await worker._maybe_run_private_baseline() == {
            "status": "fixture success"
        }
    assert lease.closed == 1
    assert worker._active_official_baseline_evidence_proxy is None


def test_exact_proxy_lease_shutdown_is_complete_and_idempotent():
    class _Server:
        def __init__(self):
            self.shutdowns = 0
            self.closes = 0

        def shutdown(self):
            self.shutdowns += 1

        def server_close(self):
            self.closes += 1

    class _Thread:
        def __init__(self):
            self.joins = []

        def join(self, timeout=None):
            self.joins.append(timeout)

        def is_alive(self):
            return False

    server = _Server()
    thread = _Thread()
    lease = scoring_worker_module._OfficialBaselineEvidenceProxyLease(
        server=server,
        thread=thread,
    )

    lease.close()
    lease.close()

    assert server.shutdowns == 1
    assert server.closes == 1
    assert thread.joins == [5.0]


def test_exact_proxy_lease_partial_shutdown_can_retry():
    class _Server:
        def __init__(self):
            self.shutdowns = 0
            self.closes = 0

        def shutdown(self):
            self.shutdowns += 1

        def server_close(self):
            self.closes += 1

    class _Thread:
        def __init__(self):
            self.joins = []

        def join(self, timeout=None):
            self.joins.append(timeout)

        def is_alive(self):
            return len(self.joins) < 2

    server = _Server()
    thread = _Thread()
    lease = scoring_worker_module._OfficialBaselineEvidenceProxyLease(
        server=server,
        thread=thread,
    )

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="thread did not stop",
    ):
        lease.close()
    lease.close()
    lease.close()

    assert server.shutdowns == 1
    assert server.closes == 1
    assert thread.joins == [5.0, 5.0]


@pytest.mark.asyncio
async def test_failed_proxy_shutdown_retains_owner_until_retry_succeeds():
    class _Server:
        def __init__(self):
            self.shutdowns = 0
            self.closes = 0

        def shutdown(self):
            self.shutdowns += 1

        def server_close(self):
            self.closes += 1

    class _Thread:
        def __init__(self):
            self.joins = []

        def join(self, timeout=None):
            self.joins.append(timeout)

        def is_alive(self):
            return len(self.joins) < 2

    worker = object.__new__(
        scoring_worker_module.ResearchLabGatewayScoringWorker
    )
    worker._active_official_baseline_evidence_proxy = None
    server = _Server()
    thread = _Thread()
    lease = scoring_worker_module._OfficialBaselineEvidenceProxyLease(
        server=server,
        thread=thread,
    )
    runs = 0

    async def implementation():
        nonlocal runs
        runs += 1
        if runs == 1:
            worker._active_official_baseline_evidence_proxy = lease
        return {"status": "fixture success"}

    worker._maybe_run_private_baseline_impl = implementation

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="thread did not stop",
    ):
        await worker._maybe_run_private_baseline()
    assert worker._active_official_baseline_evidence_proxy is lease
    assert worker._official_baseline_proxy_cleanup_pending is True
    assert thread.is_alive()

    assert await worker._maybe_run_private_baseline() == {
        "status": "fixture success"
    }
    assert worker._active_official_baseline_evidence_proxy is None
    assert worker._official_baseline_proxy_cleanup_pending is False
    assert not thread.is_alive()

    lease.close()
    assert server.shutdowns == 1
    assert server.closes == 1
    assert thread.joins == [5.0, 5.0]


def test_exact_proxy_partial_construction_closes_on_live_validation_failure(
    monkeypatch,
):
    from gateway.research_lab import provider_evidence_proxy

    class _Server:
        server_address = ("127.0.0.1", 43210)

        def __init__(self):
            self.shutdowns = 0
            self.closes = 0

        def shutdown(self):
            self.shutdowns += 1

        def server_close(self):
            self.closes += 1

    class _Thread:
        def __init__(self):
            self.joins = []

        def is_alive(self):
            return False

        def join(self, timeout=None):
            self.joins.append(timeout)

    server = _Server()
    thread = _Thread()

    def serve(**_values):
        return server, object(), thread

    monkeypatch.setattr(provider_evidence_proxy, "serve_evidence_proxy", serve)

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="did not bind a live loopback port",
    ):
        scoring_worker_module._start_official_baseline_evidence_proxy(
            benchmark_date="2026-08-23",
            rolling_window_hash="sha256:" + "f" * 64,
            artifact_hash="sha256:" + HASH["artifact"],
            worker_ref="fixture-worker",
        )

    assert server.shutdowns == 1
    assert server.closes == 1
    assert thread.joins == [5.0]


def test_required_role_preflight_fails_before_proxy_construction():
    runner, _projector, _authority, _terminal = _exact_fixture()
    worker = object.__new__(
        scoring_worker_module.ResearchLabGatewayScoringWorker
    )
    worker.worker_ref = "fixture-worker"
    worker._active_official_baseline_evidence_proxy = None
    constructed = []

    def preflight(**_values):
        raise ModelRunnerHostError("unknown required full-company role")

    def construct_proxy(**_values):
        constructed.append(True)
        raise AssertionError("proxy must not be constructed")

    worker._official_baseline_protocol_preflight = preflight
    worker._official_baseline_evidence_proxy_factory = construct_proxy

    with pytest.raises(
        ModelRunnerHostError,
        match="unknown required full-company role",
    ):
        worker._start_official_baseline_proxy_for_release(
            artifact=runner.artifact,
            selection=runner.selection,
            spec=runner.spec,
            benchmark_date="2026-08-23",
            rolling_window_hash="sha256:" + "f" * 64,
        )

    assert constructed == []
    assert worker._active_official_baseline_evidence_proxy is None


@pytest.mark.asyncio
async def test_dependency_construction_failure_closes_assigned_proxy_lease():
    runner, _projector, _authority, _terminal = _exact_fixture()

    class _Lease:
        url = "http://127.0.0.1:43210"
        capability_sha256 = "sha256:" + "a" * 64
        ready_provider_ids = ("or",)

        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    worker = object.__new__(
        scoring_worker_module.ResearchLabGatewayScoringWorker
    )
    worker.worker_ref = "fixture-worker"
    worker._active_official_baseline_evidence_proxy = None
    worker._official_baseline_protocol_preflight = lambda **_values: None
    lease = _Lease()
    worker._official_baseline_evidence_proxy_factory = (
        lambda **_values: lease
    )

    def construct_dependencies(*, context):
        del context
        raise OfficialBaselineAuthorityUnavailable(
            "dependency construction failed"
        )

    worker._construct_official_baseline_exact_dependencies = (
        construct_dependencies
    )

    async def implementation():
        worker._start_official_baseline_proxy_for_release(
            artifact=runner.artifact,
            selection=runner.selection,
            spec=runner.spec,
            benchmark_date="2026-08-23",
            rolling_window_hash="sha256:" + "f" * 64,
        )
        worker._construct_official_baseline_exact_dependencies(
            context=object()
        )

    worker._maybe_run_private_baseline_impl = implementation

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="dependency construction failed",
    ):
        await worker._maybe_run_private_baseline()

    assert lease.closed == 1
    assert worker._active_official_baseline_evidence_proxy is None
