"""Credential-free OCI calls to the common champion runner adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    PrivateModelRuntimeError,
)


_COMMON_RUNNER_BOOTSTRAP = r"""
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys

module_name, operation = sys.argv[1:3]
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)

def declared_member(role, legacy_field):
    name = payload.get("member_name")
    if not isinstance(name, str) or not name.isidentifier():
        raise RuntimeError("common runner member name is invalid")
    metadata = module.adapter_metadata()
    champion = metadata.get("champion_execution")
    if not isinstance(champion, dict):
        raise RuntimeError("common runner champion metadata is invalid")
    role_contract = champion.get("runner_role_contract")
    if isinstance(role_contract, dict):
        roles = role_contract.get("roles")
        role_entry = roles.get(role) if isinstance(roles, dict) else None
        declared = (
            role_entry.get("adapter_member")
            if isinstance(role_entry, dict)
            else None
        )
    else:
        declared = champion.get(legacy_field)
    # The frozen v2 generation predates explicit start-member metadata.  Its
    # exact consumer contract remains the only authority for that one member.
    if declared is not None and declared != name:
        raise RuntimeError("common runner member differs from artifact metadata")
    if isinstance(role_contract, dict) and declared is None:
        raise RuntimeError("common runner semantic role is unavailable")
    member = getattr(module, name, None)
    if not callable(member):
        raise RuntimeError("common runner member is unavailable")
    return member

if operation == "runner_protocol_generation":
    metadata = module.adapter_metadata()
    contract_path = Path(module.__file__).resolve().parent / "sourcing_model" / "consumer_contract.json"
    contract_bytes = contract_path.read_bytes()
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    if contract_sha256 != payload["release_identity"]["consumer_contract_sha256"]:
        raise RuntimeError("common runner consumer contract differs from release")
    contract = json.loads(contract_bytes)
    champion = metadata.get("champion_execution")
    role_contract = (
        champion.get("runner_role_contract")
        if isinstance(champion, dict)
        else None
    )
    contract_functions = contract.get("functions", {})
    contract_signatures = contract.get("exact_signatures", [])
    contract_full_parameters = contract.get("full_parameters", {})
    contract_keyword_only = contract.get("required_keyword_only", {})
    contract_asyncness = contract.get("frozen_asyncness", {})
    if isinstance(role_contract, dict):
        roles = role_contract.get("roles")
        if not isinstance(roles, dict):
            raise RuntimeError("common runner semantic roles are unavailable")
        functions = {}
        exact_signatures = []
        full_parameters = {}
        required_keyword_only = {}
        frozen_asyncness = {}
        for role, entry in sorted(roles.items()):
            if not isinstance(entry, dict):
                raise RuntimeError("common runner semantic role is invalid")
            member = entry.get("adapter_member")
            signature = entry.get("consumer_signature")
            path = (
                signature.get("consumer_contract_path")
                if isinstance(signature, dict)
                else None
            )
            if (
                not isinstance(member, str)
                or not member.isidentifier()
                or not isinstance(path, str)
                or ":" not in path
                or path.rsplit(":", 1)[1] != member
            ):
                raise RuntimeError("common runner role member path is invalid")
            source_path = path.rsplit(":", 1)[0]
            source_functions = (
                contract_functions.get(source_path)
                if isinstance(contract_functions, dict)
                else None
            )
            if not isinstance(source_functions, dict):
                raise RuntimeError("common runner role functions are unavailable")
            functions[member] = source_functions.get(member)
            if path in contract_signatures:
                exact_signatures.append(path)
            full_parameters[member] = contract_full_parameters.get(path)
            if path in contract_keyword_only:
                required_keyword_only[member] = contract_keyword_only[path]
            # Consumer contracts historically list asynchronous members and
            # may omit synchronous ``False`` entries.  The signed role
            # signature remains the exact asyncness authority.
            frozen_asyncness[member] = contract_asyncness.get(path, False)
        consumer_contract = {
            "schema_version": contract.get("schema_version"),
            "contract_id": contract.get("contract_id"),
            "functions": functions,
            "exact_signatures": sorted(exact_signatures),
            "full_parameters": full_parameters,
            "required_keyword_only": required_keyword_only,
            "exact_constants": {
                key: value
                for key, value in contract.get("exact_constants", {}).items()
                if key in {
                    "sourcing_model/model_runner.py",
                    "sourcing_model/raw_icp_normalization.py",
                }
            },
            "extensions": contract.get("extensions"),
            "frozen_asyncness": frozen_asyncness,
        }
    else:
        adapter_path = "research_lab_adapter.py"
        prefix = adapter_path + ":"
        functions = (
            contract_functions.get(adapter_path)
            if isinstance(contract_functions, dict)
            else None
        )
        if not isinstance(functions, dict):
            raise RuntimeError("common runner consumer functions are unavailable")
        consumer_contract = {
            "schema_version": contract.get("schema_version"),
            "contract_id": contract.get("contract_id"),
            "functions": functions,
            "exact_signatures": sorted(
                item for item in contract_signatures
                if isinstance(item, str) and item.startswith(prefix)
            ),
            "full_parameters": {
                key[len(prefix):]: value
                for key, value in contract_full_parameters.items()
                if isinstance(key, str) and key.startswith(prefix)
            },
            "required_keyword_only": {
                key[len(prefix):]: value
                for key, value in contract_keyword_only.items()
                if isinstance(key, str) and key.startswith(prefix)
            },
            "exact_constants": {
                key: value
                for key, value in contract.get("exact_constants", {}).items()
                if key in {
                    "sourcing_model/model_runner.py",
                    "sourcing_model/raw_icp_normalization.py",
                }
            },
        }
    result = {
        "schema_version": "leadpoet.research_lab.artifact_runner_declaration.v1",
        "champion_execution": metadata.get("champion_execution"),
        "consumer_contract_sha256": contract_sha256,
        "consumer_contract": consumer_contract,
    }
elif operation == "build_raw_runner_input":
    result = declared_member("raw_icp_input", "raw_icp_entrypoint")(
        payload["payload"],
        source_schema=payload["source_schema"],
    )
elif operation == "build_runner_start":
    result = declared_member("start", "start_entrypoint")(
        input=payload["input"],
        execution_mode=payload["execution_mode"],
        target_count=payload["target_count"],
        evaluated_on=payload["evaluated_on"],
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
    )
elif operation == "continue_runner":
    result = declared_member("continuation", "entrypoint")(
        payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
        continuation=payload.get("continuation"),
        completion=payload.get("completion"),
    )
elif operation == "build_runner_completion":
    result = declared_member("completion", "completion_entrypoint")(
        payload["action"],
        payload["result"],
    )
elif operation == "build_runner_provider_receipt_binding":
    result = declared_member(
        "provider_receipt_binding", "provider_receipt_binding_entrypoint"
    )(
        payload["action"],
        payload["result"],
    )
elif operation == "build_host_capability_manifest":
    result = declared_member(
        "host_capability_manifest", "host_capability_manifest_entrypoint"
    )(
        payload["bindings"],
    )
elif operation == "project_runner_result_for_benchmark":
    result = declared_member(
        "benchmark_projection", "benchmark_projection_entrypoint"
    )(
        payload["value"],
        start_request=payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
    )
elif operation == "build_official_baseline_execution":
    result = declared_member(
        "official_baseline_execution", "official_baseline_execution_entrypoint"
    )(
        release_identity=payload["release_identity"],
        protocol_generation_sha256=payload["protocol_generation_sha256"],
        protected_action_authority_sha256=payload[
            "protected_action_authority_sha256"
        ],
    )
elif operation == "prepare_runner_provider_request":
    result = declared_member("provider_prepare", "provider_prepare_entrypoint")(
        payload["action"],
    )
elif operation == "ingest_runner_provider_response":
    result = declared_member(
        "provider_response_ingestion",
        "provider_response_ingestion_entrypoint",
    )(
        payload["action"],
        payload["host_response"],
    )
elif operation == "prepare_runner_normalization_request":
    champion = module.adapter_metadata().get("champion_execution", {})
    role_contract = (
        champion.get("runner_role_contract")
        if isinstance(champion, dict)
        else None
    )
    if isinstance(role_contract, dict):
        roles = role_contract.get("roles")
        role_entry = (
            roles.get("normalization_prepare_legacy")
            if isinstance(roles, dict)
            else None
        )
        declared = (
            role_entry.get("adapter_member")
            if isinstance(role_entry, dict)
            else None
        )
    else:
        normalization = (
            champion.get("normalization_action")
            if isinstance(champion, dict)
            else None
        )
        declared = (
            normalization.get("dispatch_entrypoint")
            if isinstance(normalization, dict)
            else None
        )
    if declared != payload["member_name"]:
        raise RuntimeError(
            "common runner normalization member differs from artifact metadata"
        )
    member = getattr(module, payload["member_name"], None)
    if not callable(member):
        raise RuntimeError("common runner normalization member is unavailable")
    result = member(payload["action"])
elif operation == "model_runner_provider_compiler_inventory":
    result = declared_member(
        "provider_compiler_inventory", "provider_compiler_inventory_entrypoint"
    )()
elif operation == "runner_provider_compiler_preflight":
    result = declared_member(
        "provider_compiler_preflight", "provider_compiler_preflight_entrypoint"
    )(
        payload["host_capability_manifest"],
    )
elif operation == "execute_runner_verifier_action":
    result = declared_member(
        "verifier_execution", "verifier_execution_entrypoint"
    )(
        payload["action"],
    )
elif operation == "runner_official_host_binding_catalog":
    result = declared_member(
        "official_host_binding_catalog", "official_host_binding_catalog_entrypoint"
    )()
elif operation == "build_runner_official_host_capability_manifest":
    result = declared_member(
        "official_host_capability_manifest",
        "official_host_capability_manifest_entrypoint",
    )(payload["availability"])
elif operation == "runner_preflight":
    result = declared_member("preflight", "preflight_entrypoint")(
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
        execution_mode=payload["execution_mode"],
    )
elif operation == "validate_runner_preflight":
    result = declared_member(
        "preflight_validation", "preflight_validation_entrypoint"
    )(
        payload["value"],
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
        execution_mode=payload["execution_mode"],
    )
elif operation == "validate_runner_result":
    result = declared_member(
        "result_validation", "result_validation_entrypoint"
    )(
        payload["value"],
        start_request=payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
    )
else:
    raise RuntimeError("unsupported common runner operation")
with contextlib.redirect_stdout(sys.stderr):
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
sys.stdout.write(encoded)
"""


class DockerModelRunnerTransport:
    """Invoke only runner APIs, without forwarding provider credentials."""

    def __init__(self, runner: DockerPrivateModelRunner) -> None:
        if not isinstance(runner, DockerPrivateModelRunner):
            raise PrivateModelRuntimeError("Docker model runner is required")
        isolated_spec = replace(
            runner.spec,
            env_passthrough=(),
            extra_env={},
            pull_before_run=False,
        )
        self._runner = DockerPrivateModelRunner(isolated_spec)

    def _call(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._runner._run_json(
            bootstrap=_COMMON_RUNNER_BOOTSTRAP,
            argv=(self._runner.spec.module_name, operation),
            stdin_payload=payload,
        )
        if not isinstance(result, Mapping):
            raise PrivateModelRuntimeError(
                "common model runner returned a non-object response"
            )
        return result

    def continue_runner(
        self,
        start_request: Mapping[str, Any],
        *,
        expected_release_identity: Mapping[str, Any],
        continuation: Mapping[str, Any] | None,
        completion: Mapping[str, Any] | None,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "continue_runner",
            {
                "start_request": dict(start_request),
                "expected_release_identity": dict(
                    expected_release_identity
                ),
                "continuation": (
                    None if continuation is None else dict(continuation)
                ),
                "completion": (
                    None if completion is None else dict(completion)
                ),
                "member_name": member_name,
            },
        )

    def runner_protocol_generation(
        self,
        *,
        release_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_protocol_generation",
            {"release_identity": dict(release_identity)},
        )

    def build_raw_runner_input(
        self,
        payload: Mapping[str, Any],
        *,
        source_schema: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_raw_runner_input",
            {
                "payload": dict(payload),
                "source_schema": source_schema,
                "member_name": member_name,
            },
        )

    def build_runner_start(
        self,
        *,
        input: Mapping[str, Any],
        execution_mode: str,
        target_count: int,
        evaluated_on: str,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_start",
            {
                "input": dict(input),
                "execution_mode": execution_mode,
                "target_count": target_count,
                "evaluated_on": evaluated_on,
                "host_capability_manifest": dict(
                    host_capability_manifest
                ),
                "release_identity": dict(release_identity),
                "member_name": member_name,
            },
        )

    def build_runner_completion(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_completion",
            {
                "action": dict(action),
                "result": dict(result),
                "member_name": member_name,
            },
        )

    def build_runner_provider_receipt_binding(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_provider_receipt_binding",
            {
                "action": dict(action),
                "result": dict(result),
                "member_name": member_name,
            },
        )

    def build_host_capability_manifest(
        self,
        *,
        bindings: Sequence[Mapping[str, Any]],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_host_capability_manifest",
            {
                "bindings": [dict(value) for value in bindings],
                "member_name": member_name,
            },
        )

    def project_runner_result_for_benchmark(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "project_runner_result_for_benchmark",
            {
                "value": dict(value),
                "start_request": dict(start_request),
                "expected_release_identity": dict(expected_release_identity),
                "member_name": member_name,
            },
        )

    def prepare_runner_provider_request(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "prepare_runner_provider_request",
            {"action": dict(action), "member_name": member_name},
        )

    def ingest_runner_provider_response(
        self,
        action: Mapping[str, Any],
        host_response: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "ingest_runner_provider_response",
            {
                "action": dict(action),
                "host_response": dict(host_response),
                "member_name": member_name,
            },
        )

    def prepare_runner_normalization_request(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "prepare_runner_normalization_request",
            {"action": dict(action), "member_name": member_name},
        )

    def model_runner_provider_compiler_inventory(
        self,
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "model_runner_provider_compiler_inventory",
            {"member_name": member_name},
        )

    def runner_provider_compiler_preflight(
        self,
        host_capability_manifest: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_provider_compiler_preflight",
            {
                "host_capability_manifest": dict(host_capability_manifest),
                "member_name": member_name,
            },
        )

    def execute_runner_verifier_action(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "execute_runner_verifier_action",
            {"action": dict(action), "member_name": member_name},
        )

    def runner_official_host_binding_catalog(
        self,
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_official_host_binding_catalog",
            {"member_name": member_name},
        )

    def build_runner_official_host_capability_manifest(
        self,
        availability: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_official_host_capability_manifest",
            {
                "availability": dict(availability),
                "member_name": member_name,
            },
        )

    def build_official_baseline_execution(
        self,
        *,
        release_identity: Mapping[str, Any],
        protocol_generation_sha256: str,
        protected_action_authority_sha256: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_official_baseline_execution",
            {
                "release_identity": dict(release_identity),
                "protocol_generation_sha256": protocol_generation_sha256,
                "protected_action_authority_sha256": (
                    protected_action_authority_sha256
                ),
                "member_name": member_name,
            },
        )

    def runner_preflight(
        self,
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        execution_mode: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_preflight",
            {
                "host_capability_manifest": dict(host_capability_manifest),
                "release_identity": dict(release_identity),
                "execution_mode": execution_mode,
                "member_name": member_name,
            },
        )

    def validate_runner_preflight(
        self,
        value: Mapping[str, Any],
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        execution_mode: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "validate_runner_preflight",
            {
                "value": dict(value),
                "host_capability_manifest": dict(host_capability_manifest),
                "release_identity": dict(release_identity),
                "execution_mode": execution_mode,
                "member_name": member_name,
            },
        )

    def validate_runner_result(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "validate_runner_result",
            {
                "value": dict(value),
                "start_request": dict(start_request),
                "expected_release_identity": dict(expected_release_identity),
                "member_name": member_name,
            },
        )


__all__ = ["DockerModelRunnerTransport"]
