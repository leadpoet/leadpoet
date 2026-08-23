"""Credential-free OCI calls to the common champion runner adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

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

def declared_member(field):
    name = payload.get("member_name")
    if not isinstance(name, str) or not name.isidentifier():
        raise RuntimeError("common runner member name is invalid")
    metadata = module.adapter_metadata()
    champion = metadata.get("champion_execution")
    if not isinstance(champion, dict):
        raise RuntimeError("common runner champion metadata is invalid")
    declared = champion.get(field)
    # The PR323 v2 generation predates explicit start-member metadata.  Its
    # frozen consumer contract still declares the exact member.  Every newer
    # generation must publish the role directly in champion_execution.
    if declared is not None and declared != name:
        raise RuntimeError("common runner member differs from artifact metadata")
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
    adapter_path = "research_lab_adapter.py"
    prefix = adapter_path + ":"
    functions = contract.get("functions", {}).get(adapter_path)
    if not isinstance(functions, dict):
        raise RuntimeError("common runner consumer functions are unavailable")
    result = {
        "schema_version": "leadpoet.research_lab.artifact_runner_declaration.v1",
        "champion_execution": metadata.get("champion_execution"),
        "consumer_contract_sha256": contract_sha256,
        "consumer_contract": {
            "schema_version": contract.get("schema_version"),
            "contract_id": contract.get("contract_id"),
            "functions": functions,
            "exact_signatures": sorted(
                item for item in contract.get("exact_signatures", [])
                if isinstance(item, str) and item.startswith(prefix)
            ),
            "full_parameters": {
                key[len(prefix):]: value
                for key, value in contract.get("full_parameters", {}).items()
                if isinstance(key, str) and key.startswith(prefix)
            },
            "required_keyword_only": {
                key[len(prefix):]: value
                for key, value in contract.get("required_keyword_only", {}).items()
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
        },
    }
elif operation == "build_raw_runner_input":
    result = declared_member("raw_icp_entrypoint")(
        payload["payload"],
        source_schema=payload["source_schema"],
    )
elif operation == "build_runner_start":
    result = declared_member("start_entrypoint")(
        input=payload["input"],
        execution_mode=payload["execution_mode"],
        target_count=payload["target_count"],
        evaluated_on=payload["evaluated_on"],
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
    )
elif operation == "continue_runner":
    result = declared_member("entrypoint")(
        payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
        continuation=payload.get("continuation"),
        completion=payload.get("completion"),
    )
elif operation == "build_runner_completion":
    result = declared_member("completion_entrypoint")(
        payload["action"],
        payload["result"],
    )
elif operation == "build_runner_provider_receipt_binding":
    result = declared_member("provider_receipt_binding_entrypoint")(
        payload["action"],
        payload["result"],
    )
elif operation == "runner_preflight":
    result = declared_member("preflight_entrypoint")(
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
        execution_mode=payload["execution_mode"],
    )
elif operation == "validate_runner_preflight":
    result = declared_member("preflight_validation_entrypoint")(
        payload["value"],
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
        execution_mode=payload["execution_mode"],
    )
elif operation == "validate_runner_result":
    result = declared_member("result_validation_entrypoint")(
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
