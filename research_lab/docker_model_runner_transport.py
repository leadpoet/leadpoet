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
import importlib
import json
import sys

module_name, operation = sys.argv[1:3]
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)
if operation == "build_runner_start":
    result = module.build_runner_start(
        input=payload["input"],
        execution_mode=payload["execution_mode"],
        target_count=payload["target_count"],
        evaluated_on=payload["evaluated_on"],
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
    )
elif operation == "continue_runner":
    result = module.continue_runner(
        payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
        continuation=payload.get("continuation"),
        completion=payload.get("completion"),
    )
elif operation == "build_runner_completion":
    result = module.build_runner_completion(
        payload["action"],
        payload["result"],
    )
elif operation == "runner_preflight":
    result = module.runner_preflight(
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
    )
elif operation == "validate_runner_result":
    result = module.validate_runner_result(
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
            },
        )

    def build_runner_completion(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_completion",
            {"action": dict(action), "result": dict(result)},
        )

    def runner_preflight(
        self,
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_preflight",
            {
                "host_capability_manifest": dict(host_capability_manifest),
                "release_identity": dict(release_identity),
            },
        )

    def validate_runner_result(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._call(
            "validate_runner_result",
            {
                "value": dict(value),
                "start_request": dict(start_request),
                "expected_release_identity": dict(expected_release_identity),
            },
        )


__all__ = ["DockerModelRunnerTransport"]
