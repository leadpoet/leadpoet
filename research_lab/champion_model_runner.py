"""Focused Research Lab champion entry for the common model runner."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from research_lab.eval.private_runtime import DockerPrivateModelRunner

from .common_model_runner_host import (
    CommonModelRunnerHost,
    HostActionBinding,
    LoadCompletion,
    PersistTransition,
)
from .docker_model_runner_transport import DockerModelRunnerTransport
from .model_runner_protocol import ResearchLabModelRunnerProtocol
from .model_runner_protocol import ExactModelRunnerRegistration


def run_common_champion(
    *,
    runner: DockerPrivateModelRunner,
    input: Mapping[str, Any],
    execution_mode: str,
    target_count: int,
    evaluated_on: str,
    host_capability_manifest: Mapping[str, Any],
    release_identity: Mapping[str, Any],
    bindings: Sequence[HostActionBinding],
    persist_transition: PersistTransition,
    load_completion: LoadCompletion | None = None,
    continuation: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run one champion request without the legacy segmented executor."""

    transport = DockerModelRunnerTransport(runner)
    protocol = ResearchLabModelRunnerProtocol(
        transport=transport,
        expected_release_identity=release_identity,
    )
    protocol.preflight(
        host_capability_manifest=host_capability_manifest,
    )
    start_request = protocol.build_start(
        input=input,
        execution_mode=execution_mode,
        target_count=target_count,
        evaluated_on=evaluated_on,
        host_capability_manifest=host_capability_manifest,
    )
    host = CommonModelRunnerHost(
        consumer_id="research-lab-champion",
        protocol=protocol,
        bindings=bindings,
        persist_transition=persist_transition,
        load_completion=load_completion,
    )
    result = host.run(start_request, continuation=continuation)
    return protocol.validate_result(result, start_request=start_request)


def run_registered_model_unit(
    *,
    registration: ExactModelRunnerRegistration,
    input: Mapping[str, Any],
    execution_mode: str,
    target_count: int,
    evaluated_on: str,
    bindings: Sequence[HostActionBinding],
    persist_transition: PersistTransition,
    load_completion: LoadCompletion | None = None,
    continuation: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run one reviewed Lab unit through the artifact-owned protocol.

    This is the non-semantic Lab caller.  It accepts an already registered
    artifact protocol and host bindings; it does not import model code, build
    routes, select providers, or parse model results locally.
    """

    if not isinstance(registration, ExactModelRunnerRegistration):
        raise ValueError("exact Model runner registration is required")
    registration.preflight()
    protocol = registration.protocol
    protocol.preflight(
        host_capability_manifest=registration.host_capability_manifest,
    )
    start_request = protocol.build_start(
        input=input,
        execution_mode=execution_mode,
        target_count=target_count,
        evaluated_on=evaluated_on,
        host_capability_manifest=registration.host_capability_manifest,
    )
    host = CommonModelRunnerHost(
        consumer_id="research-lab-champion",
        protocol=protocol,
        bindings=bindings,
        persist_transition=persist_transition,
        load_completion=load_completion,
    )
    result = host.run(start_request, continuation=continuation)
    return protocol.validate_result(result, start_request=start_request)


__all__ = ["run_common_champion", "run_registered_model_unit"]
