"""Focused Research Lab champion entry for the common model runner."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from research_lab.eval.private_runtime import DockerPrivateModelRunner

from .common_model_runner_host import (
    CommonModelRunnerHost,
    HostActionBinding,
    LoadCompletion,
    ModelRunnerHostError,
    PersistTransition,
)
from .docker_model_runner_transport import DockerModelRunnerTransport
from .model_runner_protocol import ResearchLabModelRunnerProtocol
from .model_runner_protocol import ExactModelRunnerRegistration


class _ArtifactBoundPersistTransition(Protocol):
    def __call__(self, *, artifact_key: str, **transition: Any) -> None: ...


class _ArtifactBoundLoadCompletion(Protocol):
    def __call__(
        self,
        *,
        artifact_key: str,
        idempotency_key: str,
    ) -> Mapping[str, Any] | None: ...


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
    persist_transition: _ArtifactBoundPersistTransition,
    load_completion: _ArtifactBoundLoadCompletion | None = None,
    continuation: Mapping[str, Any] | None = None,
    continuation_artifact_key: str | None = None,
) -> Mapping[str, Any]:
    """Run one reviewed Lab unit through the artifact-owned protocol.

    This is the non-semantic Lab caller.  It accepts an already registered
    artifact protocol and host bindings; it does not import model code, build
    routes, select providers, or parse model results locally.
    """

    if not isinstance(registration, ExactModelRunnerRegistration):
        raise ValueError("exact Model runner registration is required")
    registration.preflight()
    artifact_key = registration.key
    if continuation is None:
        if continuation_artifact_key is not None:
            raise ModelRunnerHostError(
                "continuation artifact identity exists without a continuation"
            )
    elif continuation_artifact_key != artifact_key:
        raise ModelRunnerHostError(
            "continuation artifact identity differs from the registration"
        )

    def persist_artifact_transition(**transition: Any) -> None:
        persist_transition(artifact_key=artifact_key, **transition)

    def load_artifact_completion(
        idempotency_key: str,
    ) -> Mapping[str, Any] | None:
        if load_completion is None:
            return None
        return load_completion(
            artifact_key=artifact_key,
            idempotency_key=idempotency_key,
        )

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
        persist_transition=persist_artifact_transition,
        load_completion=(
            load_artifact_completion if load_completion is not None else None
        ),
    )
    result = host.run(start_request, continuation=continuation)
    return protocol.validate_result(result, start_request=start_request)


__all__ = ["run_common_champion", "run_registered_model_unit"]
