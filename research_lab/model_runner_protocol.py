"""Research Lab transport for the immutable champion runner artifact."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .common_model_runner_host import HostActionResult, ModelRunnerHostError


class ArtifactRunnerTransport(Protocol):
    """OCI boundary exposed by the reviewed Research Lab adapter."""

    def continue_runner(
        self,
        start_request: Mapping[str, Any],
        *,
        expected_release_identity: Mapping[str, Any],
        continuation: Mapping[str, Any] | None,
        completion: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...

    def build_runner_completion(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class ResearchLabModelRunnerProtocol:
    """Advance the champion only through its signed OCI adapter methods."""

    def __init__(
        self,
        *,
        transport: ArtifactRunnerTransport,
        expected_release_identity: Mapping[str, Any],
    ) -> None:
        if not isinstance(expected_release_identity, Mapping):
            raise ModelRunnerHostError("model release identity is required")
        self._transport = transport
        self._release_identity = dict(expected_release_identity)

    def advance(
        self,
        start_request: Mapping[str, Any],
        *,
        continuation: Mapping[str, Any] | None,
        completion: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        result = self._transport.continue_runner(
            start_request,
            expected_release_identity=self._release_identity,
            continuation=continuation,
            completion=completion,
        )
        if not isinstance(result, Mapping):
            raise ModelRunnerHostError("artifact continuation is invalid")
        return result

    def build_completion(
        self,
        action: Mapping[str, Any],
        result: HostActionResult,
    ) -> Mapping[str, Any]:
        completion = self._transport.build_runner_completion(
            action,
            {
                "outcome": result.outcome,
                "reason_code": result.reason_code,
                "provider_response": result.provider_response,
                "calls": result.calls,
                "cost_credits": result.cost_credits,
                "latency_ms": result.latency_ms,
            },
        )
        if not isinstance(completion, Mapping):
            raise ModelRunnerHostError("artifact completion is invalid")
        return completion


__all__ = [
    "ArtifactRunnerTransport",
    "ResearchLabModelRunnerProtocol",
]
