"""Research Lab transport for the immutable champion runner artifact."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Protocol, Sequence

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

    def runner_preflight(
        self,
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def validate_runner_result(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_runner_start(
        self,
        *,
        input: Mapping[str, Any],
        execution_mode: str,
        target_count: int,
        evaluated_on: str,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
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
        for method_name in (
            "continue_runner",
            "build_runner_completion",
            "runner_preflight",
            "validate_runner_result",
            "build_runner_start",
        ):
            if not callable(getattr(transport, method_name, None)):
                raise ModelRunnerHostError(
                    f"artifact transport method {method_name} is unavailable"
                )
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

    @property
    def release_identity(self) -> Mapping[str, Any]:
        return dict(self._release_identity)

    def build_start(
        self,
        *,
        input: Mapping[str, Any],
        execution_mode: str,
        target_count: int,
        evaluated_on: str,
        host_capability_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._transport.build_runner_start(
            input=input,
            execution_mode=execution_mode,
            target_count=target_count,
            evaluated_on=evaluated_on,
            host_capability_manifest=host_capability_manifest,
            release_identity=self._release_identity,
        )
        if not isinstance(result, Mapping):
            raise ModelRunnerHostError("artifact start request is invalid")
        return result

    def preflight(
        self,
        *,
        host_capability_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        receipt = self._transport.runner_preflight(
            host_capability_manifest=host_capability_manifest,
            release_identity=self._release_identity,
        )
        if not isinstance(receipt, Mapping):
            raise ModelRunnerHostError("artifact preflight is invalid")
        return receipt

    def validate_result(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._transport.validate_runner_result(
            value,
            start_request=start_request,
            expected_release_identity=self._release_identity,
        )
        if not isinstance(result, Mapping):
            raise ModelRunnerHostError("artifact result preflight is invalid")
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
                "provider_receipt_ref": result.provider_receipt_ref,
            },
        )
        if not isinstance(completion, Mapping):
            raise ModelRunnerHostError("artifact completion is invalid")
        return completion


_SHA256_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION = (
    "leadpoet.research_lab.exact_model_variant_audit.v1"
)


def _digest_body(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ModelRunnerHostError(f"{field_name} is invalid")
    return normalized.removeprefix("sha256:")


def _artifact_key(artifact_identity: Mapping[str, Any]) -> str:
    commit = str(artifact_identity.get("commit_sha") or "").strip().lower()
    artifact_hash = str(
        artifact_identity.get("model_artifact_hash") or ""
    ).strip().lower()
    manifest_hash = str(
        artifact_identity.get("manifest_hash") or ""
    ).strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ModelRunnerHostError("artifact commit is invalid")
    _digest_body(artifact_hash, "artifact model digest")
    _digest_body(manifest_hash, "artifact manifest hash")
    return f"{commit}:{artifact_hash}:{manifest_hash}"


@dataclass(frozen=True)
class ExactModelRunnerRegistration:
    """One reviewed artifact and its credential-free PR274 protocol."""

    artifact_identity: Mapping[str, Any]
    protocol: ResearchLabModelRunnerProtocol
    host_capability_manifest: Mapping[str, Any]

    def preflight(self) -> Mapping[str, Any]:
        if not isinstance(self.protocol, ResearchLabModelRunnerProtocol):
            raise ModelRunnerHostError("artifact protocol is invalid")
        artifact = dict(self.artifact_identity)
        release = dict(self.protocol.release_identity)
        if str(artifact.get("repository") or "") != "leadpoet/Sourcing_model":
            raise ModelRunnerHostError("artifact repository is invalid")
        if str(artifact.get("branch") or "") not in {
            "main", "leadpoet-lab"
        }:
            raise ModelRunnerHostError("artifact branch is invalid")
        if str(release.get("source_commit") or "") != str(
            artifact.get("commit_sha") or ""
        ):
            raise ModelRunnerHostError("artifact commit differs from release")
        digest_pairs = (
            ("model_artifact_hash", "model_artifact_digest"),
            ("routing_contract_hash", "consumer_contract_sha256"),
            ("routing_catalog_hash", "catalog_sha256"),
            ("routing_policy_hash", "policy_sha256"),
            ("feature_schema_hash", "feature_schema_sha256"),
        )
        for artifact_name, release_name in digest_pairs:
            if _digest_body(
                artifact.get(artifact_name), artifact_name
            ) != _digest_body(release.get(release_name), release_name):
                raise ModelRunnerHostError(
                    f"artifact {artifact_name} differs from release"
                )
        receipt = self.protocol.preflight(
            host_capability_manifest=self.host_capability_manifest,
        )
        expected = {
            "release_identity_sha256": release.get(
                "release_identity_sha256"
            ),
            "source_commit": release.get("source_commit"),
            "consumer_contract_sha256": release.get(
                "consumer_contract_sha256"
            ),
            "catalog_sha256": release.get("catalog_sha256"),
            "policy_sha256": release.get("policy_sha256"),
            "candidate_profiles_sha256": release.get(
                "candidate_profiles_sha256"
            ),
            "intent_profiles_sha256": release.get(
                "intent_profiles_sha256"
            ),
            "feature_schema_sha256": release.get(
                "feature_schema_sha256"
            ),
            "host_capability_manifest_sha256": (
                self.host_capability_manifest.get("manifest_sha256")
            ),
            "binding_contracts_sha256": release.get(
                "tool_binding_manifest_sha256"
            ),
            "candidate_waterfall_contract_sha256": release.get(
                "candidate_waterfall_contract_sha256"
            ),
        }
        if any(receipt.get(name) != value for name, value in expected.items()):
            raise ModelRunnerHostError(
                "artifact preflight receipt differs from registration"
            )
        return dict(receipt)

    @property
    def key(self) -> str:
        return _artifact_key(self.artifact_identity)

    def variant_audit_payload(self) -> Mapping[str, str]:
        """Return identity-only metadata; it never carries routing semantics."""

        return {
            "schema_version": EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION,
            "artifact_key": self.key,
        }

    def validate_variant_audit_payload(
        self, payload: Mapping[str, Any]
    ) -> None:
        if not isinstance(payload, Mapping) or dict(payload) != dict(
            self.variant_audit_payload()
        ):
            raise ModelRunnerHostError(
                "variant payload must contain only the exact Model artifact identity"
            )


class ExactModelRunnerRegistry:
    """Resolve baseline and challenger only by exact reviewed artifact ID."""

    def __init__(
        self,
        registrations: Sequence[ExactModelRunnerRegistration],
    ) -> None:
        indexed: dict[str, ExactModelRunnerRegistration] = {}
        for registration in registrations:
            if not isinstance(registration, ExactModelRunnerRegistration):
                raise ModelRunnerHostError(
                    "model runner registration is invalid"
                )
            key = registration.key
            if key in indexed:
                raise ModelRunnerHostError(
                    "model runner registration is duplicated"
                )
            registration.preflight()
            indexed[key] = registration
        if not indexed:
            raise ModelRunnerHostError(
                "model runner registry must not be empty"
            )
        self._registrations = indexed

    def preflight_all(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            key: registration.preflight()
            for key, registration in sorted(self._registrations.items())
        }

    def resolve(
        self,
        artifact_identity: Mapping[str, Any],
    ) -> ExactModelRunnerRegistration:
        registration = self._registrations.get(
            _artifact_key(artifact_identity)
        )
        if registration is None or dict(registration.artifact_identity) != dict(
            artifact_identity
        ):
            raise ModelRunnerHostError(
                "exact model runner registration is unavailable"
            )
        registration.preflight()
        return registration


__all__ = [
    "ArtifactRunnerTransport",
    "EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION",
    "ExactModelRunnerRegistration",
    "ExactModelRunnerRegistry",
    "ResearchLabModelRunnerProtocol",
]
