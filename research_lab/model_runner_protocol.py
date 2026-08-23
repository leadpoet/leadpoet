"""Research Lab transport for the immutable champion runner artifact."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
from typing import Any, Mapping, Protocol, Sequence

from .canonical import sha256_json
from .common_model_runner_host import HostActionResult, ModelRunnerHostError


ARTIFACT_RUNNER_DECLARATION_SCHEMA_VERSION = (
    "leadpoet.research_lab.artifact_runner_declaration.v1"
)
ARTIFACT_RUNNER_PROTOCOL_GENERATION_SCHEMA_VERSION = (
    "leadpoet.research_lab.artifact_runner_protocol_generation.v1"
)

_ADAPTER_PATH = "research_lab_adapter.py"
_MODEL_RUNNER_CONSTANTS_PATH = "sourcing_model/model_runner.py"
_RAW_ICP_CONSTANTS_PATH = "sourcing_model/raw_icp_normalization.py"
_GENERATION_V2 = "model-runner-protocol:v2"
_GENERATION_V3 = "model-runner-protocol:v3"

_MEMBER_SIGNATURES = {
    "start": (
        "build_runner_start",
        (),
        ("input", "execution_mode", "target_count", "evaluated_on", "host_capability_manifest", "release_identity"),
        ("input", "execution_mode", "target_count", "evaluated_on", "host_capability_manifest", "release_identity"),
    ),
    "continue": (
        "continue_runner",
        ("start_request",),
        ("start_request", "expected_release_identity", "continuation", "completion"),
        ("expected_release_identity",),
    ),
    "completion": (
        "build_runner_completion",
        ("action", "result"),
        ("action", "result"),
        (),
    ),
    "preflight": (
        "runner_preflight",
        ("host_capability_manifest", "release_identity"),
        ("host_capability_manifest", "release_identity", "execution_mode"),
        ("execution_mode",),
    ),
    "preflight_validation": (
        "validate_runner_preflight",
        ("value",),
        ("value", "host_capability_manifest", "release_identity", "execution_mode"),
        ("host_capability_manifest", "release_identity", "execution_mode"),
    ),
    "result_validation": (
        "validate_runner_result",
        ("value",),
        ("value", "start_request", "expected_release_identity"),
        ("start_request", "expected_release_identity"),
    ),
    "raw_input": (
        "build_raw_runner_input",
        ("payload",),
        ("payload", "source_schema"),
        ("source_schema",),
    ),
    "provider_receipt_binding": (
        "build_runner_provider_receipt_binding",
        ("action", "result"),
        ("action", "result"),
        (),
    ),
}

_V2_VERSIONS = {
    "MODEL_RUNNER_START_SCHEMA_VERSION": "model-runner-start:v2",
    "MODEL_RUNNER_ACTION_SCHEMA_VERSION": "model-runner-action:v1",
    "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION": "model-runner-completion:v2",
    "MODEL_RUNNER_CONTINUATION_SCHEMA_VERSION": "model-runner-continuation:v2",
    "MODEL_RUNNER_RESULT_SCHEMA_VERSION": "model-runner-result:v2",
    "MODEL_RUNNER_RECEIPT_SCHEMA_VERSION": "model-runner-receipt:v2",
    "MODEL_RUNNER_PREFLIGHT_SCHEMA_VERSION": "model-runner-preflight:v2",
    "MODEL_RELEASE_IDENTITY_SCHEMA_VERSION": "model-release-identity:v2",
}
_V3_VERSIONS = {
    "MODEL_RUNNER_START_SCHEMA_VERSION": "model-runner-start:v3",
    "MODEL_RUNNER_ACTION_SCHEMA_VERSION": "model-runner-action:v2",
    "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION": "model-runner-completion:v3",
    "MODEL_RUNNER_CONTINUATION_SCHEMA_VERSION": "model-runner-continuation:v3",
    "MODEL_RUNNER_RESULT_SCHEMA_VERSION": "model-runner-result:v3",
    "MODEL_RUNNER_RECEIPT_SCHEMA_VERSION": "model-runner-receipt:v3",
    "MODEL_RUNNER_PREFLIGHT_SCHEMA_VERSION": "model-runner-preflight:v3",
    "MODEL_RELEASE_IDENTITY_SCHEMA_VERSION": "model-release-identity:v3",
}
_V3_RAW_VERSIONS = {
    "RAW_ICP_ENVELOPE_SCHEMA_VERSION": "model-raw-icp-envelope:v1",
    "SITE_RAW_ICP_SOURCE_SCHEMA": "leadpoet-site-company-request:v1",
    "LAB_RAW_ICP_SOURCE_SCHEMA": "leadpoet-research-lab-benchmark-icp:v1",
    "NORMALIZATION_ACTION_ARGUMENTS_SCHEMA_VERSION": (
        "model-normalization-action-arguments:v1"
    ),
    "NORMALIZATION_PROVIDER_RESPONSE_SCHEMA_VERSION": (
        "model-icp-normalization-provider-response:v1"
    ),
    "NORMALIZATION_TOOL_ID": "normalization.openrouter_json_schema",
    "NORMALIZATION_CALL_CAP": 1,
    "NORMALIZATION_CREDIT_CAP": 1.0,
    "NORMALIZATION_TIMEOUT_SECONDS": 120.0,
}
_V3_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION = (
    "model-provider-receipt-binding:v1"
)


def _closed_string_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ModelRunnerHostError(f"{label} is invalid")
    return dict(value)


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ModelRunnerHostError(f"{label} is invalid")
    return tuple(value)


@dataclass(frozen=True)
class ArtifactRunnerProtocolGeneration:
    """One exact artifact-declared runner member and schema generation."""

    family: str
    champion_execution: Mapping[str, Any]
    consumer_contract_sha256: str
    consumer_contract: Mapping[str, Any]
    members: Mapping[str, str]
    versions: Mapping[str, str]
    raw_source_schemas: tuple[str, ...]
    protocol_generation_sha256: str

    @classmethod
    def from_declaration(
        cls,
        value: Mapping[str, Any],
        *,
        expected_consumer_contract_sha256: str,
    ) -> "ArtifactRunnerProtocolGeneration":
        declaration = _closed_string_mapping(
            value, label="artifact runner declaration"
        )
        if set(declaration) != {
            "schema_version",
            "champion_execution",
            "consumer_contract_sha256",
            "consumer_contract",
        } or declaration.get("schema_version") != (
            ARTIFACT_RUNNER_DECLARATION_SCHEMA_VERSION
        ):
            raise ModelRunnerHostError(
                "artifact runner declaration fields are invalid"
            )
        contract_hash = str(
            declaration.get("consumer_contract_sha256") or ""
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", contract_hash)
            or contract_hash != expected_consumer_contract_sha256
        ):
            raise ModelRunnerHostError(
                "artifact runner consumer contract differs from release"
            )
        champion = _closed_string_mapping(
            declaration["champion_execution"],
            label="artifact champion execution metadata",
        )
        contract = _closed_string_mapping(
            declaration["consumer_contract"],
            label="artifact runner consumer contract",
        )
        if set(contract) != {
            "schema_version",
            "contract_id",
            "functions",
            "exact_signatures",
            "full_parameters",
            "required_keyword_only",
            "exact_constants",
        }:
            raise ModelRunnerHostError(
                "artifact runner consumer declaration is not closed"
            )
        functions = _closed_string_mapping(
            contract["functions"], label="artifact runner functions"
        )
        full_parameters = _closed_string_mapping(
            contract["full_parameters"],
            label="artifact runner full parameters",
        )
        keyword_only = _closed_string_mapping(
            contract["required_keyword_only"],
            label="artifact runner keyword-only parameters",
        )
        exact_signatures = frozenset(
            _string_sequence(
                contract["exact_signatures"],
                label="artifact runner exact signatures",
            )
        )
        constants = _closed_string_mapping(
            contract["exact_constants"],
            label="artifact runner exact constants",
        )
        model_constants = _closed_string_mapping(
            constants.get(_MODEL_RUNNER_CONSTANTS_PATH),
            label="artifact runner model constants",
        )
        raw_constants_value = constants.get(_RAW_ICP_CONSTANTS_PATH)
        raw_constants = (
            {}
            if raw_constants_value is None
            else _closed_string_mapping(
                raw_constants_value,
                label="artifact runner raw ICP constants",
            )
        )
        if all(
            model_constants.get(name) == expected
            for name, expected in _V3_VERSIONS.items()
        ):
            family = _GENERATION_V3
            expected_versions = _V3_VERSIONS
            required_roles = frozenset(_MEMBER_SIGNATURES)
            if any(
                raw_constants.get(name) != expected
                for name, expected in _V3_RAW_VERSIONS.items()
            ) or model_constants.get(
                "MODEL_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION"
            ) != _V3_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION:
                raise ModelRunnerHostError(
                    "artifact runner v3 normalization identities differ"
                )
            expected_champion_keys = {
                "raw_icp_envelope_schema_version",
                "raw_icp_source_schemas",
                "raw_icp_entrypoint",
                "release_identity_schema_version",
                "start_schema_version",
                "action_schema_version",
                "completion_schema_version",
                "continuation_schema_version",
                "preflight_schema_version",
                "result_schema_version",
                "receipt_schema_version",
                "entrypoint",
                "start_entrypoint",
                "completion_entrypoint",
                "provider_receipt_binding_entrypoint",
                "preflight_entrypoint",
                "preflight_validation_entrypoint",
                "result_validation_entrypoint",
                "legacy_rollback_entrypoint",
                "normalization_action",
            }
        elif all(
            model_constants.get(name) == expected
            for name, expected in _V2_VERSIONS.items()
        ):
            family = _GENERATION_V2
            expected_versions = _V2_VERSIONS
            required_roles = frozenset({
                "start",
                "continue",
                "completion",
                "preflight",
                "preflight_validation",
                "result_validation",
            })
            expected_champion_keys = {
                "release_identity_schema_version",
                "start_schema_version",
                "action_schema_version",
                "completion_schema_version",
                "preflight_schema_version",
                "result_schema_version",
                "receipt_schema_version",
                "entrypoint",
                "completion_entrypoint",
                "preflight_entrypoint",
                "preflight_validation_entrypoint",
                "result_validation_entrypoint",
                "legacy_rollback_entrypoint",
            }
        else:
            raise ModelRunnerHostError(
                "artifact runner protocol generation is unsupported"
            )
        if set(champion) != expected_champion_keys:
            raise ModelRunnerHostError(
                "artifact champion execution metadata differs from generation"
            )
        champion_versions = {
            "MODEL_RELEASE_IDENTITY_SCHEMA_VERSION": champion.get(
                "release_identity_schema_version"
            ),
            "MODEL_RUNNER_START_SCHEMA_VERSION": champion.get(
                "start_schema_version"
            ),
            "MODEL_RUNNER_ACTION_SCHEMA_VERSION": champion.get(
                "action_schema_version"
            ),
            "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION": champion.get(
                "completion_schema_version"
            ),
            "MODEL_RUNNER_PREFLIGHT_SCHEMA_VERSION": champion.get(
                "preflight_schema_version"
            ),
            "MODEL_RUNNER_RESULT_SCHEMA_VERSION": champion.get(
                "result_schema_version"
            ),
            "MODEL_RUNNER_RECEIPT_SCHEMA_VERSION": champion.get(
                "receipt_schema_version"
            ),
        }
        if family == _GENERATION_V3:
            champion_versions["MODEL_RUNNER_CONTINUATION_SCHEMA_VERSION"] = (
                champion.get("continuation_schema_version")
            )
        if any(
            champion_versions.get(name) != expected_versions[name]
            for name in champion_versions
        ) or any(
            model_constants.get(name) != expected
            for name, expected in expected_versions.items()
        ):
            raise ModelRunnerHostError(
                "artifact champion schema tuple differs from consumer contract"
            )

        member_metadata_keys = {
            "start": "start_entrypoint",
            "continue": "entrypoint",
            "completion": "completion_entrypoint",
            "preflight": "preflight_entrypoint",
            "preflight_validation": "preflight_validation_entrypoint",
            "result_validation": "result_validation_entrypoint",
            "raw_input": "raw_icp_entrypoint",
            "provider_receipt_binding": (
                "provider_receipt_binding_entrypoint"
            ),
        }
        members: dict[str, str] = {}
        for role in sorted(required_roles):
            (
                expected_name,
                expected_required_positional,
                expected_parameters,
                expected_keyword_only,
            ) = (
                _MEMBER_SIGNATURES[role]
            )
            declared_name = champion.get(member_metadata_keys[role])
            if declared_name is None and family == _GENERATION_V2 and role == "start":
                declared_name = expected_name
            if declared_name != expected_name:
                raise ModelRunnerHostError(
                    f"artifact runner {role} member is unsupported"
                )
            if (
                f"{_ADAPTER_PATH}:{expected_name}" not in exact_signatures
                or _string_sequence(
                    functions.get(expected_name),
                    label=f"artifact runner {role} positional parameters",
                )
                != expected_required_positional
                or _string_sequence(
                    full_parameters.get(expected_name),
                    label=f"artifact runner {role} full parameters",
                )
                != expected_parameters
                or _string_sequence(
                    keyword_only.get(expected_name),
                    label=f"artifact runner {role} keyword-only parameters",
                )
                != expected_keyword_only
            ):
                raise ModelRunnerHostError(
                    f"artifact runner {role} signature differs"
                )
            members[role] = expected_name

        raw_sources: tuple[str, ...] = ()
        if family == _GENERATION_V3:
            raw_sources = tuple(sorted(_string_sequence(
                champion.get("raw_icp_source_schemas"),
                label="artifact raw ICP source schemas",
            )))
            if raw_sources != tuple(sorted({
                _V3_RAW_VERSIONS["SITE_RAW_ICP_SOURCE_SCHEMA"],
                _V3_RAW_VERSIONS["LAB_RAW_ICP_SOURCE_SCHEMA"],
            })) or champion.get("raw_icp_envelope_schema_version") != (
                _V3_RAW_VERSIONS["RAW_ICP_ENVELOPE_SCHEMA_VERSION"]
            ):
                raise ModelRunnerHostError(
                    "artifact raw ICP source identity differs"
                )
            normalization = _closed_string_mapping(
                champion.get("normalization_action"),
                label="artifact normalization action",
            )
            expected_normalization = {
                "action_type": "normalize_icp",
                "stage": "icp_normalization",
                "tool_id": _V3_RAW_VERSIONS["NORMALIZATION_TOOL_ID"],
                "request_schema_version": _V3_RAW_VERSIONS[
                    "NORMALIZATION_ACTION_ARGUMENTS_SCHEMA_VERSION"
                ],
                "response_schema_version": _V3_RAW_VERSIONS[
                    "NORMALIZATION_PROVIDER_RESPONSE_SCHEMA_VERSION"
                ],
                "provider_receipt_binding_schema_version": model_constants.get(
                    "MODEL_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION"
                ),
                "call_cap": _V3_RAW_VERSIONS["NORMALIZATION_CALL_CAP"],
                "credit_cap": _V3_RAW_VERSIONS[
                    "NORMALIZATION_CREDIT_CAP"
                ],
                "timeout_seconds": _V3_RAW_VERSIONS[
                    "NORMALIZATION_TIMEOUT_SECONDS"
                ],
                "completion_custody_fields": [
                    "provider_receipt_ref",
                    "provider_receipt_sha256",
                    "provider_identity_sha256",
                ],
            }
            if normalization != expected_normalization:
                raise ModelRunnerHostError(
                    "artifact normalization action identity differs"
                )

        generation_payload = {
            "schema_version": (
                ARTIFACT_RUNNER_PROTOCOL_GENERATION_SCHEMA_VERSION
            ),
            "family": family,
            "champion_execution": champion,
            "consumer_contract_sha256": contract_hash,
            "consumer_contract": contract,
            "members": dict(sorted(members.items())),
            "versions": {
                name: model_constants[name]
                for name in sorted(expected_versions)
            },
            "raw_source_schemas": list(raw_sources),
        }
        return cls(
            family=family,
            champion_execution=champion,
            consumer_contract_sha256=contract_hash,
            consumer_contract=contract,
            members=dict(sorted(members.items())),
            versions={
                name: model_constants[name]
                for name in sorted(expected_versions)
            },
            raw_source_schemas=raw_sources,
            protocol_generation_sha256=sha256_json(generation_payload),
        )

    @property
    def supports_raw_icp(self) -> bool:
        return self.family == _GENERATION_V3

    @property
    def supports_provider_receipt_binding(self) -> bool:
        return self.family == _GENERATION_V3

    def member(self, role: str) -> str:
        value = self.members.get(role)
        if not isinstance(value, str) or not value:
            raise ModelRunnerHostError(
                f"artifact runner generation has no {role} member"
            )
        return value

    def version(self, constant_name: str) -> str:
        value = self.versions.get(constant_name)
        if not isinstance(value, str) or not value:
            raise ModelRunnerHostError(
                f"artifact runner generation has no {constant_name}"
            )
        return value


class ArtifactRunnerTransport(Protocol):
    """OCI boundary exposed by the reviewed Research Lab adapter."""

    def runner_protocol_generation(
        self,
        *,
        release_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_raw_runner_input(
        self,
        payload: Mapping[str, Any],
        *,
        source_schema: str,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def continue_runner(
        self,
        start_request: Mapping[str, Any],
        *,
        expected_release_identity: Mapping[str, Any],
        continuation: Mapping[str, Any] | None,
        completion: Mapping[str, Any] | None,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def build_runner_completion(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def build_runner_provider_receipt_binding(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def runner_preflight(
        self,
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        execution_mode: str,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def validate_runner_preflight(
        self,
        value: Mapping[str, Any],
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        execution_mode: str,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def validate_runner_result(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
        member_name: str,
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
        member_name: str,
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
            "runner_protocol_generation",
            "build_raw_runner_input",
            "continue_runner",
            "build_runner_completion",
            "build_runner_provider_receipt_binding",
            "runner_preflight",
            "validate_runner_preflight",
            "validate_runner_result",
            "build_runner_start",
        ):
            if not callable(getattr(transport, method_name, None)):
                raise ModelRunnerHostError(
                    f"artifact transport method {method_name} is unavailable"
                )
        self._transport = transport
        self._release_identity = dict(expected_release_identity)
        self._generation: ArtifactRunnerProtocolGeneration | None = None
        self._generation_lock = threading.Lock()

    @property
    def protocol_generation(self) -> ArtifactRunnerProtocolGeneration:
        generation = self._generation
        if generation is not None:
            return generation
        with self._generation_lock:
            generation = self._generation
            if generation is None:
                contract_hash = str(
                    self._release_identity.get("consumer_contract_sha256")
                    or ""
                )
                if not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
                    raise ModelRunnerHostError(
                        "model release consumer contract hash is invalid"
                    )
                declaration = self._transport.runner_protocol_generation(
                    release_identity=self._release_identity,
                )
                generation = ArtifactRunnerProtocolGeneration.from_declaration(
                    declaration,
                    expected_consumer_contract_sha256=contract_hash,
                )
                release_schema = str(
                    self._release_identity.get("schema_version") or ""
                )
                if release_schema != generation.version(
                    "MODEL_RELEASE_IDENTITY_SCHEMA_VERSION"
                ):
                    raise ModelRunnerHostError(
                        "model release schema differs from runner generation"
                    )
                self._generation = generation
            return generation

    def build_raw_input(
        self,
        payload: Mapping[str, Any],
        *,
        source_schema: str,
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        if not generation.supports_raw_icp:
            raise ModelRunnerHostError(
                "artifact runner generation has no raw ICP entrypoint"
            )
        if source_schema not in generation.raw_source_schemas:
            raise ModelRunnerHostError(
                "raw ICP source schema is not artifact-declared"
            )
        result = self._transport.build_raw_runner_input(
            payload,
            source_schema=source_schema,
            member_name=generation.member("raw_input"),
        )
        if not isinstance(result, Mapping) or set(result) != {
            "kind", "raw_icp"
        } or result.get("kind") != "raw_icp":
            raise ModelRunnerHostError("artifact raw ICP input is invalid")
        envelope = result.get("raw_icp")
        champion = generation.champion_execution
        if (
            not isinstance(envelope, Mapping)
            or envelope.get("schema_version")
            != champion.get("raw_icp_envelope_schema_version")
            or envelope.get("source_schema") != source_schema
        ):
            raise ModelRunnerHostError(
                "artifact raw ICP envelope differs from generation"
            )
        return dict(result)

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
            member_name=self.protocol_generation.member("continue"),
        )
        return self._validate_state(result, "artifact continuation")

    @property
    def release_identity(self) -> Mapping[str, Any]:
        return dict(self._release_identity)

    @property
    def artifact_provider_receipt_binding_required(self) -> bool:
        """Whether this exact generation declared the receipt member."""

        return self.protocol_generation.supports_provider_receipt_binding

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
            member_name=self.protocol_generation.member("start"),
        )
        if not isinstance(result, Mapping):
            raise ModelRunnerHostError("artifact start request is invalid")
        if not isinstance(result.get("host_capability_manifest"), Mapping):
            raise ModelRunnerHostError(
                "artifact start request has no host capability manifest"
            )
        if result.get("schema_version") != self.protocol_generation.version(
            "MODEL_RUNNER_START_SCHEMA_VERSION"
        ):
            raise ModelRunnerHostError(
                "artifact start request differs from runner generation"
            )
        return dict(result)

    def preflight(
        self,
        *,
        host_capability_manifest: Mapping[str, Any],
        execution_mode: str,
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        receipt = self._transport.runner_preflight(
            host_capability_manifest=host_capability_manifest,
            release_identity=self._release_identity,
            execution_mode=execution_mode,
            member_name=generation.member("preflight"),
        )
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema_version")
            != generation.version("MODEL_RUNNER_PREFLIGHT_SCHEMA_VERSION")
            or receipt.get("execution_mode") != execution_mode
        ):
            raise ModelRunnerHostError("artifact preflight is invalid")
        validated = self._transport.validate_runner_preflight(
            receipt,
            host_capability_manifest=host_capability_manifest,
            release_identity=self._release_identity,
            execution_mode=execution_mode,
            member_name=generation.member("preflight_validation"),
        )
        if not isinstance(validated, Mapping) or dict(validated) != dict(
            receipt
        ):
            raise ModelRunnerHostError(
                "artifact preflight validator changed the receipt"
            )
        return dict(validated)

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
            member_name=self.protocol_generation.member(
                "result_validation"
            ),
        )
        return self._validate_state(result, "artifact result preflight")

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
                "provider_receipt_sha256": (
                    result.provider_receipt_sha256
                ),
                "provider_identity_sha256": (
                    result.provider_identity_sha256
                ),
            },
            member_name=self.protocol_generation.member("completion"),
        )
        if not isinstance(completion, Mapping) or completion.get(
            "schema_version"
        ) != self.protocol_generation.version(
            "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION"
        ):
            raise ModelRunnerHostError("artifact completion is invalid")
        return completion

    def build_provider_receipt_binding(
        self,
        action: Mapping[str, Any],
        result: HostActionResult,
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        if not generation.supports_provider_receipt_binding:
            raise ModelRunnerHostError(
                "artifact runner generation has no provider receipt binding"
            )
        binding = self._transport.build_runner_provider_receipt_binding(
            action,
            {
                "provider_response": result.provider_response,
                "provider_receipt_ref": result.provider_receipt_ref,
                "provider_identity_sha256": (
                    result.provider_identity_sha256
                ),
                "calls": result.calls,
                "cost_credits": result.cost_credits,
                "latency_ms": result.latency_ms,
            },
            member_name=generation.member("provider_receipt_binding"),
        )
        expected_schema = generation.champion_execution[
            "normalization_action"
        ]["provider_receipt_binding_schema_version"]
        if (
            not isinstance(binding, Mapping)
            or binding.get("schema_version") != expected_schema
            or binding.get("provider_receipt_ref")
            != result.provider_receipt_ref
            or binding.get("provider_identity_sha256")
            != result.provider_identity_sha256
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(binding.get("receipt_sha256") or "")
            )
        ):
            raise ModelRunnerHostError(
                "artifact provider receipt binding is invalid"
            )
        return dict(binding)

    def validate_normalization_action(
        self,
        action: Mapping[str, Any],
        *,
        host_capability_manifest: Mapping[str, Any],
    ) -> None:
        generation = self.protocol_generation
        if not generation.supports_raw_icp:
            raise ModelRunnerHostError(
                "artifact runner generation has no normalization action"
            )
        expected = generation.champion_execution["normalization_action"]
        arguments = action.get("arguments")
        expected_call_cap = expected["call_cap"]
        expected_credit = expected["credit_cap"]
        expected_timeout = expected["timeout_seconds"]
        if (
            not isinstance(action, Mapping)
            or action.get("schema_version")
            != generation.version("MODEL_RUNNER_ACTION_SCHEMA_VERSION")
            or action.get("action_type") != expected["action_type"]
            or action.get("action_phase") != "normalization"
            or action.get("stage") != expected["stage"]
            or action.get("tool_id") != expected["tool_id"]
            or action.get("response_schema_version")
            != expected["response_schema_version"]
            or not isinstance(arguments, Mapping)
            or "step" in arguments
            or arguments.get("schema_version")
            != expected["request_schema_version"]
            or type(arguments.get("call_cap")) is not type(expected_call_cap)
            or arguments.get("call_cap") != expected_call_cap
            or type(arguments.get("credit_cap")) is not type(expected_credit)
            or arguments.get("credit_cap") != expected_credit
            or type(arguments.get("timeout_seconds")) is not type(expected_timeout)
            or arguments.get("timeout_seconds") != expected_timeout
        ):
            raise ModelRunnerHostError(
                "artifact normalization action differs from generation"
            )
        bindings = host_capability_manifest.get("bindings")
        if not isinstance(bindings, (list, tuple)):
            raise ModelRunnerHostError(
                "normalization host bindings are unavailable"
            )
        matches = [
            item
            for item in bindings
            if isinstance(item, Mapping)
            and item.get("action_type") == expected["action_type"]
            and item.get("tool_id") == expected["tool_id"]
        ]
        if (
            len(matches) != 1
            or matches[0].get("available") is not True
            or matches[0].get("binding_contract_sha256")
            != action.get("binding_contract_sha256")
            or matches[0].get("response_schema_version")
            != expected["response_schema_version"]
        ):
            raise ModelRunnerHostError(
                "normalization binding differs from artifact action"
            )

    def _validate_state(
        self, value: Mapping[str, Any], label: str
    ) -> Mapping[str, Any]:
        """Validate only the transport envelope, never model semantics."""

        if not isinstance(value, Mapping):
            raise ModelRunnerHostError(f"{label} is invalid")
        status = str(value.get("status") or "")
        if status == "action_required":
            action = value.get("action")
            continuation = value.get("continuation")
            if not isinstance(action, Mapping) or not isinstance(
                continuation, Mapping
            ):
                raise ModelRunnerHostError(f"{label} action state is invalid")
            action_type = str(action.get("action_type") or "")
            tool_id = str(action.get("tool_id") or "").strip()
            binding_hash = str(action.get("binding_contract_sha256") or "")
            idempotency_key = str(action.get("idempotency_key") or "")
            if action.get("schema_version") != self.protocol_generation.version(
                "MODEL_RUNNER_ACTION_SCHEMA_VERSION"
            ) or continuation.get(
                "schema_version"
            ) != self.protocol_generation.version(
                "MODEL_RUNNER_CONTINUATION_SCHEMA_VERSION"
            ) or not action_type or not tool_id or not re.fullmatch(
                r"[0-9a-f]{64}", binding_hash
            ) or not re.fullmatch(r"[0-9a-f]{64}", idempotency_key):
                raise ModelRunnerHostError(f"{label} action identity is invalid")
        elif status == "completed":
            if not isinstance(value.get("result"), Mapping) or not isinstance(
                value.get("model_receipt"), Mapping
            ) or not isinstance(value.get("continuation"), Mapping):
                raise ModelRunnerHostError(f"{label} terminal result is invalid")
            if value.get("action") not in (None, {}):
                raise ModelRunnerHostError(
                    f"{label} completed action must be empty"
                )
            if value["result"].get(
                "schema_version"
            ) != self.protocol_generation.version(
                "MODEL_RUNNER_RESULT_SCHEMA_VERSION"
            ) or value["model_receipt"].get(
                "schema_version"
            ) != self.protocol_generation.version(
                "MODEL_RUNNER_RECEIPT_SCHEMA_VERSION"
            ) or value["continuation"].get(
                "schema_version"
            ) != self.protocol_generation.version(
                "MODEL_RUNNER_CONTINUATION_SCHEMA_VERSION"
            ):
                raise ModelRunnerHostError(
                    f"{label} terminal schemas differ from runner generation"
                )
        else:
            raise ModelRunnerHostError(f"{label} status is invalid")
        return dict(value)


_SHA256_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION_V1 = (
    "leadpoet.research_lab.exact_model_variant_audit.v1"
)
EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION_V2 = (
    "leadpoet.research_lab.exact_model_variant_audit.v2"
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

    def validate_identity(self) -> None:
        """Validate artifact/release metadata without invoking OCI methods."""

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

    def preflight(self, *, execution_mode: str) -> Mapping[str, Any]:
        self.validate_identity()
        # The immutable artifact validates the complete, generation-specific
        # preflight receipt.  The Lab does not duplicate a partial field list
        # that could silently accept a newer generation under old semantics.
        return self.protocol.preflight(
            host_capability_manifest=self.host_capability_manifest,
            execution_mode=execution_mode,
        )

    @property
    def protocol_generation(self) -> ArtifactRunnerProtocolGeneration:
        self.validate_identity()
        return self.protocol.protocol_generation

    @property
    def key(self) -> str:
        return _artifact_key(self.artifact_identity)

    def variant_audit_payload(self) -> Mapping[str, str]:
        """Return identity-only metadata; it never carries routing semantics."""

        generation = self.protocol_generation
        if generation.family == _GENERATION_V2:
            return {
                "schema_version": EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION_V1,
                "artifact_key": self.key,
            }
        return {
            "schema_version": EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION_V2,
            "artifact_key": self.key,
            "protocol_generation_sha256": (
                generation.protocol_generation_sha256
            ),
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
            registration.validate_identity()
            indexed[key] = registration
        if not indexed:
            raise ModelRunnerHostError(
                "model runner registry must not be empty"
            )
        self._registrations = indexed

    def validate_all(self) -> None:
        """Revalidate every registered artifact without invoking OCI."""

        for registration in self._registrations.values():
            registration.validate_identity()

    def registrations(self) -> tuple[ExactModelRunnerRegistration, ...]:
        """Return the closed exact registration set after cheap validation."""

        self.validate_all()
        return tuple(
            registration
            for _key, registration in sorted(self._registrations.items())
        )

    def preflight_all(
        self,
        *,
        execution_mode: str,
    ) -> Mapping[str, Mapping[str, Any]]:
        return {
            key: registration.preflight(execution_mode=execution_mode)
            for key, registration in sorted(self._registrations.items())
        }

    def resolve(
        self,
        artifact_identity: Mapping[str, Any],
        *,
        execution_mode: str,
    ) -> ExactModelRunnerRegistration:
        registration = self.resolve_identity(artifact_identity)
        registration.preflight(execution_mode=execution_mode)
        return registration

    def resolve_identity(
        self,
        artifact_identity: Mapping[str, Any],
    ) -> ExactModelRunnerRegistration:
        """Resolve exact registered metadata without invoking the artifact."""

        registration = self._registrations.get(
            _artifact_key(artifact_identity)
        )
        if registration is None or dict(registration.artifact_identity) != dict(
            artifact_identity
        ):
            raise ModelRunnerHostError(
                "exact model runner registration is unavailable"
            )
        registration.validate_identity()
        return registration


__all__ = [
    "ARTIFACT_RUNNER_DECLARATION_SCHEMA_VERSION",
    "ARTIFACT_RUNNER_PROTOCOL_GENERATION_SCHEMA_VERSION",
    "ArtifactRunnerProtocolGeneration",
    "ArtifactRunnerTransport",
    "EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION_V1",
    "EXACT_MODEL_VARIANT_AUDIT_SCHEMA_VERSION_V2",
    "ExactModelRunnerRegistration",
    "ExactModelRunnerRegistry",
    "ResearchLabModelRunnerProtocol",
]
