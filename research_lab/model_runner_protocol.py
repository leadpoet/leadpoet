"""Research Lab transport for the immutable champion runner artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
_ZERO_SHA256 = "0" * 64
_MODEL_PROVIDER_ACTION_TYPES = frozenset({
    "normalize_icp",
    "execute_candidate_tool",
    "execute_intent_tool",
    "execute_contact_tool",
})

_V3_BASE_ROLES = frozenset({
    "start",
    "continuation",
    "completion",
    "preflight",
    "preflight_validation",
    "result_validation",
    "raw_icp_input",
    "provider_receipt_binding",
})
_V3_OFFICIAL_BASELINE_ROLES = frozenset({
    "host_capability_manifest",
    "benchmark_projection",
    "official_baseline_execution",
    "provider_prepare",
    "provider_response_ingestion",
    "provider_compiler_inventory",
    "provider_compiler_preflight",
    "verifier_execution",
    "official_host_binding_catalog",
    "official_host_capability_manifest",
})
_V3_OPTIONAL_ROLES = frozenset({"normalization_prepare_legacy"})
_V3_FULL_COMPANY_REQUIRED_ROLES = (
    _V3_BASE_ROLES | _V3_OFFICIAL_BASELINE_ROLES
)

_MEMBER_SIGNATURES = {
    "host_capability_manifest": (
        "build_host_capability_manifest",
        ("bindings",),
        ("bindings",),
        (),
    ),
    "benchmark_projection": (
        "project_runner_result_for_benchmark",
        ("value",),
        ("value", "start_request", "expected_release_identity"),
        ("start_request", "expected_release_identity"),
    ),
    "official_baseline_execution": (
        "build_official_baseline_execution",
        (),
        (
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ),
        (
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ),
    ),
    "provider_prepare": (
        "prepare_runner_provider_request",
        ("action",),
        ("action",),
        (),
    ),
    "provider_response_ingestion": (
        "ingest_runner_provider_response",
        ("action", "host_response"),
        ("action", "host_response"),
        (),
    ),
    "normalization_prepare_legacy": (
        "prepare_runner_normalization_request",
        ("action",),
        ("action",),
        (),
    ),
    "provider_compiler_inventory": (
        "model_runner_provider_compiler_inventory",
        (),
        (),
        (),
    ),
    "provider_compiler_preflight": (
        "runner_provider_compiler_preflight",
        ("host_capability_manifest",),
        ("host_capability_manifest",),
        (),
    ),
    "verifier_execution": (
        "execute_runner_verifier_action",
        ("action",),
        ("action",),
        (),
    ),
    "official_host_binding_catalog": (
        "runner_official_host_binding_catalog",
        (),
        (),
        (),
    ),
    "official_host_capability_manifest": (
        "build_runner_official_host_capability_manifest",
        ("availability",),
        ("availability",),
        (),
    ),
    "start": (
        "build_runner_start",
        (),
        ("input", "execution_mode", "target_count", "evaluated_on", "host_capability_manifest", "release_identity"),
        ("input", "execution_mode", "target_count", "evaluated_on", "host_capability_manifest", "release_identity"),
    ),
    "continuation": (
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
    "raw_icp_input": (
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
_V3_LEGACY_VERSIONS = {
    "MODEL_RUNNER_START_SCHEMA_VERSION": "model-runner-start:v3",
    "MODEL_RUNNER_ACTION_SCHEMA_VERSION": "model-runner-action:v2",
    "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION": "model-runner-completion:v3",
    "MODEL_RUNNER_CONTINUATION_SCHEMA_VERSION": "model-runner-continuation:v3",
    "MODEL_RUNNER_RESULT_SCHEMA_VERSION": "model-runner-result:v3",
    "MODEL_RUNNER_RECEIPT_SCHEMA_VERSION": "model-runner-receipt:v3",
    "MODEL_RUNNER_PREFLIGHT_SCHEMA_VERSION": "model-runner-preflight:v3",
    "MODEL_RELEASE_IDENTITY_SCHEMA_VERSION": "model-release-identity:v3",
}
_V3_INGESTION_CUSTODY_VERSIONS = {
    **_V3_LEGACY_VERSIONS,
    "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION": "model-runner-completion:v4",
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
_V3_LEGACY_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION = (
    "model-provider-receipt-binding:v1"
)
_V3_INGESTION_CUSTODY_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION = (
    "model-provider-receipt-binding:v2"
)
_V3_COMPLETION_ACCOUNTING_SCHEMA_VERSION = (
    "model-runner-completion-accounting:v2"
)
_V3_OFFICIAL_BASELINE_MEMBER_METADATA_KEYS = {
    "host_capability_manifest": "host_capability_manifest_entrypoint",
    "benchmark_projection": "benchmark_projection_entrypoint",
    "official_baseline_execution": "official_baseline_execution_entrypoint",
    "provider_prepare": "provider_prepare_entrypoint",
    "provider_response_ingestion": (
        "provider_response_ingestion_entrypoint"
    ),
    "provider_compiler_inventory": "provider_compiler_inventory_entrypoint",
    "provider_compiler_preflight": "provider_compiler_preflight_entrypoint",
    "verifier_execution": "verifier_execution_entrypoint",
    "official_host_binding_catalog": (
        "official_host_binding_catalog_entrypoint"
    ),
    "official_host_capability_manifest": (
        "official_host_capability_manifest_entrypoint"
    ),
}
_V3_OFFICIAL_BASELINE_SCHEMA_VERSIONS = {
    "host_capability_manifest_schema_version": "host-capability-manifest:v1",
    "benchmark_projection_schema_version": (
        "model-runner-benchmark-projection:v1"
    ),
    "official_baseline_execution_schema_version": (
        "leadpoet.research_lab.official_baseline_execution.v1"
    ),
    "provider_prepare_schema_version": "model-runner-provider-dispatch:v1",
    "provider_response_ingestion_schema_version": (
        "model-runner-provider-response-ingestion:v1"
    ),
    "provider_compiler_inventory_schema_version": (
        "model-runner-provider-compiler-inventory:v1"
    ),
    "provider_compiler_preflight_schema_version": (
        "model-runner-provider-compiler-preflight:v1"
    ),
    "verifier_execution_schema_version": (
        "model-runner-verifier-execution:v1"
    ),
    "official_host_binding_catalog_schema_version": (
        "model-runner-official-host-binding-catalog:v1"
    ),
    # This identity belongs only to the signed official role-contract bundle.
    # Legacy generations without that bundle retain their exact declarations.
    "provider_response_schema_version": "model-provider-response:v3",
    "candidate_provider_record_schema_version": (
        "model-candidate-provider-record:v1"
    ),
    "candidate_provider_projection_schema_version": (
        "model-candidate-provider-projection:v1"
    ),
    "company_verifier_result_schema_version": (
        "company-verifier-response:v2"
    ),
    "company_fit_source_evidence_schema_version": (
        "company-fit-source-evidence:v1"
    ),
    "company_verifier_evidence_schema_version": (
        "company-verifier-evidence:v2"
    ),
    "model_verified_lead_evidence_schema_version": (
        "model-verified-lead-evidence:v2"
    ),
}
_V3_OFFICIAL_BASELINE_CONTRACT_KEYS = frozenset({
    "host_capability_manifest_contract",
    "benchmark_projection_contract",
    "official_baseline_execution_contract",
    "provider_prepare_contract",
    "provider_response_ingestion_contract",
    "verifier_execution_contract",
    "official_host_binding_catalog_contract",
    "candidate_provider_projection_contract",
    "company_verifier_result_contract",
    "company_fit_source_evidence_contract",
})
_V3_OFFICIAL_BASELINE_HASH_KEYS = frozenset({
    "company_fit_proof_contract_sha256",
})
_V3_OFFICIAL_BASELINE_METADATA_KEYS = frozenset(
    _V3_OFFICIAL_BASELINE_MEMBER_METADATA_KEYS.values()
) | frozenset(_V3_OFFICIAL_BASELINE_SCHEMA_VERSIONS) | (
    _V3_OFFICIAL_BASELINE_CONTRACT_KEYS
) | _V3_OFFICIAL_BASELINE_HASH_KEYS

_RUNNER_ROLE_CONTRACT_SCHEMA_V1 = "model-runner-role-contract:v1"
_RUNNER_ROLE_CONTRACT_SCHEMA_V2 = "model-runner-role-contract:v2"
_RUNNER_ROLE_CONTRACT_SCHEMA_VERSIONS = frozenset({
    _RUNNER_ROLE_CONTRACT_SCHEMA_V1,
    _RUNNER_ROLE_CONTRACT_SCHEMA_V2,
})
_RUNNER_ROLE_LEGACY_COMPATIBILITY_MAJOR = 1
_RUNNER_ROLE_INGESTION_CUSTODY_COMPATIBILITY_MAJOR = 2
_RUNNER_ROLE_ADDITIVE_COMPATIBILITY = {
    "known_required_roles": (
        "bind_stable_interface_and_exact_signed_consumer_signature"
    ),
    "unknown_required_roles": "reject_before_preflight_or_spend",
    "unknown_optional_roles": "accept_ignore_and_hash_bind",
    "member_names_are_discovered_from_roles": True,
    "commit_allowlists": False,
    "exact_member_tuple_allowlists": False,
}
_ROLE_INTERFACE_SHAPES = {
    "raw_icp_input": (
        ("payload",),
        ("source_schema",),
        ("source_schema",),
    ),
    "start": (
        (),
        (
            "input",
            "execution_mode",
            "target_count",
            "evaluated_on",
            "host_capability_manifest",
            "release_identity",
        ),
        (
            "input",
            "execution_mode",
            "target_count",
            "evaluated_on",
            "host_capability_manifest",
            "release_identity",
        ),
    ),
    "continuation": (
        ("start_request",),
        ("expected_release_identity", "continuation", "completion"),
        ("expected_release_identity",),
    ),
    "completion": (("action", "result"), (), ()),
    "provider_receipt_binding": (("action", "result"), (), ()),
    "preflight": (
        ("host_capability_manifest", "release_identity"),
        ("execution_mode",),
        ("execution_mode",),
    ),
    "preflight_validation": (
        ("value",),
        ("host_capability_manifest", "release_identity", "execution_mode"),
        ("host_capability_manifest", "release_identity", "execution_mode"),
    ),
    "result_validation": (
        ("value",),
        ("start_request", "expected_release_identity"),
        ("start_request", "expected_release_identity"),
    ),
    "benchmark_projection": (
        ("value",),
        ("start_request", "expected_release_identity"),
        ("start_request", "expected_release_identity"),
    ),
    "host_capability_manifest": (("bindings",), (), ()),
    "official_baseline_execution": (
        (),
        (
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ),
        (
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ),
    ),
    "provider_prepare": (("action",), (), ()),
    "provider_response_ingestion": (
        ("action", "host_response"),
        (),
        (),
    ),
    "provider_compiler_inventory": ((), (), ()),
    "provider_compiler_preflight": (
        ("host_capability_manifest",),
        (),
        (),
    ),
    "verifier_execution": (("action",), (), ()),
    "official_host_binding_catalog": ((), (), ()),
    "official_host_capability_manifest": (("availability",), (), ()),
    "normalization_prepare_legacy": (("action",), (), ()),
}

_PROVIDER_RESPONSE_INGESTION_FIELD_ORDER = (
    "schema_version",
    "action_sha256",
    "dispatch_sha256",
    "compiler_id",
    "compiler_contract_sha256",
    "request_sha256",
    "host_response_schema_version",
    "host_response_sha256",
    "provider",
    "parsed_response_schema_version",
    "parsed_response",
    "parsed_response_sha256",
    "ingestion_sha256",
)
_PROVIDER_RESPONSE_INGESTION_FIELDS = frozenset(
    _PROVIDER_RESPONSE_INGESTION_FIELD_ORDER
)
_HOST_PROVIDER_RESPONSE_FIELDS = frozenset({
    "schema_version",
    "provider",
    "status_code",
    "body",
})


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


def _validate_contract_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    """Validate a model-owned, self-hashed static contract identity."""

    identity = _closed_string_mapping(value, label=label)
    contract_hash = identity.get("contract_sha256")
    if not isinstance(contract_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", contract_hash
    ):
        raise ModelRunnerHostError(f"{label} hash is invalid")
    payload = {
        key: item for key, item in identity.items() if key != "contract_sha256"
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelRunnerHostError(f"{label} is not canonical JSON") from exc
    if hashlib.sha256(encoded).hexdigest() != contract_hash:
        raise ModelRunnerHostError(f"{label} hash differs")
    return identity


def _bare_sha256_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelRunnerHostError("artifact value is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_provider_response_ingestion_contract(
    value: Any,
    *,
    provider_dispatch_contract_sha256: str,
    requires_ingestion_custody: bool | None = None,
) -> Mapping[str, Any]:
    """Validate the host-visible ingestion contract before any paid action."""

    contract = _validate_contract_identity(
        value,
        label="artifact provider response ingestion contract",
    )
    legacy_fields = {
        "schema_version",
        "ingestion_entrypoint",
        "ingestion_signature",
        "host_response_schema_version",
        "host_response_closed_fields",
        "host_response_body_authority",
        "dispatch_schema_version",
        "dispatch_contract_sha256",
        "compiler_inventory_sha256",
        "parsed_response_schema_versions",
        "closed_fields",
        "raw_host_response_returned",
        "completion_input",
        "provider_receipt_binding_input",
        "completion_reparses_response",
        "durable_custody",
        "custody_receipt_field",
        "replay_requirement",
        "custody_join",
        "host_semantic_projection_allowed",
        "hash_algorithm",
        "canonical_json",
        "contract_sha256",
    }
    ingestion_custody = (
        contract.get("ingestion_receipt_required_for_response") is True
    )
    if (
        requires_ingestion_custody is not None
        and ingestion_custody is not requires_ingestion_custody
    ):
        raise ModelRunnerHostError(
            "artifact provider response ingestion custody generation differs"
        )
    expected_fields = (
        legacy_fields | {"ingestion_receipt_required_for_response"}
        if ingestion_custody
        else legacy_fields
    )
    completion_input = (
        "original_unchanged_host_response+"
        "exact_model_provider_response_ingestion"
        if ingestion_custody
        else "original_unchanged_host_response"
    )
    custody_join = (
        [
            "action_sha256",
            "dispatch_sha256",
            "request_sha256",
            "host_response_sha256",
            "parsed_response_sha256",
            "ingestion_sha256",
        ]
        if ingestion_custody
        else [
            "action_sha256",
            "host_response_sha256",
            "parsed_response_sha256",
        ]
    )
    parsed_schemas = _string_sequence(
        contract.get("parsed_response_schema_versions"),
        label="artifact parsed provider response schemas",
    )
    if (
        not expected_fields.issubset(contract)
        or contract.get("schema_version")
        != "model-runner-provider-response-ingestion:v1"
        or contract.get("ingestion_signature")
        != ["action", "host_response"]
        or contract.get("host_response_schema_version")
        != "host-provider-response:v1"
        or contract.get("host_response_closed_fields")
        != ["schema_version", "provider", "status_code", "body"]
        or contract.get("host_response_body_authority")
        != "action_bound_provider_compiler_parser"
        or contract.get("dispatch_schema_version")
        != "model-runner-provider-dispatch:v1"
        or contract.get("dispatch_contract_sha256")
        != provider_dispatch_contract_sha256
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(contract.get("compiler_inventory_sha256") or ""),
        )
        or not parsed_schemas
        or len(parsed_schemas) != len(set(parsed_schemas))
        or contract.get("closed_fields")
        != list(_PROVIDER_RESPONSE_INGESTION_FIELD_ORDER)
        or contract.get("raw_host_response_returned") is not False
        or contract.get("completion_input") != completion_input
        or contract.get("provider_receipt_binding_input")
        != completion_input
        or contract.get("completion_reparses_response") is not True
        or contract.get("durable_custody")
        != "host_provider_action_receipt_before_completion"
        or contract.get("custody_receipt_field")
        != "model_provider_response_ingestion"
        or contract.get("replay_requirement")
        != "reload_raw_response_reingest_and_require_byte_identical_receipt"
        or contract.get("custody_join") != custody_join
        or contract.get("host_semantic_projection_allowed") is not False
    ):
        raise ModelRunnerHostError(
            "artifact provider response ingestion contract differs"
        )
    return contract


def _runner_interface_contract(
    role: str,
    *,
    interface_major: int,
) -> dict[str, Any]:
    positional, host_keywords, required_keyword_only = (
        _ROLE_INTERFACE_SHAPES[role]
    )
    return {
        "interface_id": "leadpoet.model_runner." + role,
        "interface_major": interface_major,
        "positional_parameters": list(positional),
        "host_keyword_parameters": list(host_keywords),
        "required_keyword_only": list(required_keyword_only),
        "is_async": False,
    }


def _validate_runner_role_contract(
    value: Any,
    *,
    consumer_contract_id: str,
    functions: Mapping[str, Any],
    full_parameters: Mapping[str, Any],
    keyword_only: Mapping[str, Any],
    exact_signatures: frozenset[str],
    frozen_asyncness: Mapping[str, Any],
    compatibility_major: int,
    interface_majors: Mapping[str, int],
) -> dict[str, str]:
    """Validate the signed additive map and bind exact artifact members."""

    role_contract = _closed_string_mapping(
        value, label="artifact runner role contract"
    )
    expected_top_fields = {
        "schema_version",
        "compatibility_major",
        "consumer_contract_id",
        "roles",
        "activation_profiles",
        "additive_compatibility",
        "extensions",
        "canonical_json",
        "hash_algorithm",
        "contract_sha256",
    }
    contract_sha256 = str(role_contract.get("contract_sha256") or "")
    role_contract_schema = role_contract.get("schema_version")
    extensions = _closed_string_mapping(
        role_contract.get("extensions"),
        label="artifact runner role contract extensions",
    )
    contract_payload = {
        key: item
        for key, item in role_contract.items()
        if key != "contract_sha256"
    }
    if (
        set(role_contract) != expected_top_fields
        or role_contract_schema not in _RUNNER_ROLE_CONTRACT_SCHEMA_VERSIONS
        or role_contract.get("compatibility_major") != compatibility_major
        or role_contract.get("consumer_contract_id") != consumer_contract_id
        or role_contract.get("canonical_json")
        != "utf8-json-sort-keys-compact-ascii-no-nan"
        or role_contract.get("hash_algorithm") != "sha256"
        or any(
            "." not in key
            or re.fullmatch(r"[a-z][a-z0-9_.-]{2,127}", key) is None
            for key in extensions
        )
        or role_contract.get("additive_compatibility")
        != _RUNNER_ROLE_ADDITIVE_COMPATIBILITY
        or not re.fullmatch(r"[0-9a-f]{64}", contract_sha256)
        or _bare_sha256_json(contract_payload) != contract_sha256
    ):
        raise ModelRunnerHostError(
            "artifact runner role contract identity differs"
        )

    roles = _closed_string_mapping(
        role_contract.get("roles"), label="artifact runner roles"
    )
    profiles = _closed_string_mapping(
        role_contract.get("activation_profiles"),
        label="artifact runner activation profiles",
    )
    full_company = _closed_string_mapping(
        profiles.get("full_company"),
        label="artifact runner full-company profile",
    )
    required_roles = tuple(sorted(_string_sequence(
        full_company.get("required_roles"),
        label="artifact runner full-company required roles",
    )))
    minimum_majors = _closed_string_mapping(
        full_company.get("minimum_interface_major"),
        label="artifact runner full-company minimum majors",
    )
    if (
        set(full_company) != {
            "required_roles",
            "minimum_interface_major",
            "unknown_required_role_policy",
        }
        or full_company.get("unknown_required_role_policy")
        != "reject_before_preflight_or_spend"
        or required_roles != tuple(sorted(_V3_FULL_COMPANY_REQUIRED_ROLES))
        or set(minimum_majors) != set(required_roles)
        or any(
            minimum_majors.get(role) != interface_majors.get(role)
            for role in required_roles
        )
    ):
        raise ModelRunnerHostError(
            "artifact runner full-company role requirements differ"
        )

    members: dict[str, str] = {}
    known_roles = _V3_FULL_COMPANY_REQUIRED_ROLES | _V3_OPTIONAL_ROLES
    for role, raw_entry in sorted(roles.items()):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role):
            raise ModelRunnerHostError("artifact runner role name is invalid")
        entry = _closed_string_mapping(
            raw_entry, label=f"artifact runner {role} role"
        )
        if set(entry) != {
            "interface_id",
            "interface_major",
            "interface_contract",
            "interface_contract_sha256",
            "adapter_member",
            "consumer_signature",
            "consumer_signature_sha256",
            "required_for_profiles",
        }:
            raise ModelRunnerHostError(
                f"artifact runner {role} role fields differ"
            )
        interface = _closed_string_mapping(
            entry.get("interface_contract"),
            label=f"artifact runner {role} interface contract",
        )
        interface_sha256 = str(
            entry.get("interface_contract_sha256") or ""
        )
        signature = _closed_string_mapping(
            entry.get("consumer_signature"),
            label=f"artifact runner {role} consumer signature",
        )
        signature_sha256 = str(
            entry.get("consumer_signature_sha256") or ""
        )
        member = str(entry.get("adapter_member") or "")
        interface_major = interface.get("interface_major")
        interface_is_async = interface.get("is_async")
        required_for_profiles = tuple(_string_sequence(
            entry.get("required_for_profiles"),
            label=f"artifact runner {role} required profiles",
        ))
        if (
            set(interface) != {
                "interface_id",
                "interface_major",
                "positional_parameters",
                "host_keyword_parameters",
                "required_keyword_only",
                "is_async",
            }
            or interface.get("interface_id")
            != "leadpoet.model_runner." + role
            or type(interface_major) is not int
            or interface_major < 1
            or type(interface_is_async) is not bool
            or entry.get("interface_id") != interface.get("interface_id")
            or entry.get("interface_major") != interface_major
            or not re.fullmatch(r"[0-9a-f]{64}", interface_sha256)
            or _bare_sha256_json(interface) != interface_sha256
            or not member.isidentifier()
            or required_for_profiles
            != tuple(sorted(set(required_for_profiles)))
        ):
            raise ModelRunnerHostError(
                f"artifact runner {role} interface identity differs"
            )
        for field in (
            "positional_parameters",
            "host_keyword_parameters",
            "required_keyword_only",
        ):
            values = _string_sequence(
                interface.get(field),
                label=f"artifact runner {role} interface {field}",
            )
            if len(values) != len(set(values)):
                raise ModelRunnerHostError(
                    f"artifact runner {role} interface {field} is duplicated"
                )
        if not set(interface["required_keyword_only"]).issubset(
            set(interface["host_keyword_parameters"])
        ):
            raise ModelRunnerHostError(
                f"artifact runner {role} interface keywords differ"
            )

        expected_signature_fields = {
            "consumer_contract_id",
            "consumer_contract_path",
            "positional_parameters",
            "full_parameters",
            "required_keyword_only",
            "is_async",
        }
        if role_contract_schema == _RUNNER_ROLE_CONTRACT_SCHEMA_V2:
            expected_signature_fields.update({
                "required_positional_parameters",
                "defaulted_positional_parameters",
            })
        if set(signature) != expected_signature_fields:
            raise ModelRunnerHostError(
                f"artifact runner {role} consumer signature fields differ"
            )
        path = str(signature.get("consumer_contract_path") or "")
        path_parts = path.rsplit(":", 1)
        signature_positional = _string_sequence(
            signature.get("positional_parameters"),
            label=f"artifact runner {role} signature positional parameters",
        )
        signature_full = _string_sequence(
            signature.get("full_parameters"),
            label=f"artifact runner {role} signature full parameters",
        )
        signature_keyword_only = _string_sequence(
            signature.get("required_keyword_only"),
            label=f"artifact runner {role} signature keyword-only parameters",
        )
        signature_required_positional: tuple[str, ...]
        signature_defaulted_positional: tuple[str, ...]
        if role_contract_schema == _RUNNER_ROLE_CONTRACT_SCHEMA_V2:
            signature_required_positional = _string_sequence(
                signature.get("required_positional_parameters"),
                label=(
                    f"artifact runner {role} signature required positional "
                    "parameters"
                ),
            )
            signature_defaulted_positional = _string_sequence(
                signature.get("defaulted_positional_parameters"),
                label=(
                    f"artifact runner {role} signature defaulted positional "
                    "parameters"
                ),
            )
        else:
            signature_required_positional = signature_positional
            signature_defaulted_positional = ()
        if (
            signature.get("consumer_contract_id") != consumer_contract_id
            or len(path_parts) != 2
            or path_parts[1] != member
            or type(signature.get("is_async")) is not bool
            or signature.get("is_async") is not interface_is_async
            or not re.fullmatch(r"[0-9a-f]{64}", signature_sha256)
            or _bare_sha256_json(signature) != signature_sha256
            or path not in exact_signatures
            or _string_sequence(
                functions.get(member),
                label=f"artifact runner {role} exact positional parameters",
            )
            != signature_positional
            or _string_sequence(
                full_parameters.get(member),
                label=f"artifact runner {role} exact full parameters",
            )
            != signature_full
            or _string_sequence(
                keyword_only.get(member, []),
                label=f"artifact runner {role} exact keyword-only parameters",
            )
            != signature_keyword_only
            or frozen_asyncness.get(member) is not signature.get("is_async")
            or len(signature_positional) != len(set(signature_positional))
            or len(signature_required_positional)
            != len(set(signature_required_positional))
            or len(signature_defaulted_positional)
            != len(set(signature_defaulted_positional))
            or (
                role_contract_schema == _RUNNER_ROLE_CONTRACT_SCHEMA_V2
                and signature_required_positional
                + signature_defaulted_positional
                != signature_positional
            )
        ):
            raise ModelRunnerHostError(
                f"artifact runner {role} exact consumer signature differs"
            )

        if role not in known_roles:
            if "full_company" in required_for_profiles:
                raise ModelRunnerHostError(
                    "artifact runner has an unknown required full-company role"
                )
            continue
        expected_interface = _runner_interface_contract(
            role,
            interface_major=interface_majors[role],
        )
        stable_positional = tuple(
            expected_interface["positional_parameters"]
        )
        stable_host_keywords = tuple(
            expected_interface["host_keyword_parameters"]
        )
        stable_required_keywords = tuple(
            expected_interface["required_keyword_only"]
        )
        if role_contract_schema == _RUNNER_ROLE_CONTRACT_SCHEMA_V1:
            positional_call_is_compatible = (
                signature_positional == stable_positional
            )
        else:
            positional_call_is_compatible = (
                signature_positional[: len(stable_positional)]
                == stable_positional
                and len(signature_required_positional)
                <= len(stable_positional)
            )
        if (
            interface != expected_interface
            # V1 has no signed default metadata, so it remains exact. V2 may
            # add only trailing parameters explicitly signed as defaulted;
            # every required positional must still fit in the stable call.
            or not positional_call_is_compatible
            or signature_keyword_only != stable_required_keywords
            or not set(stable_positional + stable_host_keywords).issubset(
                signature_full
            )
        ):
            raise ModelRunnerHostError(
                f"artifact runner {role} interface major or host call differs"
            )
        should_require = role in _V3_FULL_COMPANY_REQUIRED_ROLES
        if ("full_company" in required_for_profiles) is not should_require:
            raise ModelRunnerHostError(
                f"artifact runner {role} activation profile differs"
            )
        members[role] = member

    if not _V3_FULL_COMPANY_REQUIRED_ROLES.issubset(members):
        raise ModelRunnerHostError(
            "artifact runner full-company role members are incomplete"
        )
    return dict(sorted(members.items()))


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
        contract_fields = {
            "schema_version",
            "contract_id",
            "functions",
            "exact_signatures",
            "full_parameters",
            "required_keyword_only",
            "exact_constants",
        }
        role_contract_present = champion.get("runner_role_contract") is not None
        if role_contract_present:
            contract_fields.update({"extensions", "frozen_asyncness"})
            contract_extensions = _closed_string_mapping(
                contract.get("extensions"),
                label="artifact runner consumer contract extensions",
            )
            if any(
                "." not in key
                or re.fullmatch(r"[a-z][a-z0-9_.-]{2,127}", key) is None
                for key in contract_extensions
            ):
                raise ModelRunnerHostError(
                    "artifact runner consumer contract extensions are invalid"
                )
        if set(contract) != contract_fields:
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
        frozen_asyncness = (
            _closed_string_mapping(
                contract.get("frozen_asyncness"),
                label="artifact runner frozen asyncness",
            )
            if role_contract_present
            else {}
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
        role_contract_members: dict[str, str] | None = None
        ingestion_custody_generation = all(
            model_constants.get(name) == expected
            for name, expected in (
                _V3_INGESTION_CUSTODY_VERSIONS.items()
            )
        )
        legacy_v3_generation = all(
            model_constants.get(name) == expected
            for name, expected in _V3_LEGACY_VERSIONS.items()
        )
        if ingestion_custody_generation or legacy_v3_generation:
            family = _GENERATION_V3
            expected_versions = dict(
                _V3_INGESTION_CUSTODY_VERSIONS
                if ingestion_custody_generation
                else _V3_LEGACY_VERSIONS
            )
            expected_provider_receipt_schema = (
                _V3_INGESTION_CUSTODY_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION
                if ingestion_custody_generation
                else _V3_LEGACY_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION
            )
            expected_role_compatibility_major = (
                _RUNNER_ROLE_INGESTION_CUSTODY_COMPATIBILITY_MAJOR
                if ingestion_custody_generation
                else _RUNNER_ROLE_LEGACY_COMPATIBILITY_MAJOR
            )
            expected_role_interface_majors = {
                role: (
                    2
                    if ingestion_custody_generation
                    and role in {"completion", "provider_receipt_binding"}
                    else 1
                )
                for role in (
                    _V3_FULL_COMPANY_REQUIRED_ROLES | _V3_OPTIONAL_ROLES
                )
            }
            required_roles = _V3_BASE_ROLES
            if any(
                raw_constants.get(name) != expected
                for name, expected in _V3_RAW_VERSIONS.items()
            ) or model_constants.get(
                "MODEL_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION"
            ) != expected_provider_receipt_schema:
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
            completion_accounting = model_constants.get(
                "MODEL_RUNNER_COMPLETION_ACCOUNTING_SCHEMA_VERSION"
            )
            if completion_accounting is not None:
                if completion_accounting != (
                    _V3_COMPLETION_ACCOUNTING_SCHEMA_VERSION
                ):
                    raise ModelRunnerHostError(
                        "artifact runner v3 completion accounting differs"
                    )
                expected_versions[
                    "MODEL_RUNNER_COMPLETION_ACCOUNTING_SCHEMA_VERSION"
                ] = _V3_COMPLETION_ACCOUNTING_SCHEMA_VERSION
                expected_champion_keys.add(
                    "completion_accounting_schema_version"
                )

            official_metadata_present = bool(
                set(champion) & _V3_OFFICIAL_BASELINE_METADATA_KEYS
            )
            if official_metadata_present and not role_contract_present:
                raise ModelRunnerHostError(
                    "artifact official baseline role contract is unavailable"
                )
            if role_contract_present:
                if not official_metadata_present:
                    raise ModelRunnerHostError(
                        "artifact official baseline declaration is incomplete"
                    )
                expected_champion_keys.update(
                    _V3_OFFICIAL_BASELINE_METADATA_KEYS
                )
                expected_champion_keys.add("runner_role_contract")
                expected_champion_keys.difference_update({
                    "raw_icp_entrypoint",
                    "entrypoint",
                    "start_entrypoint",
                    "completion_entrypoint",
                    "provider_receipt_binding_entrypoint",
                    "preflight_entrypoint",
                    "preflight_validation_entrypoint",
                    "result_validation_entrypoint",
                    "legacy_rollback_entrypoint",
                    *_V3_OFFICIAL_BASELINE_MEMBER_METADATA_KEYS.values(),
                })
                required_roles = _V3_FULL_COMPANY_REQUIRED_ROLES
                if completion_accounting != (
                    _V3_COMPLETION_ACCOUNTING_SCHEMA_VERSION
                ):
                    raise ModelRunnerHostError(
                        "artifact official baseline completion accounting differs"
                    )
                if any(
                    champion.get(key) != expected
                    for key, expected in (
                        _V3_OFFICIAL_BASELINE_SCHEMA_VERSIONS.items()
                    )
                ):
                    raise ModelRunnerHostError(
                        "artifact official baseline schema identities differ"
                    )
                for contract_key in sorted(
                    _V3_OFFICIAL_BASELINE_CONTRACT_KEYS
                ):
                    _validate_contract_identity(
                        champion.get(contract_key),
                        label=(
                            "artifact official baseline " + contract_key
                        ),
                    )
                provider_dispatch_contract = _validate_contract_identity(
                    champion.get("provider_prepare_contract"),
                    label=(
                        "artifact official baseline "
                        "provider_prepare_contract"
                    ),
                )
                _validate_provider_response_ingestion_contract(
                    champion.get("provider_response_ingestion_contract"),
                    provider_dispatch_contract_sha256=str(
                        provider_dispatch_contract["contract_sha256"]
                    ),
                    requires_ingestion_custody=(
                        ingestion_custody_generation
                    ),
                )
                for hash_key in sorted(_V3_OFFICIAL_BASELINE_HASH_KEYS):
                    if not re.fullmatch(
                        r"[0-9a-f]{64}", str(champion.get(hash_key) or "")
                    ):
                        raise ModelRunnerHostError(
                            "artifact official baseline "
                            + hash_key
                            + " is invalid"
                        )
                role_contract_members = _validate_runner_role_contract(
                    champion.get("runner_role_contract"),
                    consumer_contract_id=str(contract.get("contract_id") or ""),
                    functions=functions,
                    full_parameters=full_parameters,
                    keyword_only=keyword_only,
                    exact_signatures=exact_signatures,
                    frozen_asyncness=frozen_asyncness,
                    compatibility_major=(
                        expected_role_compatibility_major
                    ),
                    interface_majors=expected_role_interface_majors,
                )
        elif all(
            model_constants.get(name) == expected
            for name, expected in _V2_VERSIONS.items()
        ):
            if role_contract_present:
                raise ModelRunnerHostError(
                    "artifact runner v2 role contract is unsupported"
                )
            family = _GENERATION_V2
            expected_versions = _V2_VERSIONS
            required_roles = frozenset({
                "start",
                "continuation",
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
        if (
            not role_contract_present
            and set(champion) != expected_champion_keys
        ) or (
            role_contract_present
            and not expected_champion_keys.issubset(champion)
        ):
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
            if (
                "MODEL_RUNNER_COMPLETION_ACCOUNTING_SCHEMA_VERSION"
                in expected_versions
            ):
                champion_versions[
                    "MODEL_RUNNER_COMPLETION_ACCOUNTING_SCHEMA_VERSION"
                ] = champion.get("completion_accounting_schema_version")
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
            "continuation": "entrypoint",
            "completion": "completion_entrypoint",
            "preflight": "preflight_entrypoint",
            "preflight_validation": "preflight_validation_entrypoint",
            "result_validation": "result_validation_entrypoint",
            "raw_icp_input": "raw_icp_entrypoint",
            "provider_receipt_binding": (
                "provider_receipt_binding_entrypoint"
            ),
            **_V3_OFFICIAL_BASELINE_MEMBER_METADATA_KEYS,
        }
        if role_contract_present:
            members = dict(role_contract_members or {})
        else:
            members: dict[str, str] = {}
            for role in sorted(required_roles):
                (
                    expected_name,
                    expected_required_positional,
                    expected_parameters,
                    expected_keyword_only,
                ) = _MEMBER_SIGNATURES[role]
                declared_name = champion.get(member_metadata_keys[role])
                if (
                    declared_name is None
                    and family == _GENERATION_V2
                    and role == "start"
                ):
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
                        keyword_only.get(expected_name, []),
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
            required_raw_sources = {
                _V3_RAW_VERSIONS["SITE_RAW_ICP_SOURCE_SCHEMA"],
                _V3_RAW_VERSIONS["LAB_RAW_ICP_SOURCE_SCHEMA"],
            }
            if (
                (role_contract_present and not required_raw_sources.issubset(
                    raw_sources
                ))
                or (
                    not role_contract_present
                    and raw_sources != tuple(sorted(required_raw_sources))
                )
                or len(raw_sources) != len(set(raw_sources))
                or champion.get("raw_icp_envelope_schema_version") != (
                _V3_RAW_VERSIONS["RAW_ICP_ENVELOPE_SCHEMA_VERSION"]
                )
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
                "completion_custody_fields": (
                    [
                        "provider_response_ingestion",
                        "provider_receipt_ref",
                        "provider_receipt_sha256",
                        "provider_identity_sha256",
                    ]
                    if ingestion_custody_generation
                    else [
                        "provider_receipt_ref",
                        "provider_receipt_sha256",
                        "provider_identity_sha256",
                    ]
                ),
            }
            if role_contract_present and (
                "normalization_prepare_legacy" in members
            ):
                dispatch_contract = _validate_contract_identity(
                    normalization.get("dispatch_contract"),
                    label="artifact normalization dispatch contract",
                )
                expected_normalization.update({
                    "dispatch_schema_version": (
                        "model-runner-normalization-dispatch:v1"
                    ),
                    "dispatch_entrypoint": (
                        members["normalization_prepare_legacy"]
                    ),
                    "dispatch_contract": dispatch_contract,
                })
            if (
                (
                    role_contract_present
                    and any(
                        normalization.get(key) != expected
                        for key, expected in expected_normalization.items()
                    )
                )
                or (
                    not role_contract_present
                    and normalization != expected_normalization
                )
            ):
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

    @property
    def supports_official_baseline(self) -> bool:
        return _V3_OFFICIAL_BASELINE_ROLES.issubset(self.members)

    @property
    def supports_provider_response_ingestion(self) -> bool:
        return "provider_response_ingestion" in self.members

    @property
    def requires_raw_provider_response_custody(self) -> bool:
        """Whether completion consumes raw host bytes plus ingestion proof."""

        return self.versions.get(
            "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION"
        ) == _V3_INGESTION_CUSTODY_VERSIONS[
            "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION"
        ]

    def official_contract_sha256(self, metadata_key: str) -> str:
        if not self.supports_official_baseline:
            raise ModelRunnerHostError(
                "artifact runner generation has no official baseline bundle"
            )
        if metadata_key not in _V3_OFFICIAL_BASELINE_CONTRACT_KEYS:
            raise ModelRunnerHostError(
                "artifact official baseline contract key is unsupported"
            )
        identity = _validate_contract_identity(
            self.champion_execution.get(metadata_key),
            label=f"artifact official baseline {metadata_key}",
        )
        return "sha256:" + str(identity["contract_sha256"])

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

    def build_host_capability_manifest(
        self,
        *,
        bindings: Sequence[Mapping[str, Any]],
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def project_runner_result_for_benchmark(
        self,
        terminal_result: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def build_official_baseline_execution(
        self,
        *,
        release_identity: Mapping[str, Any],
        protocol_generation_sha256: str,
        protected_action_authority_sha256: str,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def prepare_runner_provider_request(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def ingest_runner_provider_response(
        self,
        action: Mapping[str, Any],
        host_response: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def prepare_runner_normalization_request(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def model_runner_provider_compiler_inventory(
        self,
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def runner_provider_compiler_preflight(
        self,
        host_capability_manifest: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def execute_runner_verifier_action(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def runner_official_host_binding_catalog(
        self,
        *,
        member_name: str,
    ) -> Mapping[str, Any]: ...

    def build_runner_official_host_capability_manifest(
        self,
        availability: Mapping[str, Any],
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

    def _official_transport_method(self, method_name: str) -> Any:
        if not self.protocol_generation.supports_official_baseline:
            raise ModelRunnerHostError(
                "artifact runner generation has no official baseline bundle"
            )
        method = getattr(self._transport, method_name, None)
        if not callable(method):
            raise ModelRunnerHostError(
                f"artifact transport method {method_name} is unavailable"
            )
        return method

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
            member_name=generation.member("raw_icp_input"),
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
            member_name=self.protocol_generation.member("continuation"),
        )
        return self._validate_state(result, "artifact continuation")

    @property
    def release_identity(self) -> Mapping[str, Any]:
        return dict(self._release_identity)

    @property
    def artifact_provider_receipt_binding_required(self) -> bool:
        """Whether this exact generation declared the receipt member."""

        return self.protocol_generation.supports_provider_receipt_binding

    @property
    def requires_raw_provider_response_custody(self) -> bool:
        """Whether this generation requires exact raw-response replay."""

        return self.protocol_generation.requires_raw_provider_response_custody

    @property
    def artifact_official_baseline_supported(self) -> bool:
        return self.protocol_generation.supports_official_baseline

    def _validate_host_capability_manifest_result(
        self,
        value: Any,
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "binding_contracts_sha256",
            "bindings",
            "manifest_sha256",
        }:
            raise ModelRunnerHostError(
                "artifact host capability manifest is not closed"
            )
        manifest = dict(value)
        declared_schema = generation.champion_execution[
            "host_capability_manifest_schema_version"
        ]
        returned_bindings = manifest.get("bindings")
        if (
            manifest.get("schema_version") != declared_schema
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(manifest.get("binding_contracts_sha256") or ""),
            )
            or not isinstance(returned_bindings, list)
            or any(
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "schema_version",
                    "action_type",
                    "tool_id",
                    "binding_contract_sha256",
                    "response_schema_version",
                    "available",
                    "idempotency",
                    "max_response_bytes",
                }
                for item in returned_bindings
            )
            or manifest.get("manifest_sha256")
            != _bare_sha256_json(
                {
                    key: item
                    for key, item in manifest.items()
                    if key != "manifest_sha256"
                }
            )
        ):
            raise ModelRunnerHostError(
                "artifact host capability manifest is invalid"
            )
        return manifest

    def build_host_capability_manifest(
        self,
        *,
        bindings: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "build_host_capability_manifest"
        )(
            bindings=bindings,
            member_name=generation.member("host_capability_manifest"),
        )
        return self._validate_host_capability_manifest_result(result)

    def project_runner_result_for_benchmark(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "project_runner_result_for_benchmark"
        )(
            value,
            start_request=start_request,
            expected_release_identity=self._release_identity,
            member_name=generation.member("benchmark_projection"),
        )
        if not isinstance(result, Mapping):
            raise ModelRunnerHostError(
                "artifact benchmark projection is invalid"
            )
        return dict(result)

    def build_official_baseline_execution(
        self,
        *,
        protected_action_authority_sha256: str,
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        if not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(protected_action_authority_sha256 or ""),
        ):
            raise ModelRunnerHostError(
                "protected action authority identity is invalid"
            )
        result = self._official_transport_method(
            "build_official_baseline_execution"
        )(
            release_identity=self._release_identity,
            protocol_generation_sha256=(
                generation.protocol_generation_sha256
            ),
            protected_action_authority_sha256=(
                protected_action_authority_sha256
            ),
            member_name=generation.member("official_baseline_execution"),
        )
        expected = {
            "schema_version": generation.champion_execution[
                "official_baseline_execution_schema_version"
            ],
            "runner_family": "exact_model_runner:v3",
            "execution_mode": "measured_lab",
            "release_identity_sha256": sha256_json(self._release_identity),
            "protocol_generation_sha256": (
                generation.protocol_generation_sha256
            ),
            "benchmark_projection_sha256": (
                generation.official_contract_sha256(
                    "benchmark_projection_contract"
                )
            ),
            "protected_action_authority_sha256": (
                protected_action_authority_sha256
            ),
        }
        if not isinstance(result, Mapping) or dict(result) != expected:
            raise ModelRunnerHostError(
                "artifact official baseline selection differs"
            )
        return expected

    def provider_compiler_inventory(self) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "model_runner_provider_compiler_inventory"
        )(
            member_name=generation.member("provider_compiler_inventory")
        )
        if (
            not isinstance(result, Mapping)
            or result.get("schema_version")
            != generation.champion_execution[
                "provider_compiler_inventory_schema_version"
            ]
            or result.get("inventory_sha256")
            != generation.champion_execution[
                "provider_response_ingestion_contract"
            ].get("compiler_inventory_sha256")
        ):
            raise ModelRunnerHostError(
                "artifact provider compiler inventory is invalid"
            )
        return dict(result)

    def provider_compiler_preflight(
        self,
        host_capability_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "runner_provider_compiler_preflight"
        )(
            host_capability_manifest,
            member_name=generation.member("provider_compiler_preflight"),
        )
        if (
            not isinstance(result, Mapping)
            or result.get("schema_version")
            != generation.champion_execution[
                "provider_compiler_preflight_schema_version"
            ]
        ):
            raise ModelRunnerHostError(
                "artifact provider compiler preflight is invalid"
            )
        return dict(result)

    def official_host_binding_catalog(self) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "runner_official_host_binding_catalog"
        )(
            member_name=generation.member(
                "official_host_binding_catalog"
            )
        )
        if not isinstance(result, Mapping) or set(result) != {
            "schema_version",
            "bindings",
            "binding_contracts_sha256",
            "catalog_sha256",
        }:
            raise ModelRunnerHostError(
                "artifact official host binding catalog is not closed"
            )
        catalog = dict(result)
        bindings = catalog.get("bindings")
        contract = _validate_contract_identity(
            generation.champion_execution[
                "official_host_binding_catalog_contract"
            ],
            label="artifact official host binding catalog contract",
        )
        if (
            catalog.get("schema_version")
            != generation.champion_execution[
                "official_host_binding_catalog_schema_version"
            ]
            or not isinstance(bindings, list)
            or any(
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "schema_version",
                    "action_type",
                    "tool_id",
                    "binding_contract_sha256",
                    "response_schema_version",
                    "idempotency",
                    "max_response_bytes",
                }
                or not isinstance(item.get("action_type"), str)
                or not item.get("action_type")
                or not isinstance(item.get("tool_id"), str)
                or not item.get("tool_id")
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(item.get("binding_contract_sha256") or ""),
                )
                or not isinstance(item.get("response_schema_version"), str)
                or not item.get("response_schema_version")
                or type(item.get("max_response_bytes")) is not int
                or item.get("max_response_bytes", 0) <= 0
                for item in bindings
            )
            or catalog.get("binding_contracts_sha256")
            != _bare_sha256_json(bindings)
            or catalog.get("catalog_sha256")
            != _bare_sha256_json(
                {
                    key: item
                    for key, item in catalog.items()
                    if key != "catalog_sha256"
                }
            )
            or catalog.get("binding_contracts_sha256")
            != contract.get("binding_contracts_sha256")
            or catalog.get("catalog_sha256")
            != contract.get("catalog_sha256")
        ):
            raise ModelRunnerHostError(
                "artifact official host binding catalog is invalid"
            )
        return catalog

    def build_official_host_capability_manifest(
        self,
        availability: Mapping[str, bool],
    ) -> Mapping[str, Any]:
        if not isinstance(availability, Mapping) or any(
            not isinstance(key, str) or not key or type(value) is not bool
            for key, value in availability.items()
        ):
            raise ModelRunnerHostError(
                "official host binding availability is invalid"
            )
        generation = self.protocol_generation
        result = self._official_transport_method(
            "build_runner_official_host_capability_manifest"
        )(
            dict(availability),
            member_name=generation.member(
                "official_host_capability_manifest"
            ),
        )
        manifest = self._validate_host_capability_manifest_result(result)
        catalog = self.official_host_binding_catalog()
        catalog_bindings = catalog["bindings"]
        tool_ids = [str(item["tool_id"]) for item in catalog_bindings]
        expected_availability = set(tool_ids)
        expected_bindings = [
            {**dict(item), "available": availability[item["tool_id"]]}
            for item in catalog_bindings
        ]
        if (
            len(expected_availability) != len(tool_ids)
            or set(availability) != expected_availability
            or manifest["binding_contracts_sha256"]
            != catalog["binding_contracts_sha256"]
            or manifest["bindings"] != expected_bindings
        ):
            raise ModelRunnerHostError(
                "artifact official host capability manifest differs from catalog"
            )
        return manifest

    def execute_verifier_action(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "execute_runner_verifier_action"
        )(
            action,
            member_name=generation.member("verifier_execution"),
        )
        expected_action_sha256 = str(action.get("action_sha256") or "")
        expected_action_type = str(action.get("action_type") or "")
        if not isinstance(result, Mapping) or set(result) != {
            "schema_version",
            "action_sha256",
            "action_type",
            "calls",
            "cost_credits",
            "provider_receipt_allowed",
            "result",
            "result_sha256",
            "execution_sha256",
        }:
            raise ModelRunnerHostError(
                "artifact verifier execution is not closed"
            )
        execution = dict(result)
        verifier_result = execution.get("result")
        if (
            execution.get("schema_version")
            != generation.champion_execution[
                "verifier_execution_schema_version"
            ]
            or expected_action_type
            not in {"verify_company", "verify_intent", "verify_contact"}
            or execution.get("action_type") != expected_action_type
            or not re.fullmatch(r"[0-9a-f]{64}", expected_action_sha256)
            or execution.get("action_sha256") != expected_action_sha256
            or execution.get("calls") != 0
            or type(execution.get("cost_credits")) not in {int, float}
            or float(execution["cost_credits"]) != 0.0
            or execution.get("provider_receipt_allowed") is not False
            or not isinstance(verifier_result, Mapping)
            or execution.get("result_sha256")
            != _bare_sha256_json(verifier_result)
            or execution.get("execution_sha256")
            != _bare_sha256_json(
                {
                    key: item
                    for key, item in execution.items()
                    if key != "execution_sha256"
                }
            )
        ):
            raise ModelRunnerHostError(
                "artifact verifier execution is invalid"
            )
        return execution

    def prepare_provider_request(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "prepare_runner_provider_request"
        )(
            action,
            member_name=generation.member("provider_prepare"),
        )
        if (
            not isinstance(result, Mapping)
            or result.get("schema_version")
            != generation.champion_execution[
                "provider_prepare_schema_version"
            ]
        ):
            raise ModelRunnerHostError(
                "artifact provider dispatch request is invalid"
            )
        return dict(result)

    def ingest_provider_response(
        self,
        action: Mapping[str, Any],
        host_response: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Invoke model-owned parsing and validate only its signed envelope."""

        generation = self.protocol_generation
        if not generation.supports_provider_response_ingestion:
            raise ModelRunnerHostError(
                "artifact runner generation has no provider response ingestor"
            )
        if (
            not isinstance(host_response, Mapping)
            or set(host_response) != _HOST_PROVIDER_RESPONSE_FIELDS
            or host_response.get("schema_version")
            != "host-provider-response:v1"
            or not isinstance(host_response.get("provider"), str)
            or not host_response.get("provider")
            or isinstance(host_response.get("status_code"), bool)
            or not isinstance(host_response.get("status_code"), int)
            or not 100 <= host_response["status_code"] <= 599
            or not isinstance(host_response.get("body"), Mapping)
        ):
            raise ModelRunnerHostError(
                "host provider response envelope is invalid"
            )
        dispatch = self.prepare_provider_request(action)
        result = self._official_transport_method(
            "ingest_runner_provider_response"
        )(
            action,
            host_response,
            member_name=generation.member("provider_response_ingestion"),
        )
        contract = _validate_provider_response_ingestion_contract(
            generation.champion_execution[
                "provider_response_ingestion_contract"
            ],
            provider_dispatch_contract_sha256=str(
                _validate_contract_identity(
                    generation.champion_execution[
                        "provider_prepare_contract"
                    ],
                    label="artifact provider dispatch contract",
                )["contract_sha256"]
            ),
            requires_ingestion_custody=(
                generation.requires_raw_provider_response_custody
            ),
        )
        if not isinstance(result, Mapping):
            raise ModelRunnerHostError(
                "artifact provider response ingestion is invalid"
            )
        receipt = dict(result)
        parsed_response = receipt.get("parsed_response")
        ingestion_payload = {
            key: item
            for key, item in receipt.items()
            if key != "ingestion_sha256"
        }
        if (
            set(receipt) != _PROVIDER_RESPONSE_INGESTION_FIELDS
            or receipt.get("schema_version")
            != generation.champion_execution[
                "provider_response_ingestion_schema_version"
            ]
            or receipt.get("action_sha256")
            != action.get("action_sha256")
            or receipt.get("dispatch_sha256")
            != dispatch.get("dispatch_sha256")
            or receipt.get("compiler_id") != dispatch.get("compiler_id")
            or receipt.get("compiler_contract_sha256")
            != dispatch.get("compiler_contract_sha256")
            or receipt.get("request_sha256")
            != dispatch.get("request_sha256")
            or receipt.get("host_response_schema_version")
            != contract["host_response_schema_version"]
            or receipt.get("host_response_sha256")
            != _bare_sha256_json(host_response)
            or receipt.get("provider") != host_response.get("provider")
            or receipt.get("provider") != dispatch.get("provider")
            or receipt.get("parsed_response_schema_version")
            not in contract["parsed_response_schema_versions"]
            or not isinstance(parsed_response, Mapping)
            or parsed_response.get("schema_version")
            != receipt.get("parsed_response_schema_version")
            or receipt.get("parsed_response_sha256")
            != _bare_sha256_json(parsed_response)
            or receipt.get("ingestion_sha256")
            != _bare_sha256_json(ingestion_payload)
        ):
            raise ModelRunnerHostError(
                "artifact provider response ingestion identity differs"
            )
        return receipt

    def prepare_normalization_request(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        generation = self.protocol_generation
        result = self._official_transport_method(
            "prepare_runner_normalization_request"
        )(
            action,
            member_name=generation.member("normalization_prepare_legacy"),
        )
        expected_action_sha256 = str(action.get("action_sha256") or "")
        if not isinstance(result, Mapping) or set(result) != {
            "schema_version",
            "action_sha256",
            "provider",
            "method",
            "url",
            "credential_binding",
            "static_headers",
            "body",
            "body_sha256",
            "call_cap",
            "credit_cap",
            "timeout_seconds",
            "max_response_bytes",
            "request_sha256",
        }:
            raise ModelRunnerHostError(
                "artifact normalization dispatch is not closed"
            )
        dispatch = dict(result)
        body = dispatch.get("body")
        normalization = generation.champion_execution[
            "normalization_action"
        ]
        if (
            dispatch.get("schema_version")
            != normalization["dispatch_schema_version"]
            or not re.fullmatch(r"[0-9a-f]{64}", expected_action_sha256)
            or dispatch.get("action_sha256") != expected_action_sha256
            or not isinstance(body, Mapping)
            or dispatch.get("body_sha256") != _bare_sha256_json(body)
            or dispatch.get("request_sha256")
            != _bare_sha256_json(
                {
                    key: item
                    for key, item in dispatch.items()
                    if key != "request_sha256"
                }
            )
        ):
            raise ModelRunnerHostError(
                "artifact normalization dispatch is invalid"
            )
        return dispatch

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
        result_payload = {
            "outcome": result.outcome,
            "reason_code": result.reason_code,
            "provider_response": result.provider_response,
            "calls": result.calls,
            "cost_credits": result.cost_credits,
            "latency_ms": result.latency_ms,
            "provider_receipt_ref": result.provider_receipt_ref,
            "provider_receipt_sha256": result.provider_receipt_sha256,
            "provider_identity_sha256": result.provider_identity_sha256,
        }
        if self.protocol_generation.requires_raw_provider_response_custody:
            result_payload["provider_response_ingestion"] = (
                result.model_provider_response_ingestion
            )
        completion = self._transport.build_runner_completion(
            action,
            result_payload,
            member_name=self.protocol_generation.member("completion"),
        )
        if not isinstance(completion, Mapping) or completion.get(
            "schema_version"
        ) != self.protocol_generation.version(
            "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION"
        ):
            raise ModelRunnerHostError("artifact completion is invalid")
        if (
            self.protocol_generation.requires_raw_provider_response_custody
            and action.get("action_type") in _MODEL_PROVIDER_ACTION_TYPES
        ):
            ingestion = result.model_provider_response_ingestion
            if ingestion is None:
                expected = {
                    "provider_response": None,
                    "provider_response_sha256": _bare_sha256_json(None),
                    "provider_response_ingestion_sha256": _ZERO_SHA256,
                    "provider_dispatch_sha256": _ZERO_SHA256,
                    "provider_request_sha256": _ZERO_SHA256,
                    "host_provider_response_sha256": _ZERO_SHA256,
                }
            elif isinstance(ingestion, Mapping):
                expected = {
                    "provider_response": ingestion.get("parsed_response"),
                    "provider_response_sha256": ingestion.get(
                        "parsed_response_sha256"
                    ),
                    "provider_response_ingestion_sha256": ingestion.get(
                        "ingestion_sha256"
                    ),
                    "provider_dispatch_sha256": ingestion.get(
                        "dispatch_sha256"
                    ),
                    "provider_request_sha256": ingestion.get(
                        "request_sha256"
                    ),
                    "host_provider_response_sha256": ingestion.get(
                        "host_response_sha256"
                    ),
                }
            else:
                raise ModelRunnerHostError(
                    "artifact completion provider ingestion is invalid"
                )
            if (
                completion.get("action_sha256")
                != action.get("action_sha256")
                or any(
                    completion.get(field) != value
                    for field, value in expected.items()
                )
            ):
                raise ModelRunnerHostError(
                    "artifact completion provider custody differs"
                )
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
        result_payload = {
            "provider_response": result.provider_response,
            "provider_receipt_ref": result.provider_receipt_ref,
            "provider_identity_sha256": result.provider_identity_sha256,
            "calls": result.calls,
            "cost_credits": result.cost_credits,
            "latency_ms": result.latency_ms,
        }
        if generation.requires_raw_provider_response_custody:
            result_payload["provider_response_ingestion"] = (
                result.model_provider_response_ingestion
            )
        binding = self._transport.build_runner_provider_receipt_binding(
            action,
            result_payload,
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
        if generation.requires_raw_provider_response_custody:
            ingestion = result.model_provider_response_ingestion
            if ingestion is None:
                expected_custody = {
                    "provider_response_sha256": _bare_sha256_json(None),
                    "provider_response_ingestion_sha256": _ZERO_SHA256,
                    "provider_dispatch_sha256": _ZERO_SHA256,
                    "provider_request_sha256": _ZERO_SHA256,
                    "host_provider_response_sha256": _ZERO_SHA256,
                }
            elif isinstance(ingestion, Mapping):
                expected_custody = {
                    "provider_response_sha256": ingestion.get(
                        "parsed_response_sha256"
                    ),
                    "provider_response_ingestion_sha256": ingestion.get(
                        "ingestion_sha256"
                    ),
                    "provider_dispatch_sha256": ingestion.get(
                        "dispatch_sha256"
                    ),
                    "provider_request_sha256": ingestion.get(
                        "request_sha256"
                    ),
                    "host_provider_response_sha256": ingestion.get(
                        "host_response_sha256"
                    ),
                }
            else:
                raise ModelRunnerHostError(
                    "artifact provider receipt ingestion is invalid"
                )
            if (
                binding.get("action_sha256")
                != action.get("action_sha256")
                or any(
                    binding.get(field) != value
                    for field, value in expected_custody.items()
                )
            ):
                raise ModelRunnerHostError(
                    "artifact provider receipt custody differs"
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
