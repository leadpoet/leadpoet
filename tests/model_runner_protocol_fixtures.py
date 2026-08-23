"""Closed v2/v3 artifact runner declarations for host boundary tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


CONTRACT_HASH = "c" * 64

_MEMBERS = {
    "build_runner_start": (
        [],
        [
            "input",
            "execution_mode",
            "target_count",
            "evaluated_on",
            "host_capability_manifest",
            "release_identity",
        ],
        [
            "input",
            "execution_mode",
            "target_count",
            "evaluated_on",
            "host_capability_manifest",
            "release_identity",
        ],
    ),
    "continue_runner": (
        ["start_request"],
        [
            "start_request",
            "expected_release_identity",
            "continuation",
            "completion",
        ],
        ["expected_release_identity"],
    ),
    "build_runner_completion": (
        ["action", "result"],
        ["action", "result"],
        [],
    ),
    "runner_preflight": (
        ["host_capability_manifest", "release_identity"],
        ["host_capability_manifest", "release_identity", "execution_mode"],
        ["execution_mode"],
    ),
    "validate_runner_preflight": (
        ["value"],
        [
            "value",
            "host_capability_manifest",
            "release_identity",
            "execution_mode",
        ],
        ["host_capability_manifest", "release_identity", "execution_mode"],
    ),
    "validate_runner_result": (
        ["value"],
        ["value", "start_request", "expected_release_identity"],
        ["start_request", "expected_release_identity"],
    ),
    "build_raw_runner_input": (
        ["payload"],
        ["payload", "source_schema"],
        ["source_schema"],
    ),
    "build_runner_provider_receipt_binding": (
        ["action", "result"],
        ["action", "result"],
        [],
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

_RAW_CONSTANTS = {
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


def runner_release_identity(
    family: str = "v3",
    *,
    contract_hash: str = CONTRACT_HASH,
    **values: Any,
) -> dict[str, Any]:
    versions = _V3_VERSIONS if family == "v3" else _V2_VERSIONS
    return {
        "schema_version": versions["MODEL_RELEASE_IDENTITY_SCHEMA_VERSION"],
        "consumer_contract_sha256": contract_hash,
        **values,
    }


def runner_declaration(
    family: str = "v3",
    *,
    contract_hash: str = CONTRACT_HASH,
) -> dict[str, Any]:
    if family not in {"v2", "v3"}:
        raise ValueError("fixture runner family must be v2 or v3")
    versions = dict(_V3_VERSIONS if family == "v3" else _V2_VERSIONS)
    model_constants = {
        **versions,
        "MODEL_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION": (
            "model-provider-receipt-binding:v1"
        ),
    }
    champion: dict[str, Any] = {
        "release_identity_schema_version": versions[
            "MODEL_RELEASE_IDENTITY_SCHEMA_VERSION"
        ],
        "start_schema_version": versions["MODEL_RUNNER_START_SCHEMA_VERSION"],
        "action_schema_version": versions["MODEL_RUNNER_ACTION_SCHEMA_VERSION"],
        "completion_schema_version": versions[
            "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION"
        ],
        "preflight_schema_version": versions[
            "MODEL_RUNNER_PREFLIGHT_SCHEMA_VERSION"
        ],
        "result_schema_version": versions["MODEL_RUNNER_RESULT_SCHEMA_VERSION"],
        "receipt_schema_version": versions[
            "MODEL_RUNNER_RECEIPT_SCHEMA_VERSION"
        ],
        "entrypoint": "continue_runner",
        "completion_entrypoint": "build_runner_completion",
        "preflight_entrypoint": "runner_preflight",
        "preflight_validation_entrypoint": "validate_runner_preflight",
        "result_validation_entrypoint": "validate_runner_result",
        "legacy_rollback_entrypoint": "run_icp",
    }
    required_names = {
        "build_runner_start",
        "continue_runner",
        "build_runner_completion",
        "runner_preflight",
        "validate_runner_preflight",
        "validate_runner_result",
    }
    exact_constants: dict[str, Mapping[str, Any]] = {
        "sourcing_model/model_runner.py": model_constants,
    }
    if family == "v3":
        champion.update({
            "raw_icp_envelope_schema_version": (
                "model-raw-icp-envelope:v1"
            ),
            "raw_icp_source_schemas": [
                "leadpoet-research-lab-benchmark-icp:v1",
                "leadpoet-site-company-request:v1",
            ],
            "raw_icp_entrypoint": "build_raw_runner_input",
            "start_entrypoint": "build_runner_start",
            "continuation_schema_version": versions[
                "MODEL_RUNNER_CONTINUATION_SCHEMA_VERSION"
            ],
            "provider_receipt_binding_entrypoint": (
                "build_runner_provider_receipt_binding"
            ),
            "normalization_action": {
                "action_type": "normalize_icp",
                "stage": "icp_normalization",
                "tool_id": "normalization.openrouter_json_schema",
                "request_schema_version": (
                    "model-normalization-action-arguments:v1"
                ),
                "response_schema_version": (
                    "model-icp-normalization-provider-response:v1"
                ),
                "provider_receipt_binding_schema_version": (
                    "model-provider-receipt-binding:v1"
                ),
                "call_cap": 1,
                "credit_cap": 1.0,
                "timeout_seconds": 120.0,
                "completion_custody_fields": [
                    "provider_receipt_ref",
                    "provider_receipt_sha256",
                    "provider_identity_sha256",
                ],
            },
        })
        required_names.update({
            "build_raw_runner_input",
            "build_runner_provider_receipt_binding",
        })
        exact_constants["sourcing_model/raw_icp_normalization.py"] = (
            _RAW_CONSTANTS
        )
    functions = {
        name: list(_MEMBERS[name][0]) for name in sorted(required_names)
    }
    full_parameters = {
        name: list(_MEMBERS[name][1]) for name in sorted(required_names)
    }
    keyword_only = {
        name: list(_MEMBERS[name][2]) for name in sorted(required_names)
    }
    declaration = {
        "schema_version": (
            "leadpoet.research_lab.artifact_runner_declaration.v1"
        ),
        "champion_execution": champion,
        "consumer_contract_sha256": contract_hash,
        "consumer_contract": {
            "schema_version": 1,
            "contract_id": "test-runner-contract-" + family,
            "functions": functions,
            "exact_signatures": [
                "research_lab_adapter.py:" + name
                for name in sorted(required_names)
            ],
            "full_parameters": full_parameters,
            "required_keyword_only": keyword_only,
            "exact_constants": exact_constants,
        },
    }
    return deepcopy(declaration)


def runner_versions(family: str = "v3") -> dict[str, str]:
    return dict(_V3_VERSIONS if family == "v3" else _V2_VERSIONS)
