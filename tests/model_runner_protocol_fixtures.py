"""Closed v2/v3 artifact runner declarations for host boundary tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping


CONTRACT_HASH = "c" * 64

_MEMBERS = {
    "build_host_capability_manifest": (
        ["bindings"],
        ["bindings"],
        [],
    ),
    "project_runner_result_for_benchmark": (
        ["value"],
        ["value", "start_request", "expected_release_identity"],
        ["start_request", "expected_release_identity"],
    ),
    "build_official_baseline_execution": (
        [],
        [
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ],
        [
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ],
    ),
    "prepare_runner_provider_request": (
        ["action"],
        ["action"],
        [],
    ),
    "ingest_runner_provider_response": (
        ["action", "host_response"],
        ["action", "host_response"],
        [],
    ),
    "prepare_runner_normalization_request": (
        ["action"],
        ["action"],
        [],
    ),
    "model_runner_provider_compiler_inventory": ([], [], []),
    "runner_provider_compiler_preflight": (
        ["host_capability_manifest"],
        ["host_capability_manifest"],
        [],
    ),
    "execute_runner_verifier_action": (["action"], ["action"], []),
    "runner_official_host_binding_catalog": ([], [], []),
    "build_runner_official_host_capability_manifest": (
        ["availability"],
        ["availability"],
        [],
    ),
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
_V4_VERSIONS = {
    **_V3_VERSIONS,
    "MODEL_RUNNER_COMPLETION_SCHEMA_VERSION": "model-runner-completion:v4",
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

_ROLE_MEMBERS = {
    "raw_icp_input": "build_raw_runner_input",
    "start": "build_runner_start",
    "continuation": "continue_runner",
    "completion": "build_runner_completion",
    "provider_receipt_binding": "build_runner_provider_receipt_binding",
    "preflight": "runner_preflight",
    "preflight_validation": "validate_runner_preflight",
    "result_validation": "validate_runner_result",
    "benchmark_projection": "project_runner_result_for_benchmark",
    "host_capability_manifest": "build_host_capability_manifest",
    "official_baseline_execution": "build_official_baseline_execution",
    "provider_prepare": "prepare_runner_provider_request",
    "provider_response_ingestion": "ingest_runner_provider_response",
    "provider_compiler_inventory": "model_runner_provider_compiler_inventory",
    "provider_compiler_preflight": "runner_provider_compiler_preflight",
    "verifier_execution": "execute_runner_verifier_action",
    "official_host_binding_catalog": "runner_official_host_binding_catalog",
    "official_host_capability_manifest": (
        "build_runner_official_host_capability_manifest"
    ),
    "normalization_prepare_legacy": "prepare_runner_normalization_request",
}
_ROLE_HOST_CALLS = {
    "raw_icp_input": (["payload"], ["source_schema"], ["source_schema"]),
    "start": ([], list(_MEMBERS["build_runner_start"][1]), list(
        _MEMBERS["build_runner_start"][2]
    )),
    "continuation": (
        ["start_request"],
        ["expected_release_identity", "continuation", "completion"],
        ["expected_release_identity"],
    ),
    "completion": (["action", "result"], [], []),
    "provider_receipt_binding": (["action", "result"], [], []),
    "preflight": (
        ["host_capability_manifest", "release_identity"],
        ["execution_mode"],
        ["execution_mode"],
    ),
    "preflight_validation": (
        ["value"],
        ["host_capability_manifest", "release_identity", "execution_mode"],
        ["host_capability_manifest", "release_identity", "execution_mode"],
    ),
    "result_validation": (
        ["value"],
        ["start_request", "expected_release_identity"],
        ["start_request", "expected_release_identity"],
    ),
    "benchmark_projection": (
        ["value"],
        ["start_request", "expected_release_identity"],
        ["start_request", "expected_release_identity"],
    ),
    "host_capability_manifest": (["bindings"], [], []),
    "official_baseline_execution": (
        [],
        [
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ],
        [
            "release_identity",
            "protocol_generation_sha256",
            "protected_action_authority_sha256",
        ],
    ),
    "provider_prepare": (["action"], [], []),
    "provider_response_ingestion": (
        ["action", "host_response"],
        [],
        [],
    ),
    "provider_compiler_inventory": ([], [], []),
    "provider_compiler_preflight": (["host_capability_manifest"], [], []),
    "verifier_execution": (["action"], [], []),
    "official_host_binding_catalog": ([], [], []),
    "official_host_capability_manifest": (["availability"], [], []),
    "normalization_prepare_legacy": (["action"], [], []),
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _role_contract(
    contract_id: str,
    *,
    ingestion_custody: bool = False,
) -> dict[str, Any]:
    required_roles = tuple(sorted(
        role for role in _ROLE_MEMBERS
        if role != "normalization_prepare_legacy"
    ))
    roles: dict[str, Any] = {}
    for role, member in sorted(_ROLE_MEMBERS.items()):
        positional, host_keywords, required_keyword_only = (
            _ROLE_HOST_CALLS[role]
        )
        interface_major = (
            2
            if ingestion_custody
            and role in {"completion", "provider_receipt_binding"}
            else 1
        )
        interface = {
            "interface_id": "leadpoet.model_runner." + role,
            "interface_major": interface_major,
            "positional_parameters": list(positional),
            "host_keyword_parameters": list(host_keywords),
            "required_keyword_only": list(required_keyword_only),
            "is_async": False,
        }
        signature = {
            "consumer_contract_id": contract_id,
            "consumer_contract_path": "research_lab_adapter.py:" + member,
            "positional_parameters": list(_MEMBERS[member][0]),
            "full_parameters": list(_MEMBERS[member][1]),
            "required_keyword_only": list(_MEMBERS[member][2]),
            "is_async": False,
        }
        roles[role] = {
            "interface_id": interface["interface_id"],
            "interface_major": interface_major,
            "interface_contract": interface,
            "interface_contract_sha256": _canonical_sha256(interface),
            "adapter_member": member,
            "consumer_signature": signature,
            "consumer_signature_sha256": _canonical_sha256(signature),
            "required_for_profiles": (
                ["full_company"] if role in required_roles else []
            ),
        }
    payload = {
        "schema_version": "model-runner-role-contract:v1",
        "compatibility_major": 2 if ingestion_custody else 1,
        "consumer_contract_id": contract_id,
        "roles": roles,
        "activation_profiles": {
            "full_company": {
                "required_roles": list(required_roles),
                "minimum_interface_major": {
                    role: roles[role]["interface_major"]
                    for role in required_roles
                },
                "unknown_required_role_policy": (
                    "reject_before_preflight_or_spend"
                ),
            }
        },
        "additive_compatibility": {
            "known_required_roles": (
                "bind_stable_interface_and_exact_signed_consumer_signature"
            ),
            "unknown_required_roles": "reject_before_preflight_or_spend",
            "unknown_optional_roles": "accept_ignore_and_hash_bind",
            "member_names_are_discovered_from_roles": True,
            "commit_allowlists": False,
            "exact_member_tuple_allowlists": False,
        },
        "extensions": {},
        "canonical_json": "utf8-json-sort-keys-compact-ascii-no-nan",
        "hash_algorithm": "sha256",
    }
    return {**payload, "contract_sha256": _canonical_sha256(payload)}


def runner_release_identity(
    family: str = "v3",
    *,
    contract_hash: str = CONTRACT_HASH,
    **values: Any,
) -> dict[str, Any]:
    versions = (
        _V4_VERSIONS
        if family == "v4"
        else _V3_VERSIONS
        if family == "v3"
        else _V2_VERSIONS
    )
    return {
        "schema_version": versions["MODEL_RELEASE_IDENTITY_SCHEMA_VERSION"],
        "consumer_contract_sha256": contract_hash,
        **values,
    }


def runner_declaration(
    family: str = "v3",
    *,
    contract_hash: str = CONTRACT_HASH,
    official_baseline: bool = False,
) -> dict[str, Any]:
    if family not in {"v2", "v3", "v4"}:
        raise ValueError("fixture runner family must be v2, v3, or v4")
    contract_id = "test-runner-contract-" + family
    versions = dict(
        _V4_VERSIONS
        if family == "v4"
        else _V3_VERSIONS
        if family == "v3"
        else _V2_VERSIONS
    )
    ingestion_custody = family == "v4"
    model_constants = {
        **versions,
        "MODEL_PROVIDER_RECEIPT_BINDING_SCHEMA_VERSION": (
            "model-provider-receipt-binding:v2"
            if ingestion_custody
            else "model-provider-receipt-binding:v1"
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
    if family in {"v3", "v4"}:
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
                    "model-provider-receipt-binding:v2"
                    if ingestion_custody
                    else "model-provider-receipt-binding:v1"
                ),
                "call_cap": 1,
                "credit_cap": 1.0,
                "timeout_seconds": 120.0,
                "completion_custody_fields": [
                    *(
                        ["provider_response_ingestion"]
                        if ingestion_custody
                        else []
                    ),
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
        if official_baseline:
            model_constants[
                "MODEL_RUNNER_COMPLETION_ACCOUNTING_SCHEMA_VERSION"
            ] = "model-runner-completion-accounting:v2"
            champion["completion_accounting_schema_version"] = (
                "model-runner-completion-accounting:v2"
            )
            def identity(
                schema_version: str, **values: Any
            ) -> dict[str, Any]:
                payload = {
                    "schema_version": schema_version,
                    "contract_fields": ["fixture"],
                    **values,
                }
                encoded = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
                return {
                    **payload,
                    "contract_sha256": hashlib.sha256(encoded).hexdigest(),
                }

            champion["normalization_action"].update({
                "dispatch_schema_version": (
                    "model-runner-normalization-dispatch:v1"
                ),
                "dispatch_entrypoint": (
                    "prepare_runner_normalization_request"
                ),
                "dispatch_contract": identity(
                    "model-runner-normalization-dispatch:v1"
                ),
            })
            fixture_catalog_bindings = [{
                "schema_version": "host-capability-binding:v1",
                "action_type": "verify_company",
                "tool_id": "verifier.company",
                "binding_contract_sha256": "1" * 64,
                "response_schema_version": "company-verifier-result:v1",
                "idempotency": "idempotent",
                "max_response_bytes": 100_000,
            }]
            fixture_binding_contracts_sha256 = hashlib.sha256(
                json.dumps(
                    fixture_catalog_bindings,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            fixture_catalog_payload = {
                "schema_version": (
                    "model-runner-official-host-binding-catalog:v1"
                ),
                "bindings": fixture_catalog_bindings,
                "binding_contracts_sha256": (
                    fixture_binding_contracts_sha256
                ),
            }
            fixture_catalog_sha256 = hashlib.sha256(
                json.dumps(
                    fixture_catalog_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()

            champion.update({
                "host_capability_manifest_entrypoint": (
                    "build_host_capability_manifest"
                ),
                "host_capability_manifest_schema_version": (
                    "host-capability-manifest:v1"
                ),
                "host_capability_manifest_contract": identity(
                    "host-capability-manifest-contract:v1"
                ),
                "benchmark_projection_entrypoint": (
                    "project_runner_result_for_benchmark"
                ),
                "benchmark_projection_schema_version": (
                    "model-runner-benchmark-projection:v1"
                ),
                "benchmark_projection_contract": identity(
                    "model-runner-benchmark-projection-contract:v1"
                ),
                "official_baseline_execution_entrypoint": (
                    "build_official_baseline_execution"
                ),
                "official_baseline_execution_schema_version": (
                    "leadpoet.research_lab.official_baseline_execution.v1"
                ),
                "official_baseline_execution_contract": identity(
                    "leadpoet.research_lab.official_baseline_execution-contract.v1"
                ),
                "provider_prepare_entrypoint": (
                    "prepare_runner_provider_request"
                ),
                "provider_prepare_schema_version": (
                    "model-runner-provider-dispatch:v1"
                ),
                "provider_prepare_contract": identity(
                    "model-runner-provider-dispatch-contract:v1"
                ),
                "provider_response_ingestion_entrypoint": (
                    "ingest_runner_provider_response"
                ),
                "provider_response_ingestion_schema_version": (
                    "model-runner-provider-response-ingestion:v1"
                ),
                "provider_response_ingestion_contract": {
                    **(
                        ingestion_payload := {
                            "schema_version": (
                                "model-runner-provider-response-ingestion:v1"
                            ),
                            "ingestion_entrypoint": (
                                "ingest_model_provider_response"
                            ),
                            "ingestion_signature": [
                                "action",
                                "host_response",
                            ],
                            "host_response_schema_version": (
                                "host-provider-response:v1"
                            ),
                            "host_response_closed_fields": [
                                "schema_version",
                                "provider",
                                "status_code",
                                "body",
                            ],
                            "host_response_body_authority": (
                                "action_bound_provider_compiler_parser"
                            ),
                            "dispatch_schema_version": (
                                "model-runner-provider-dispatch:v1"
                            ),
                            "dispatch_contract_sha256": identity(
                                "model-runner-provider-dispatch-contract:v1"
                            )["contract_sha256"],
                            "compiler_inventory_sha256": "f" * 64,
                            "parsed_response_schema_versions": [
                                "model-icp-normalization-provider-response:v1",
                                "model-provider-response:v3",
                            ],
                            "closed_fields": [
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
                            ],
                            "raw_host_response_returned": False,
                            **(
                                {
                                    "ingestion_receipt_required_for_response": (
                                        True
                                    )
                                }
                                if ingestion_custody
                                else {}
                            ),
                            "completion_input": (
                                "original_unchanged_host_response+"
                                "exact_model_provider_response_ingestion"
                                if ingestion_custody
                                else "original_unchanged_host_response"
                            ),
                            "provider_receipt_binding_input": (
                                "original_unchanged_host_response+"
                                "exact_model_provider_response_ingestion"
                                if ingestion_custody
                                else "original_unchanged_host_response"
                            ),
                            "completion_reparses_response": True,
                            "durable_custody": (
                                "host_provider_action_receipt_before_completion"
                            ),
                            "custody_receipt_field": (
                                "model_provider_response_ingestion"
                            ),
                            "replay_requirement": (
                                "reload_raw_response_reingest_and_require_byte_identical_receipt"
                            ),
                            "custody_join": [
                                "action_sha256",
                                *(
                                    [
                                        "dispatch_sha256",
                                        "request_sha256",
                                    ]
                                    if ingestion_custody
                                    else []
                                ),
                                "host_response_sha256",
                                "parsed_response_sha256",
                                *(
                                    ["ingestion_sha256"]
                                    if ingestion_custody
                                    else []
                                ),
                            ],
                            "host_semantic_projection_allowed": False,
                            "hash_algorithm": "sha256",
                            "canonical_json": (
                                "utf8-json-sort-keys-compact-ascii-no-nan"
                            ),
                        }
                    ),
                    "contract_sha256": _canonical_sha256(
                        ingestion_payload
                    ),
                },
                "provider_compiler_inventory_entrypoint": (
                    "model_runner_provider_compiler_inventory"
                ),
                "provider_compiler_inventory_schema_version": (
                    "model-runner-provider-compiler-inventory:v1"
                ),
                "provider_compiler_preflight_entrypoint": (
                    "runner_provider_compiler_preflight"
                ),
                "provider_compiler_preflight_schema_version": (
                    "model-runner-provider-compiler-preflight:v1"
                ),
                "verifier_execution_entrypoint": (
                    "execute_runner_verifier_action"
                ),
                "verifier_execution_schema_version": (
                    "model-runner-verifier-execution:v1"
                ),
                "verifier_execution_contract": identity(
                    "model-runner-verifier-execution-contract:v1"
                ),
                "official_host_binding_catalog_entrypoint": (
                    "runner_official_host_binding_catalog"
                ),
                "official_host_binding_catalog_schema_version": (
                    "model-runner-official-host-binding-catalog:v1"
                ),
                "official_host_binding_catalog_contract": identity(
                    "model-runner-official-host-binding-catalog-contract:v1",
                    binding_contracts_sha256=(
                        fixture_binding_contracts_sha256
                    ),
                    catalog_sha256=fixture_catalog_sha256,
                ),
                "official_host_capability_manifest_entrypoint": (
                    "build_runner_official_host_capability_manifest"
                ),
                "provider_response_schema_version": (
                    "model-provider-response:v3"
                ),
                "candidate_provider_record_schema_version": (
                    "model-candidate-provider-record:v1"
                ),
                "candidate_provider_projection_schema_version": (
                    "model-candidate-provider-projection:v1"
                ),
                "candidate_provider_projection_contract": identity(
                    "model-candidate-provider-projection:v1"
                ),
                "company_verifier_result_schema_version": (
                    "company-verifier-response:v2"
                ),
                "company_verifier_result_contract": identity(
                    "company-verifier-response:v2"
                ),
                "company_fit_source_evidence_schema_version": (
                    "company-fit-source-evidence:v1"
                ),
                "company_fit_source_evidence_contract": identity(
                    "company-fit-source-evidence:v1"
                ),
                "company_verifier_evidence_schema_version": (
                    "company-verifier-evidence:v2"
                ),
                "model_verified_lead_evidence_schema_version": (
                    "model-verified-lead-evidence:v2"
                ),
                "company_fit_proof_contract_sha256": "d" * 64,
            })
            required_names.update({
                "build_host_capability_manifest",
                "project_runner_result_for_benchmark",
                "build_official_baseline_execution",
                "prepare_runner_provider_request",
                "ingest_runner_provider_response",
                "prepare_runner_normalization_request",
                "model_runner_provider_compiler_inventory",
                "runner_provider_compiler_preflight",
                "execute_runner_verifier_action",
                "runner_official_host_binding_catalog",
                "build_runner_official_host_capability_manifest",
            })
            champion["runner_role_contract"] = _role_contract(
                contract_id,
                ingestion_custody=ingestion_custody,
            )
    functions = {
        name: list(_MEMBERS[name][0]) for name in sorted(required_names)
    }
    full_parameters = {
        name: list(_MEMBERS[name][1]) for name in sorted(required_names)
    }
    keyword_only = {
        name: list(_MEMBERS[name][2])
        for name in sorted(required_names)
        if _MEMBERS[name][2]
    }
    consumer_contract: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": contract_id,
        "functions": functions,
        "exact_signatures": [
            "research_lab_adapter.py:" + name
            for name in sorted(required_names)
        ],
        "full_parameters": full_parameters,
        "required_keyword_only": keyword_only,
        "exact_constants": exact_constants,
    }
    if official_baseline:
        consumer_contract["extensions"] = {}
        consumer_contract["frozen_asyncness"] = {
            name: False for name in sorted(required_names)
        }
    declaration = {
        "schema_version": (
            "leadpoet.research_lab.artifact_runner_declaration.v1"
        ),
        "champion_execution": champion,
        "consumer_contract_sha256": contract_hash,
        "consumer_contract": consumer_contract,
    }
    return deepcopy(declaration)


def runner_versions(family: str = "v3") -> dict[str, str]:
    return dict(
        _V4_VERSIONS
        if family == "v4"
        else _V3_VERSIONS
        if family == "v3"
        else _V2_VERSIONS
    )
