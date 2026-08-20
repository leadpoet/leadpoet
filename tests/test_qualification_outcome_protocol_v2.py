from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from types import ModuleType

import pytest
import research_lab.sourcing_model_contract_check as contract_check
import gateway.tee.model_sandbox_v2 as model_sandbox_module

from gateway.research_lab import scoring_worker as sw
from gateway.research_lab.attested_scoring_v2 import AttestedScoringV2Error
from gateway.research_lab.model_authority_v2 import (
    AttestedPrivateModelRunnerV2,
    AttestedPrivateModelRunnerV2Error,
    QualificationOutcomeCompleteV2,
    QualificationOutcomeIncompleteV2Error,
    _host_provider_observation_v1,
    _model_qualification_authority_v1,
)
from gateway.tee.model_sandbox_v2 import (
    MODEL_SANDBOX_VISIBLE_ROOT,
    ROOTFS_MANIFEST_NAME,
    RunscModelSandboxV2,
    RunscSandboxConfigV2,
    ModelSandboxV2Error,
    _MEASURED_METADATA_BOOTSTRAP,
    _build_consumer_runtime_probe_from_observation_v1,
    _consumer_runtime_probe_v1,
    _local_provider_replay_resolver_v2,
    _model_adapter_bootstrap_for_compatibility_receipt_v1,
    _runtime_probe_observation_plan_v1,
    _runtime_invariant_policy_v1,
    _runtime_probe_expected_invariants,
    _validate_qualification_terminal_observation_v1,
    validate_consumer_runtime_probe_v1,
)
from gateway.tee.provider_client_v2 import (
    BrokeredProviderTransportV2,
    _ExecutionScope,
    ProviderClientV2Error,
)
from gateway.tee.sandbox_provider_socket_v2 import SandboxProviderSocketServerV2
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
)
from research_lab.eval.private_runtime import (
    QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2,
    QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2,
    QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2,
    PrivateModelRuntimeError,
    _docker_adapter_bootstrap_for_qualify_compatibility,
    _docker_adapter_bootstrap_for_qualification_protocol_v2,
    begin_attested_receipt_hash_collection,
    canonicalize_private_model_icp,
    end_attested_receipt_hash_collection,
    publish_attested_receipt_hash,
    qualification_outcome_contract_v2,
    validate_qualification_outcome_envelope_v2,
    validate_qualification_outcome_protocol_metadata_v2,
    validate_qualification_outcome_protocol_probe_cases_v1,
    validate_sourcing_adapter_metadata,
)
from research_lab.eval.provider_evidence_cache import (
    canonical_request_fingerprint,
)
from research_lab.eval.snapshot_store import (
    MODE_RECORD,
    ProviderSnapshotStore,
    build_snapshot_request,
)
from research_lab.eval import DockerPrivateModelSpec, PrivateModelArtifactManifest
from research_lab.sourcing_model_contract_check import (
    _qualification_protocol_adapter_surface_v2,
    _qualification_protocol_entrypoint_declared_v2,
    compute_compatibility_source_tree_hash_v1,
    semantic_compatibility_policy_identity_v1,
    source_tree_compatibility_admission,
)


HASH_A = "sha256:" + "a" * 64


def _plain_hash(value) -> str:
    return sha256_json(value).removeprefix("sha256:")


def _metadata(*, major: int = 2, capabilities=None, extra_cases=()):
    return {
        "protocol_id": "sourcing-model.qualification-outcome",
        "major": major,
        "minor": 7,
        "entrypoint": "run_icp_outcome",
        "result_schema_version": "sourcing-model.qualification-outcome.v2",
        "route_completion_receipt_schema_version": (
            "sourcing-model.route-completion-receipt.v1"
        ),
        "contract_sha256": QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2,
        "capabilities": sorted(
            capabilities
            or {
                "complete_confirmed_empty",
                "consumer_nonce_probe",
                "incomplete_retryable",
                "model_owned_route_completion_receipt",
                "partial_companies_on_incomplete",
            }
        ),
        "probe": {
            "schema_version": "sourcing-model.qualification-outcome-probe.v1",
            "mode": "consumer_qualification_protocol_probe",
            "case_ids": sorted(
                {
                    "complete_confirmed_empty",
                    "incomplete_retryable",
                    *extra_cases,
                }
            ),
        },
        "extensions": {},
    }


def _ready_v2_adapter_metadata() -> dict:
    routing_catalog = {"schema_version": 1}
    routing_policy = {"schema_version": 1}
    runtime_catalog = {
        "schema_version": 1,
        "tools": [
            {"tool_id": tool_id}
            for tool_id in (
                "candidate.backlog",
                "candidate.registry_feed",
                "candidate.jobs_feed",
                "candidate.deepline_firmographic",
                "candidate.model_semantic",
                "intent.existing_evidence",
                "intent.jobs_feed",
                "intent.company_search",
                "intent.first_party",
                "intent.newsroom",
            )
        ],
    }
    runtime_policy = {"schema_version": 1}
    return {
        "adapter_version": "sourcing-model-research-lab-adapter:v-next",
        "scoring_adapter_version": "qualification-company-scorer:v1",
        "component_registry_version": "sourcing-model-components:v2",
        "capability_contract_version": "sourcing-model-runtime-capabilities:v3",
        "runtime_capabilities": [
            "deadline",
            "emit",
            "http_fetch",
            "probe_origin",
            "resolve_host",
        ],
        "resilience_policy_version": "sourcing-model-resilience:v2",
        "firmographic_discovery": {
            "firmographic_policy_version": (
                "sourcing-model-firmographic-discovery:v2"
            )
        },
        "industry_taxonomy": {
            "taxonomy_content_hash": "sha256:" + "d" * 64
        },
        "routing": {
            "compiler_version": "routing-compiler-v-next",
            "catalog": routing_catalog,
            "catalog_sha256": _plain_hash(routing_catalog),
            "policy": routing_policy,
            "policy_sha256": _plain_hash(routing_policy),
            "intent_sources": ["company_site", "job_listing", "news"],
            "source_add_requires_manifest_sha256": True,
            "private_bindings_exposed": False,
        },
        "runtime_routing": {
            "compiler_version": "routing-compiler-v-next",
            "catalog": runtime_catalog,
            "catalog_sha256": _plain_hash(runtime_catalog),
            "policy": runtime_policy,
            "policy_sha256": _plain_hash(runtime_policy),
            "candidate_tool_lanes": {
                "candidate.backlog": "backlog",
                "candidate.registry_feed": "registry_signal",
                "candidate.jobs_feed": "jobs_signal",
                "candidate.deepline_firmographic": "deepline_firmographic",
                "candidate.model_semantic": "model_semantic",
            },
            "intent_tool_tiers": {
                "intent.existing_evidence": "fused",
                "intent.jobs_feed": "jobs_feed",
                "intent.company_search": "company_search",
                "intent.first_party": "first_party",
                "intent.newsroom": "newsroom",
            },
            "private_bindings_exposed": False,
        },
        "component_registry": {
            "source_router": {
                "strategy_options": ["company_site", "job_listing", "news"]
            }
        },
        "qualification_outcome_protocol": _metadata(),
    }


def _required_route_outcomes(commitments, state: str) -> list[dict[str, str]]:
    return [
        {"commitment": commitment, "state": state}
        for commitment in sorted(commitments)
    ]


def _envelope(case_id: str, nonce: str) -> dict:
    complete = case_id == "complete_confirmed_empty"
    summary = {
        "attempted": 1,
        "completed": 0,
        "confirmed_empty": 1 if complete else 0,
        "retryable_failed": 0 if complete else 1,
        "terminal_failed": 0,
        "skipped": 0,
        "retried": 0,
    }
    receipt_body = {
        "schema_version": "sourcing-model.route-completion-receipt.v1",
        "contract_sha256": QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2,
        "outcome_authority": "sourcing_model",
        "completion_state": "complete" if complete else "incomplete",
        "disposition": case_id,
        "retryable": not complete,
        "partial": False,
        "returned_count": 0,
        "invocation_sha256": "1" * 64,
        "route_summary": summary,
        "failure_classes": [] if complete else ["retryable_provider"],
        "probe": {
            "schema_version": "sourcing-model.qualification-outcome-probe.v1",
            "case_id": case_id,
            "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        },
        "extensions": {},
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": _plain_hash(receipt_body),
    }
    return {
        "schema_version": "sourcing-model.qualification-outcome.v2",
        "protocol_major": 2,
        "protocol_minor": 4,
        "contract_sha256": QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2,
        "completion_state": receipt["completion_state"],
        "companies": [],
        "route_completion_receipt": receipt,
        "extensions": {},
    }


def _rehash_receipt(envelope: dict) -> dict:
    receipt = envelope["route_completion_receipt"]
    receipt["receipt_sha256"] = _plain_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )
    return envelope


def _write_protocol_tree(
    root: Path,
    *,
    harmless_revision: str,
    outcome_signature: str = "icp, context=None",
) -> tuple[str, dict]:
    root.mkdir()
    (root / "research_lab_adapter.py").write_text(
        "def adapter_metadata():\n"
        "    return {}\n\n"
        f"def run_icp_outcome({outcome_signature}):\n"
        "    return {}\n\n"
        f"HARMLESS_REVISION = {harmless_revision!r}\n",
        encoding="utf-8",
    )
    (root / "sourcing_model").mkdir()
    (root / "sourcing_model" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (root / "sourcing_model" / "qualification_route.py").write_text(
        "def transport_headers():\n"
        "    return {}\n",
        encoding="utf-8",
    )
    contract_path = root / "consumer-contract.json"
    parity_path = root / "consumer-parity.json"
    contract_path.write_text(
        json.dumps({"contract_id": "model-contract:v-next"}),
        encoding="utf-8",
    )
    parity_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    source_hash = compute_compatibility_source_tree_hash_v1(root)
    manifest = {
        "model_artifact_hash": source_hash,
        "git_commit_sha": hashlib.sha1(harmless_revision.encode()).hexdigest(),
        "manifest_hash": "sha256:" + hashlib.sha256(
            ("manifest:" + harmless_revision).encode()
        ).hexdigest(),
        "image_digest": "example.invalid/model@sha256:" + hashlib.sha256(
            ("image:" + harmless_revision).encode()
        ).hexdigest(),
        "compatibility_contract": {
            "contract_id": "model-contract:v-next",
            "path": contract_path.name,
            "sha256": sha256_bytes(contract_path.read_bytes()),
        },
        "consumer_parity_fixtures": {
            "path": parity_path.name,
            "sha256": sha256_bytes(parity_path.read_bytes()),
        },
    }
    return source_hash, manifest


def _attempt(
    *,
    ordinal: int,
    terminal: str,
    http_status=None,
    logical_operation_id: str = "model-route-a",
) -> dict:
    accepted = terminal in {
        "authenticated_response",
        "attested_local_response",
    }
    external = terminal == "authenticated_response"
    return build_transport_attempt(
        request_id=("%032x" % (ordinal + 1)),
        logical_operation_id=logical_operation_id,
        job_id="model-job-a",
        purpose="research_lab.private_model_run.v2",
        provider_id="public_web",
        attempt_number=ordinal,
        method="POST",
        destination_host="provider.example.com",
        destination_port=443,
        path_hash="sha256:" + "2" * 64,
        nonsecret_headers_hash="sha256:" + "3" * 64,
        body_hash="sha256:" + "4" * 64,
        credential_ref_hash="sha256:" + "5" * 64,
        retry_policy_hash="sha256:" + "6" * 64,
        timeout_ms=1000,
        started_at=f"2026-08-20T00:00:0{ordinal}Z",
        terminal_status=terminal,
        http_status=http_status,
        response_hash="sha256:" + "7" * 64 if accepted else None,
        request_artifact_hash="sha256:" + "8" * 64,
        response_artifact_hash="sha256:" + "9" * 64 if accepted else None,
        tls_peer_chain_hash="sha256:" + "a" * 64 if external else None,
        tls_protocol="tls1_3" if external else None,
        failure_code=None if accepted else "connection_reset",
        completed_at=f"2026-08-20T00:00:1{ordinal}Z",
    )


def _provider_scope() -> _ExecutionScope:
    return _ExecutionScope(
        job_id="model-job-a",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="model-call-a",
        retry_policy_hashes={},
        default_timeout_ms=1000,
        terminal_sink=None,
    )


@pytest.mark.asyncio
async def test_failed_execution_authority_root_reaches_attempt_collector() -> None:
    authority = {
        "status": "failed",
        "receipt": {"receipt_hash": HASH_A, "status": "failed"},
        "receipt_graph": {"root_receipt_hash": HASH_A},
    }
    runner = object.__new__(AttestedPrivateModelRunnerV2)
    runner.spec = type("Spec", (), {"timeout_seconds": 30})()
    runner._shared_state = {
        "sequence": 0,
        "receipts": [],
        "authorities": [],
        "compatibility_admissions": [],
        "generated_caches": {},
        "evidence_summaries": {},
        "catalog_snapshot_futures": {},
        "lock": threading.Lock(),
    }

    async def fail(**_kwargs):
        raise AttestedScoringV2Error(
            "V2 scoring failed closed: execution_modelsandboxv2error",
            authority=authority,
        )

    runner._execute_operation = fail
    collected, token = begin_attested_receipt_hash_collection()
    try:
        with pytest.raises(AttestedPrivateModelRunnerV2Error) as captured:
            await runner._invoke_operation()
    finally:
        end_attested_receipt_hash_collection(token)

    assert captured.value.authority == authority
    assert collected == {HASH_A}
    assert runner.attested_receipts() == [authority["receipt"]]
    assert runner.attested_authorities() == [authority]


def test_cross_repository_semantic_contract_fixture_is_exact() -> None:
    document = qualification_outcome_contract_v2()

    assert _plain_hash(document) == QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2
    assert document["entrypoint"] == "run_icp_outcome"
    assert document["completion_rules"]["unreceipted_empty_allowed"] is False
    assert document["production_complete_authority"] == (
        "model_semantics_joined_to_every_state_compatible_"
        "required_route_latest_terminal"
    )
    assert document["production_confirmed_empty_authority"] == (
        "model_semantics_joined_to_every_state_compatible_"
        "required_route_latest_terminal"
    )


def test_same_major_metadata_allows_harmless_additive_capability() -> None:
    document = _metadata(
        capabilities={
            "complete_confirmed_empty",
            "consumer_nonce_probe",
            "incomplete_retryable",
            "model_owned_route_completion_receipt",
            "partial_companies_on_incomplete",
            "vendor.optional_route_diagnostic",
        },
        extra_cases={"vendor.optional_case"},
    )
    document["extensions"] = {
        "com.example.optional": {"enabled": True},
    }

    assert validate_qualification_outcome_protocol_metadata_v2(document) == document


def test_extension_limits_are_contract_derived_and_route_specific() -> None:
    commitments = [
        f"{index:064x}"
        for index in range(QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2)
    ]
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    receipt = envelope["route_completion_receipt"]
    receipt["route_summary"]["attempted"] = len(commitments)
    receipt["route_summary"]["confirmed_empty"] = len(commitments)
    receipt["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes(commitments, "confirmed_empty")
        )
    }

    assert validate_qualification_outcome_envelope_v2(
        _rehash_receipt(envelope)
    )["route_completion_receipt"]["extensions"] == receipt["extensions"]

    oversized = deepcopy(envelope)
    oversized_commitments = [
        f"{index:064x}"
        for index in range(
            QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2 + 1
        )
    ]
    oversized_receipt = oversized["route_completion_receipt"]
    oversized_receipt["route_summary"]["attempted"] = len(
        oversized_commitments
    )
    oversized_receipt["route_summary"]["confirmed_empty"] = len(
        oversized_commitments
    )
    oversized_receipt["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes(
                oversized_commitments,
                "confirmed_empty",
            )
        )
    }
    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_envelope_v2(
            _rehash_receipt(oversized)
        )

    unrelated = _metadata()
    unrelated["extensions"] = {
        "com.example.optional": list(range(256)),
    }
    assert validate_qualification_outcome_protocol_metadata_v2(unrelated)
    unrelated["extensions"]["com.example.optional"].append(256)
    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_protocol_metadata_v2(unrelated)


def test_route_outcome_budget_covers_configured_all_flow_with_headroom() -> None:
    # Current bounded ALL flow: four branches, goal 50, candidate adjudication,
    # acquisition, and bounded liveness produce at most 3,742 obligations.
    configured_all_flow_upper_bound = 3_742
    assert QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2 >= (
        configured_all_flow_upper_bound * 2
    )


def test_generic_extension_nested_and_exact_byte_bounds_match_contract() -> None:
    def accepted(value, *, key="com.example.optional") -> bool:
        metadata = _metadata()
        metadata["extensions"] = {key: value}
        try:
            validate_qualification_outcome_protocol_metadata_v2(metadata)
        except PrivateModelRuntimeError:
            return False
        return True

    assert accepted({f"field{index}": index for index in range(32)})
    assert not accepted({f"field{index}": index for index in range(33)})
    assert accepted("a" * 128)
    assert not accepted("a" * 129)
    assert accepted({"a" * 64: True})
    assert not accepted({"a" * 65: True})
    assert accepted(True, key="a" * 32 + ".example")
    assert not accepted(True, key="a" * 33 + ".example")
    assert accepted([[[0]]])
    assert not accepted([[[[0]]]])

    def extension_with_canonical_size(target: int):
        key = "com.example.boundary"
        def build(count: int, final_size: int = 0):
            values = ["a" * 128 for _ in range(max(0, count - 2))]
            if count >= 2:
                values.extend(
                    [
                        "b" * min(final_size, 128),
                        "c" * max(0, final_size - 128),
                    ]
                )
            elif count:
                values.append("b" * final_size)
            fields = {
                f"field{index:02d}": values[index * 256 : (index + 1) * 256]
                for index in range(32)
            }
            return {key: fields}

        low, high = 1, 32 * 256
        while low <= high:
            count = (low + high) // 2
            base = build(count)
            base_size = len(canonical_json(base).encode("utf-8"))
            remainder = target - base_size
            if 0 <= remainder <= (256 if count >= 2 else 128):
                document = build(count, remainder)
                assert len(canonical_json(document).encode("utf-8")) == target
                return document
            if remainder < 0:
                high = count - 1
            else:
                low = count + 1
        raise AssertionError("extension byte-boundary fixture is unavailable")

    maximum_bytes = qualification_outcome_contract_v2()[
        "extension_evolution"
    ]["maximum_canonical_bytes"]
    exact = _metadata()
    exact["extensions"] = extension_with_canonical_size(maximum_bytes)
    assert validate_qualification_outcome_protocol_metadata_v2(exact)
    oversized = _metadata()
    oversized["extensions"] = extension_with_canonical_size(maximum_bytes + 1)
    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_protocol_metadata_v2(oversized)


def test_required_route_extension_is_receipt_only() -> None:
    commitment = "1" * 64
    metadata = _metadata()
    metadata["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes([commitment], "confirmed_empty")
        )
    }
    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_protocol_metadata_v2(metadata)

    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    envelope["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes([commitment], "confirmed_empty")
        )
    }
    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_envelope_v2(envelope)


@pytest.mark.parametrize("mutation", ["missing_capability", "unknown_major"])
def test_missing_capability_or_unknown_major_rejects_preactivation(mutation) -> None:
    document = _metadata()
    if mutation == "missing_capability":
        document["capabilities"].remove("incomplete_retryable")
    else:
        document["major"] = 3

    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_protocol_metadata_v2(document)


def test_two_exact_artifacts_select_same_versioned_profile_without_hash_allowlist(
    tmp_path,
) -> None:
    first_hash, first_manifest = _write_protocol_tree(
        tmp_path / "first",
        harmless_revision="first",
    )
    second_hash, second_manifest = _write_protocol_tree(
        tmp_path / "second",
        harmless_revision="second",
        outcome_signature="target, options=None, trace=None, **kwargs",
    )

    first = source_tree_compatibility_admission(
        tmp_path / "first",
        manifest=first_manifest,
        source_tree_hash=first_hash,
    )
    second = source_tree_compatibility_admission(
        tmp_path / "second",
        manifest=second_manifest,
        source_tree_hash=second_hash,
    )

    assert first_hash != second_hash
    assert first["source_tree_hash"] != second["source_tree_hash"]
    for field in (
        "schema_version",
        "consumer_api_version",
        "decision",
        "admission_mode",
        "policy_hash",
        "entrypoints",
    ):
        assert first[field] == second[field]
    assert first["admission_mode"] == "qualification_protocol_v2"
    first_policy = _runtime_invariant_policy_v1(first)
    assert first_policy == {"profile": "qualification_protocol_v2"}
    assert _runtime_probe_expected_invariants(first_policy) == {
        "profile": "qualification_protocol_v2"
    }
    first_bootstrap = _docker_adapter_bootstrap_for_qualification_protocol_v2()
    second_bootstrap = _docker_adapter_bootstrap_for_qualification_protocol_v2()
    assert first_bootstrap == second_bootstrap
    assert "_research_lab_patch_affected_runtime_capability_scope(module)" not in (
        first_bootstrap
    )
    assert "getattr(module, 'run_icp_outcome')" in first_bootstrap


def test_v2_native_capability_scope_does_not_remove_e55_transition_patch() -> None:
    legacy_bootstrap = _docker_adapter_bootstrap_for_qualify_compatibility(
        preserve_native_qualify=True,
    )
    v2_bootstrap = _docker_adapter_bootstrap_for_qualification_protocol_v2()

    assert "_research_lab_patch_affected_runtime_capability_scope(module)" in (
        legacy_bootstrap
    )
    assert "_research_lab_patch_affected_runtime_capability_scope(module)" not in (
        v2_bootstrap
    )


def test_declared_outcome_entrypoint_missing_second_argument_cannot_fallback(
    tmp_path,
) -> None:
    source_hash, manifest = _write_protocol_tree(
        tmp_path / "invalid",
        harmless_revision="invalid",
        outcome_signature="icp",
    )

    admission = source_tree_compatibility_admission(
        tmp_path / "invalid",
        manifest=manifest,
        source_tree_hash=source_hash,
    )

    assert admission["admission_mode"] == "qualification_protocol_v2"
    assert not _qualification_protocol_adapter_surface_v2(tmp_path / "invalid")
    probe = _consumer_runtime_probe_v1(
        compatibility_receipt=admission,
        metadata={},
        expected_module_name="research_lab_adapter",
        expected_callable_name="adapter_metadata",
        invariants={"profile": "qualification_protocol_v2"},
    )
    with pytest.raises(ModelSandboxV2Error, match="source admission"):
        validate_consumer_runtime_probe_v1(
            probe,
            compatibility_receipt=admission,
            metadata={},
            expected_source_tree_hash=source_hash,
            expected_manifest_hash=manifest["manifest_hash"],
            expected_image_digest=manifest["image_digest"],
            expected_module_name="research_lab_adapter",
            expected_callable_name="adapter_metadata",
        )


@pytest.mark.parametrize(
    "declaration",
    [
        "run_icp_outcome = dynamically_resolved_entrypoint\n",
        "from model_entrypoints import execute as run_icp_outcome\n",
        (
            "def adapter_metadata():\n"
            "    return {'qualification_outcome_protocol': {}}\n"
        ),
    ],
)
def test_dynamic_or_metadata_v2_declaration_cannot_fallback(
    tmp_path,
    declaration,
) -> None:
    root = tmp_path / "invalid-declaration"
    _source_hash, manifest = _write_protocol_tree(
        root,
        harmless_revision="dynamic-declaration",
    )
    (root / "research_lab_adapter.py").write_text(
        declaration,
        encoding="utf-8",
    )
    source_hash = compute_compatibility_source_tree_hash_v1(root)
    manifest["model_artifact_hash"] = source_hash

    admission = source_tree_compatibility_admission(
        root,
        manifest=manifest,
        source_tree_hash=source_hash,
    )

    assert admission["admission_mode"] == "qualification_protocol_v2"


@pytest.mark.asyncio
async def test_hermetic_v2_incomplete_to_complete_transition(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path / "valid-alias"
    _source_hash, manifest = _write_protocol_tree(
        root,
        harmless_revision="valid-alias",
    )
    (root / "research_lab_adapter.py").write_text(
        "import hashlib\n"
        "import json\n\n"
        f"_METADATA = {repr(_metadata())}\n\n"
        "def adapter_metadata():\n"
        "    return _METADATA\n\n"
        "def _canonical_hash(value):\n"
        "    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')\n"
        "    return hashlib.sha256(payload).hexdigest()\n\n"
        "def _outcome(target, options=None, trace=None, **kwargs):\n"
        "    probe = options['probe']\n"
        "    case_id = probe['case_id']\n"
        "    complete = case_id == 'complete_confirmed_empty'\n"
        "    receipt = {\n"
        "        'schema_version': 'sourcing-model.route-completion-receipt.v1',\n"
        f"        'contract_sha256': '{QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2}',\n"
        "        'outcome_authority': 'sourcing_model',\n"
        "        'completion_state': 'complete' if complete else 'incomplete',\n"
        "        'disposition': case_id,\n"
        "        'retryable': not complete,\n"
        "        'partial': False,\n"
        "        'returned_count': 0,\n"
        "        'invocation_sha256': '1' * 64,\n"
        "        'route_summary': {\n"
        "            'attempted': 1, 'completed': 0,\n"
        "            'confirmed_empty': 1 if complete else 0,\n"
        "            'retryable_failed': 0 if complete else 1,\n"
        "            'terminal_failed': 0, 'skipped': 0, 'retried': 0,\n"
        "        },\n"
        "        'failure_classes': [] if complete else ['retryable_provider'],\n"
        "        'probe': {\n"
        "            'schema_version': 'sourcing-model.qualification-outcome-probe.v1',\n"
        "            'case_id': case_id,\n"
        "            'nonce_sha256': hashlib.sha256(probe['nonce'].encode('ascii')).hexdigest(),\n"
        "        },\n"
        "        'extensions': {},\n"
        "    }\n"
        "    receipt['receipt_sha256'] = _canonical_hash(receipt)\n"
        "    return {\n"
        "        'schema_version': 'sourcing-model.qualification-outcome.v2',\n"
        "        'protocol_major': 2, 'protocol_minor': 7,\n"
        f"        'contract_sha256': '{QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2}',\n"
        "        'completion_state': receipt['completion_state'],\n"
        "        'companies': [], 'route_completion_receipt': receipt,\n"
        "        'extensions': {},\n"
        "    }\n\n"
        "run_icp_outcome = _outcome\n",
        encoding="utf-8",
    )
    source_hash = compute_compatibility_source_tree_hash_v1(root)
    manifest["model_artifact_hash"] = source_hash
    admission = source_tree_compatibility_admission(
        root,
        manifest=manifest,
        source_tree_hash=source_hash,
    )
    module = runpy.run_path(str(root / "research_lab_adapter.py"))
    cases = {}
    nonce_hashes = {}
    for case_id, nonce in (
        ("complete_confirmed_empty", "alias-complete-nonce-001"),
        ("incomplete_retryable", "alias-incomplete-nonce-1"),
    ):
        cases[case_id] = module["run_icp_outcome"](
            {},
            {
                "mode": "consumer_qualification_protocol_probe",
                "probe": {
                    "schema_version": (
                        "sourcing-model.qualification-outcome-probe.v1"
                    ),
                    "case_id": case_id,
                    "nonce": nonce,
                },
            },
        )
        nonce_hashes[case_id] = hashlib.sha256(
            nonce.encode("ascii")
        ).hexdigest()

    validate_qualification_outcome_protocol_probe_cases_v1(
        cases,
        expected_nonce_sha256s=nonce_hashes,
    )
    assert admission["admission_mode"] == "qualification_protocol_v2"
    assert not _qualification_protocol_adapter_surface_v2(root)

    artifact = PrivateModelArtifactManifest(
        model_artifact_hash=source_hash,
        git_commit_sha=manifest["git_commit_sha"],
        image_digest=manifest["image_digest"],
        config_hash="sha256:" + "6" * 64,
        component_registry_version="sourcing-model-components:v2",
        scoring_adapter_version="qualification-company-scorer:v1",
        manifest_uri="s3://private-model/qualification-v2.json",
        manifest_hash=manifest["manifest_hash"],
        signature_ref="kms:qualification-v2-test",
        compatibility_contract=manifest["compatibility_contract"],
        consumer_parity_fixtures=manifest["consumer_parity_fixtures"],
    )
    input_doc = {
        "icp": {"industry": "software", "max_companies": 3},
        "context": {"mode": "private_baseline"},
    }

    def runtime_envelope(
        disposition: str,
        *,
        commitments=(),
    ) -> dict:
        envelope = _envelope(disposition, "runtime-transition-nonce-01")
        receipt = envelope["route_completion_receipt"]
        receipt["probe"] = None
        receipt["invocation_sha256"] = _plain_hash(input_doc)
        if commitments:
            state = (
                "confirmed_empty"
                if disposition == "complete_confirmed_empty"
                else "completed"
                if disposition == "complete_nonempty"
                else "retryable_failed"
                if disposition == "incomplete_retryable"
                else "terminal_failed"
            )
            receipt["extensions"] = {
                QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
                    _required_route_outcomes(commitments, state)
                )
            }
        receipt["receipt_sha256"] = _plain_hash(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )
        return validate_qualification_outcome_envelope_v2(envelope)

    def measured_authority(
        disposition: str,
        *,
        terminal_status: str,
        http_status,
        commitment: str = "",
    ) -> tuple[dict, str]:
        attempt = _attempt(
            ordinal=0,
            terminal=terminal_status,
            http_status=http_status,
            logical_operation_id=(
                "transition-required-route"
                if commitment
                else "transition-incomplete-route"
            ),
        )
        scope = _provider_scope()
        operation_id = str(attempt["logical_operation_id"])
        scope.record_intent(operation_id, 0, commitment)
        scope.record_terminal(
            operation_id,
            0,
            terminal_status,
            http_status,
            str(attempt["attempt_hash"]),
        )
        observation = scope.completion_observation()
        envelope = runtime_envelope(
            disposition,
            commitments=(commitment,) if commitment else (),
        )
        _validate_qualification_terminal_observation_v1(
            envelope,
            observation,
        )
        execution_root = sha256_json(
            {
                "schema_version": "test.signed-execution-root.v1",
                "disposition": disposition,
                "artifact": source_hash,
            }
        )
        sandbox_result = {
            "input_hash": sha256_json(input_doc),
            "provider_terminal_observation": observation,
            "provider_terminal_observation_hash": sha256_json(observation),
        }
        outcome = {
            "transport_attempts": [attempt],
            "execution_receipt": {"receipt_hash": execution_root},
            "execution_receipt_graph": {
                "root_receipt_hash": execution_root
            },
        }
        authority = _model_qualification_authority_v1(
            envelope=envelope,
            input_doc=input_doc,
            sandbox_result=sandbox_result,
            outcome=outcome,
            artifact=artifact,
        )
        return authority, execution_root

    incomplete_authority, incomplete_root = measured_authority(
        "incomplete_retryable",
        terminal_status="transport_failure",
        http_status=None,
        commitment="6" * 64,
    )
    complete_authority, complete_root = measured_authority(
        "complete_confirmed_empty",
        terminal_status="attested_local_response",
        http_status=404,
        commitment="7" * 64,
    )
    incomplete = QualificationOutcomeIncompleteV2Error(
        "qualification incomplete",
        model_qualification_authority=incomplete_authority,
        authority={
            "receipt": {"receipt_hash": incomplete_root},
            "receipt_graph": {"root_receipt_hash": incomplete_root},
        },
    )
    complete = QualificationOutcomeCompleteV2(
        [],
        model_qualification_authority=complete_authority,
    )

    class TransitionRunner:
        def __init__(self):
            self.calls = 0
            self.spec = DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                extra_env={
                    sw.PROVIDER_COST_EVALUATION_SCOPE_ENV: (
                        "sha256:" + "8" * 64
                    )
                },
            )

        def with_spec(self, spec):
            self.spec = spec
            return self

        async def __call__(self, _icp, _context):
            self.calls += 1
            if self.calls == 1:
                publish_attested_receipt_hash(incomplete_root)
                raise incomplete
            publish_attested_receipt_hash(complete_root)
            return complete

    class NeverScore:
        def __init__(self):
            self.calls = 0

        async def score_with_breakdowns(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError(
                "incomplete or confirmed-empty transition was scored"
            )

    async def no_repo_drift(**_kwargs):
        return None

    async def no_maintenance_boundary(**_kwargs):
        return None

    monkeypatch.setattr(
        sw,
        "_enforce_baseline_wave_maintenance_boundary",
        no_maintenance_boundary,
    )
    monkeypatch.setattr(
        sw,
        "_record_private_baseline_stage",
        lambda **_kwargs: None,
    )
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "qualification-transition-worker"
    worker.config = SimpleNamespace(
        private_baseline_concurrency=1,
        private_baseline_provider_retry_rounds=1,
        private_baseline_retry_concurrency=1,
        scoring_worker_total_workers=1,
    )
    worker._ensure_private_baseline_repo_head_unchanged = no_repo_drift
    item = {
        "icp": input_doc["icp"],
        "icp_ref": "icp:qualification-transition",
        "icp_hash": "qualification-transition-hash",
        "set_id": 1,
        "day_index": 1,
        "day_rank": 1,
    }
    runner = TransitionRunner()
    scorer = NeverScore()
    persisted_attempts = []

    async def persist_attempt(row, *, retry_round):
        persisted_attempts.append(
            sw._baseline_attempt_ledger_entry(
                row,
                retry_round=retry_round,
                gateway_runtime_commit_sha="9" * 40,
            )
        )
        return True

    results, retry_stats = await worker._run_baseline_batch_inner(
        runner=runner,
        retry_runner=runner,
        scorer=scorer,
        window=SimpleNamespace(benchmark_items=[item]),
        run_start=time.time(),
        attempt_checkpoint=persist_attempt,
        provider_cost_base_scope="sha256:" + "8" * 64,
        benchmark_date="2026-08-20",
    )

    assert retry_stats == {"retried": 1, "recovered": 1, "unresolved": 0}
    assert runner.calls == 2
    assert scorer.calls == 0
    assert len(persisted_attempts) == 2
    first_checkpoint = persisted_attempts[0]["result_row"]
    final_checkpoint = persisted_attempts[1]["result_row"]
    assert first_checkpoint["_runtime_error"] == "attempt_failed"
    assert first_checkpoint["_retryable"] is True
    assert first_checkpoint[sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD] == [
        incomplete_root
    ]
    assert first_checkpoint[sw._MODEL_QUALIFICATION_AUTHORITY_FIELD] == (
        incomplete_authority
    )
    assert final_checkpoint["_runtime_error"] == ""
    assert final_checkpoint["_retryable"] is False
    assert final_checkpoint[sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD] == [
        complete_root
    ]
    assert results[0][sw._MODEL_QUALIFICATION_AUTHORITY_FIELD] == (
        complete_authority
    )
    assert "companies" not in json.dumps(
        persisted_attempts,
        sort_keys=True,
    )


@pytest.mark.parametrize(
    ("evidence_mode", "local_authority"),
    [
        ("live", ""),
        ("cache_live", "provider_evidence_cache"),
        ("frozen", "snapshot"),
    ],
)
def test_live_cache_and_frozen_routes_share_host_measured_commitment_authority(
    monkeypatch,
    request,
    evidence_mode,
    local_authority,
) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    short_root = Path(tempfile.mkdtemp(prefix="lp-replay-", dir="/tmp"))
    request.addfinalizer(lambda: shutil.rmtree(short_root, ignore_errors=True))
    socket_path = short_root / "provider.sock"
    url = "https://api.exa.ai/search"
    body = b"{}"
    response_body = b'{"ok":true}'
    commitment = "d" * 64
    cache = {}
    snapshot_root = None
    snapshot_manifest_hash = ""
    if evidence_mode == "cache_live":
        fingerprint = canonical_request_fingerprint("POST", url, body)
        cache = {
            "schema_version": "1.1",
            "entries": {
                fingerprint: {
                    "status": 200,
                    "body_b64": base64.b64encode(response_body).decode("ascii"),
                }
            },
        }
        cache_path = short_root / "provider-evidence.json"
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        monkeypatch.setenv(shim.EVIDENCE_CACHE_PATH_ENV, str(cache_path))
    elif evidence_mode == "frozen":
        snapshot_root = short_root / "snapshot"
        record_store = ProviderSnapshotStore(
            str(snapshot_root),
            mode=MODE_RECORD,
        )
        record_store.record_response(
            build_snapshot_request("POST", url, body=body),
            status=200,
            body_text=response_body.decode("utf-8"),
        )
        record_store.write_dev_icp_items(
            [{"icp_ref": "qualification-replay-test"}]
        )
        manifest = record_store.build_manifest(
            recorded_at="2026-08-20T00:00:00Z"
        )
        record_store.write_manifest(manifest)
        snapshot_manifest_hash = str(manifest["manifest_hash"])
        monkeypatch.setenv(shim.SNAPSHOT_DIR_ENV, str(snapshot_root))

    hook_calls = []
    package = ModuleType("sourcing_model")
    package.__path__ = []
    route_module = ModuleType("sourcing_model.qualification_route")

    def transport_headers():
        hook_calls.append("called")
        return {
            "X-Leadpoet-Qualification-Route-Commitment": commitment,
        }

    route_module.transport_headers = transport_headers
    package.qualification_route = route_module
    monkeypatch.setitem(sys.modules, "sourcing_model", package)
    monkeypatch.setitem(
        sys.modules,
        "sourcing_model.qualification_route",
        route_module,
    )
    monkeypatch.setenv(shim.QUALIFICATION_PROTOCOL_V2_ENV, "1")
    monkeypatch.setenv(shim.EVIDENCE_MODE_ENV, evidence_mode)
    monkeypatch.setenv(shim.SOCKET_ENV, str(socket_path))

    coordinator_requests = []

    def coordinator(provider_request):
        coordinator_requests.append(deepcopy(dict(provider_request)))
        attempt = build_transport_attempt(
            request_id="1" * 32,
            logical_operation_id=provider_request["logical_operation_id"],
            job_id=provider_request["job_id"],
            purpose=provider_request["purpose"],
            provider_id=provider_request["provider_id"],
            attempt_number=provider_request["attempt_number"],
            method=provider_request["method"],
            destination_host="api.exa.ai",
            destination_port=443,
            path_hash="sha256:" + "1" * 64,
            nonsecret_headers_hash="sha256:" + "2" * 64,
            body_hash="sha256:" + "3" * 64,
            credential_ref_hash="sha256:" + "4" * 64,
            retry_policy_hash=provider_request["retry_policy_hash"],
            timeout_ms=provider_request["timeout_ms"],
            started_at="2026-08-20T00:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=sha256_bytes(response_body),
            request_artifact_hash="sha256:" + "6" * 64,
            response_artifact_hash=sha256_bytes(response_body),
            tls_peer_chain_hash="sha256:" + "8" * 64,
            tls_protocol="tls1_3",
            failure_code=None,
            completed_at="2026-08-20T00:00:01Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(response_body).decode("ascii"),
            "failure_code": None,
            "encrypted_artifact_id": "sha256:" + "9" * 64,
            "encrypted_request_artifact_id": "sha256:" + "a" * 64,
            "evidence_artifact_hashes": [],
            "transport_attempt": attempt,
        }

    retained_attempts = []
    transport = BrokeredProviderTransportV2(coordinator)
    scope = transport.create_scope(
        job_id="qualification-replay-authority",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="qualification-replay-authority",
        retry_policy_hashes={"exa": "sha256:" + "b" * 64},
        terminal_sink=lambda attempt: retained_attempts.append(
            deepcopy(dict(attempt))
        ),
        artifact_sink=lambda _artifact: None,
    )
    resolver = _local_provider_replay_resolver_v2(
        evidence_mode=evidence_mode,
        evidence_cache=cache,
        evidence_cache_hash=sha256_json(cache),
        snapshot_root=snapshot_root,
        snapshot_manifest_hash=snapshot_manifest_hash,
    )
    server = SandboxProviderSocketServerV2(
        socket_path=socket_path,
        transport=transport,
        execution_scope=scope,
        local_replay_resolver=resolver,
    )
    server.start()
    try:
        terminal = shim.execute(
            method="POST",
            url=url,
            headers={"accept": "application/json"},
            body=body,
            timeout_ms=1000,
        )
    finally:
        server.close()
    scope.assert_accepted_result_is_complete()
    observation = scope.completion_observation()

    assert hook_calls == ["called"]
    assert terminal["http_status"] == 200
    assert base64.b64decode(terminal["body_b64"]) == response_body
    assert observation["required_route_commitments"] == [commitment]
    assert observation["successful_required_route_count"] == 1
    assert len(retained_attempts) == 1
    assert retained_attempts[0]["attempt_hash"] == (
        observation["required_route_terminals"][0]["attempt_hash"]
    )
    if local_authority:
        assert coordinator_requests == []
        assert retained_attempts[0]["terminal_status"] == (
            "attested_local_response"
        )
    else:
        assert len(coordinator_requests) == 1
        assert retained_attempts[0]["terminal_status"] == (
            "authenticated_response"
        )
    assert all(
        "x-leadpoet-qualification-route-commitment"
        not in {str(name).lower() for name in item["headers"]}
        for item in coordinator_requests
    )


def test_local_replay_hint_cannot_override_host_authority(tmp_path) -> None:
    url = "https://api.exa.ai/search"
    body = b"{}"
    fingerprint = canonical_request_fingerprint("POST", url, body)
    cache = {
        "schema_version": "1.1",
        "entries": {
            fingerprint: {
                "status": 200,
                "body_b64": base64.b64encode(b'{"host":true}').decode(
                    "ascii"
                ),
            }
        },
    }
    request_doc = {
        "method": "POST",
        "url": url,
        "headers": {},
        "body": body,
        "timeout_ms": 1000,
    }
    resolver = _local_provider_replay_resolver_v2(
        evidence_mode="cache_live",
        evidence_cache=cache,
        evidence_cache_hash=sha256_json(cache),
        snapshot_root=None,
        snapshot_manifest_hash="",
    )
    resolved = resolver(request_doc, "provider_evidence_cache")
    assert base64.b64decode(resolved["body_b64"]) == b'{"host":true}'
    with pytest.raises(ModelSandboxV2Error, match="no matching host authority"):
        resolver(request_doc, "snapshot")
    live_resolver = _local_provider_replay_resolver_v2(
        evidence_mode="live",
        evidence_cache=cache,
        evidence_cache_hash=sha256_json(cache),
        snapshot_root=None,
        snapshot_manifest_hash="",
    )
    with pytest.raises(ModelSandboxV2Error, match="no matching host authority"):
        live_resolver(request_doc, "provider_evidence_cache")
    with pytest.raises(ModelSandboxV2Error, match="cache authority differs"):
        _local_provider_replay_resolver_v2(
            evidence_mode="cache_live",
            evidence_cache=cache,
            evidence_cache_hash="sha256:" + "0" * 64,
            snapshot_root=None,
            snapshot_manifest_hash="",
        )
    tampered_cache = deepcopy(cache)
    tampered_cache["entries"][fingerprint]["body_b64"] = "not-base64"
    tampered_resolver = _local_provider_replay_resolver_v2(
        evidence_mode="frozen",
        evidence_cache=tampered_cache,
        evidence_cache_hash=sha256_json(tampered_cache),
        snapshot_root=None,
        snapshot_manifest_hash="",
    )
    with pytest.raises(ModelSandboxV2Error, match="cache body is invalid"):
        tampered_resolver(request_doc, "provider_evidence_cache")


@pytest.mark.asyncio
async def test_production_path_hermetic_v2_transition_emits_stage_ledger(
    monkeypatch,
    tmp_path,
    request,
) -> None:
    stage_ledger = {
        "schema_version": "leadpoet.qualification-transition-stage-ledger.v1",
        "stages": {},
    }

    def passed(stage: str) -> None:
        stage_ledger["stages"][stage] = "passed"

    short_base = Path(tempfile.mkdtemp(prefix="lpq-", dir="/tmp"))
    request.addfinalizer(lambda: shutil.rmtree(short_base, ignore_errors=True))
    rootfs = short_base / "rootfs"
    marker = rootfs / ROOTFS_MANIFEST_NAME
    marker.parent.mkdir(parents=True)
    marker.write_text('{"rootfs":"qualification-transition"}\n')
    visible = rootfs / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/") / "transition"
    visible.mkdir(parents=True)
    (rootfs / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/")).chmod(0o711)
    source_root = visible / "source"
    source_hash, manifest = _write_protocol_tree(
        source_root,
        harmless_revision="production-path-transition",
    )
    route_commitment = "7" * 64
    adapter_metadata = _ready_v2_adapter_metadata()
    (source_root / "research_lab_adapter.py").write_text(
        "import contextlib\n"
        "import hashlib\n"
        "import json\n"
        "import urllib.request\n\n"
        "from sourcing_model import runtime_capabilities\n\n"
        f"_METADATA = {adapter_metadata!r}\n"
        f"_ROUTE_COMMITMENT = {route_commitment!r}\n\n"
        "def _hash(value):\n"
        "    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')\n"
        "    return hashlib.sha256(payload).hexdigest()\n\n"
        "def adapter_metadata():\n"
        "    return _METADATA\n\n"
        "@contextlib.contextmanager\n"
        "def _bound_runtime_capabilities():\n"
        "    before = runtime_capabilities.snapshot()\n"
        "    required = ('deadline', 'resolve_host', 'probe_origin', 'emit')\n"
        "    if not all(callable(before.get(name)) for name in required):\n"
        "        raise RuntimeError('host runtime capabilities are unavailable')\n"
        "    runtime_capabilities.register('producer_stage_slot', lambda: _ROUTE_COMMITMENT)\n"
        "    try:\n"
        "        yield before\n"
        "    finally:\n"
        "        runtime_capabilities.restore(before)\n\n"
        "def run_icp_outcome(target, options=None, trace=None, **kwargs):\n"
        "    context = dict(options or {})\n"
        "    probe = context.get('probe') if context.get('mode') == 'consumer_qualification_protocol_probe' else None\n"
        "    if probe is not None:\n"
        "        case_id = probe['case_id']\n"
        "        complete = case_id == 'complete_confirmed_empty'\n"
        "        probe_receipt = {\n"
        "            'schema_version': 'sourcing-model.qualification-outcome-probe.v1',\n"
        "            'case_id': case_id,\n"
        "            'nonce_sha256': hashlib.sha256(probe['nonce'].encode('ascii')).hexdigest(),\n"
        "        }\n"
        "        extensions = {}\n"
        "    else:\n"
        "        complete = context.get('transition_step') == 'complete'\n"
        "        case_id = 'complete_confirmed_empty' if complete else 'incomplete_retryable'\n"
        "        probe_receipt = None\n"
        "        with _bound_runtime_capabilities() as before:\n"
        "            if not callable(runtime_capabilities.capability('producer_stage_slot')):\n"
        "                raise RuntimeError('producer capability binding failed')\n"
        "            request = urllib.request.Request('https://api.exa.ai/search', data=b'{}', method='POST')\n"
        "            with urllib.request.urlopen(request, timeout=2) as response:\n"
        "                response.read()\n"
        "        after = runtime_capabilities.snapshot()\n"
        "        if set(after) != set(before) or any(after[name] is not before[name] for name in before):\n"
        "            raise RuntimeError('host runtime capability scope was not restored')\n"
        "        route_state = 'confirmed_empty' if complete else 'retryable_failed'\n"
        "        extensions = {'com.leadpoet.required-route-outcomes': [{'commitment': _ROUTE_COMMITMENT, 'state': route_state}]}\n"
        "        envelope_extensions = {'com.leadpoet.capability-scope-proof': {'preserved': True, 'restored': True}}\n"
        "    receipt = {\n"
        "        'schema_version': 'sourcing-model.route-completion-receipt.v1',\n"
        f"        'contract_sha256': '{QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2}',\n"
        "        'outcome_authority': 'sourcing_model',\n"
        "        'completion_state': 'complete' if complete else 'incomplete',\n"
        "        'disposition': case_id,\n"
        "        'retryable': not complete,\n"
        "        'partial': False,\n"
        "        'returned_count': 0,\n"
        "        'invocation_sha256': _hash({'icp': target, 'context': context}),\n"
        "        'route_summary': {\n"
        "            'attempted': 1, 'completed': 0,\n"
        "            'confirmed_empty': 1 if complete else 0,\n"
        "            'retryable_failed': 0 if complete else 1,\n"
        "            'terminal_failed': 0, 'skipped': 0, 'retried': 0,\n"
        "        },\n"
        "        'failure_classes': [] if complete else ['retryable_provider'],\n"
        "        'probe': probe_receipt,\n"
        "        'extensions': extensions,\n"
        "    }\n"
        "    receipt['receipt_sha256'] = _hash(receipt)\n"
        "    return {\n"
        "        'schema_version': 'sourcing-model.qualification-outcome.v2',\n"
        "        'protocol_major': 2, 'protocol_minor': 9,\n"
        f"        'contract_sha256': '{QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2}',\n"
        "        'completion_state': receipt['completion_state'],\n"
        "        'companies': [],\n"
        "        'route_completion_receipt': receipt,\n"
        "        'extensions': envelope_extensions if probe is None else {},\n"
        "    }\n",
        encoding="utf-8",
    )
    (source_root / "sourcing_model" / "runtime_capabilities.py").write_text(
        "_CAPABILITIES = {}\n\n"
        "class HostResolution:\n"
        "    TIMEOUT = 'timeout'\n\n"
        "class OriginReachability:\n"
        "    UNKNOWN = 'unknown'\n\n"
        "def register(name, value):\n"
        "    _CAPABILITIES[str(name)] = value\n\n"
        "def capability(name):\n"
        "    return _CAPABILITIES[str(name)]\n\n"
        "def snapshot():\n"
        "    return dict(_CAPABILITIES)\n\n"
        "def restore(value):\n"
        "    _CAPABILITIES.clear()\n"
        "    _CAPABILITIES.update(dict(value))\n",
        encoding="utf-8",
    )
    (source_root / "sourcing_model" / "qualification_route.py").write_text(
        "ROUTE_COMMITMENT_HEADER = 'X-Leadpoet-Qualification-Route-Commitment'\n"
        f"_ROUTE_COMMITMENT = {route_commitment!r}\n\n"
        "def transport_headers():\n"
        "    return {ROUTE_COMMITMENT_HEADER: _ROUTE_COMMITMENT}\n",
        encoding="utf-8",
    )
    source_hash = compute_compatibility_source_tree_hash_v1(source_root)
    manifest["model_artifact_hash"] = source_hash
    compatibility_receipt = source_tree_compatibility_admission(
        source_root,
        manifest=manifest,
        source_tree_hash=source_hash,
    )
    assert compatibility_receipt["admission_mode"] == "qualification_protocol_v2"
    passed("exact_artifact_v2_admission")

    artifact = PrivateModelArtifactManifest(
        model_artifact_hash=source_hash,
        git_commit_sha=manifest["git_commit_sha"],
        image_digest=manifest["image_digest"],
        config_hash="sha256:" + "6" * 64,
        component_registry_version="sourcing-model-components:v2",
        scoring_adapter_version="qualification-company-scorer:v1",
        manifest_uri="s3://private-model/qualification-v2.json",
        manifest_hash=manifest["manifest_hash"],
        signature_ref="kms:qualification-v2-test",
        compatibility_contract=manifest["compatibility_contract"],
        consumer_parity_fixtures=manifest["consumer_parity_fixtures"],
    )
    successor_root = visible / "source-successor"
    shutil.copytree(source_root, successor_root)
    successor_adapter = successor_root / "research_lab_adapter.py"
    successor_adapter.write_text(
        successor_adapter.read_text(encoding="utf-8")
        + "\n_HARMLESS_SUCCESSOR_REVISION = 'optional-refactor-v2'\n",
        encoding="utf-8",
    )
    successor_hash = compute_compatibility_source_tree_hash_v1(successor_root)
    successor_manifest = deepcopy(manifest)
    successor_manifest["model_artifact_hash"] = successor_hash
    successor_receipt = source_tree_compatibility_admission(
        successor_root,
        manifest=successor_manifest,
        source_tree_hash=successor_hash,
    )
    successor_artifact = PrivateModelArtifactManifest(
        model_artifact_hash=successor_hash,
        git_commit_sha=successor_manifest["git_commit_sha"],
        image_digest=successor_manifest["image_digest"],
        config_hash="sha256:" + "6" * 64,
        component_registry_version="sourcing-model-components:v2",
        scoring_adapter_version="qualification-company-scorer:v1",
        manifest_uri="s3://private-model/qualification-v2-successor.json",
        manifest_hash=successor_manifest["manifest_hash"],
        signature_ref="kms:qualification-v2-test",
        compatibility_contract=successor_manifest["compatibility_contract"],
        consumer_parity_fixtures=successor_manifest["consumer_parity_fixtures"],
    )
    assert successor_hash != source_hash
    assert successor_receipt["admission_mode"] == "qualification_protocol_v2"

    def measure_metadata_and_nonce_probes(
        selected_root: Path,
        selected_receipt,
        selected_artifact,
    ):
        observation_plan = _runtime_probe_observation_plan_v1(selected_receipt)
        metadata_completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _MEASURED_METADATA_BOOTSTRAP,
                "research_lab_adapter",
                "adapter_metadata",
            ],
            input=json.dumps({"observation_plan": observation_plan}),
            text=True,
            capture_output=True,
            cwd=tmp_path,
            env={
                "HOME": str(tmp_path),
                "PATH": os.environ.get("PATH", ""),
                "LEADPOET_MODEL_SOURCE_ROOT": str(selected_root),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
            timeout=10,
        )
        assert metadata_completed.returncode == 0, metadata_completed.stderr
        measured_metadata = json.loads(metadata_completed.stdout)
        metadata = validate_sourcing_adapter_metadata(
            measured_metadata["metadata"],
            expected_semantic_bindings={},
            require_company_fit_contract=False,
        )
        return _build_consumer_runtime_probe_from_observation_v1(
            measured_metadata["runtime_observation"],
            compatibility_receipt=selected_receipt,
            metadata=metadata,
            expected_source_tree_hash=selected_artifact.model_artifact_hash,
            expected_manifest_hash=selected_artifact.manifest_hash,
            expected_image_digest=selected_artifact.image_digest,
            expected_module_name="research_lab_adapter",
            expected_callable_name="adapter_metadata",
            observation_plan=observation_plan,
        )

    consumer_probe = measure_metadata_and_nonce_probes(
        source_root,
        compatibility_receipt,
        artifact,
    )
    successor_probe = measure_metadata_and_nonce_probes(
        successor_root,
        successor_receipt,
        successor_artifact,
    )
    assert consumer_probe["invariants"]["profile"] == "qualification_protocol_v2"
    assert set(
        consumer_probe["invariants"]["qualification_outcome_protocol"]["cases"]
    ) == {"complete_confirmed_empty", "incomplete_retryable"}
    assert successor_probe["invariants"]["profile"] == (
        "qualification_protocol_v2"
    )
    passed("metadata_and_fresh_nonce_probes")

    selected_bootstrap = _model_adapter_bootstrap_for_compatibility_receipt_v1(
        compatibility_receipt,
        artifact=artifact,
    )
    assert "getattr(module, 'run_icp_outcome')" in selected_bootstrap
    assert "_research_lab_patch_affected_runtime_capability_scope(module)" not in (
        selected_bootstrap
    )
    successor_bootstrap = _model_adapter_bootstrap_for_compatibility_receipt_v1(
        successor_receipt,
        artifact=successor_artifact,
    )
    assert successor_bootstrap == selected_bootstrap
    passed("versioned_adapter_bootstrap_selection")

    runsc = tmp_path / "runsc"
    runsc.write_bytes(b"qualification-transition-runsc-boundary")
    runsc.chmod(0o755)
    config = RunscSandboxConfigV2(
        runsc_path=runsc,
        runsc_sha256=sha256_bytes(runsc.read_bytes()),
        rootfs_path=rootfs,
        rootfs_manifest_hash=sha256_bytes(marker.read_bytes()),
        python_path="/usr/local/bin/python3",
        uid=os.getuid(),
        gid=os.getgid(),
    )
    repo_root = Path(__file__).resolve().parents[1]
    coordinator_attempts = []
    coordinator_requests = []

    def coordinator(request):
        coordinator_requests.append(deepcopy(dict(request)))
        ordinal = len(coordinator_attempts)
        attempt = build_transport_attempt(
            request_id=f"{ordinal + 1:032x}",
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="api.exa.ai",
            destination_port=443,
            path_hash="sha256:" + "1" * 64,
            nonsecret_headers_hash="sha256:" + "2" * 64,
            body_hash="sha256:" + "3" * 64,
            credential_ref_hash="sha256:" + "4" * 64,
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at=f"2026-08-20T00:00:{ordinal:02d}Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash="sha256:" + "5" * 64,
            request_artifact_hash="sha256:" + "6" * 64,
            response_artifact_hash="sha256:" + "7" * 64,
            tls_peer_chain_hash="sha256:" + "8" * 64,
            tls_protocol="tls1_3",
            failure_code=None,
            completed_at=f"2026-08-20T00:00:{ordinal + 1:02d}Z",
        )
        coordinator_attempts.append(attempt)
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(b'{"ok":true}').decode("ascii"),
            "failure_code": None,
            "encrypted_artifact_id": "sha256:" + "9" * 64,
            "encrypted_request_artifact_id": "sha256:" + "a" * 64,
            "evidence_artifact_hashes": [],
            "transport_attempt": attempt,
        }

    transport = BrokeredProviderTransportV2(coordinator)

    def local_runsc_boundary(command, **kwargs):
        if "run" not in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        bundle_arg = next(
            item for item in command if item.startswith("--bundle=")
        )
        oci = json.loads(
            (Path(bundle_arg.split("=", 1)[1]) / "config.json").read_text()
        )
        process_env = dict(
            item.split("=", 1) for item in oci["process"]["env"]
        )
        for field in (
            "LEADPOET_MODEL_SOURCE_ROOT",
            "LEADPOET_SANDBOX_PROVIDER_SOCKET",
        ):
            if field in process_env:
                process_env[field] = str(
                    rootfs / process_env[field].lstrip("/")
                )
        process_env["PYTHONPATH"] = os.pathsep.join(
            (
                str(repo_root),
                str(source_root),
                *(
                    path
                    for path in sys.path
                    if path.endswith("site-packages") and Path(path).is_dir()
                ),
            )
        )
        process_env["HOME"] = str(tmp_path)
        args = [sys.executable, *oci["process"]["args"][1:]]
        completed = subprocess.run(
            args,
            input=kwargs["input"],
            text=True,
            capture_output=True,
            cwd=tmp_path,
            env=process_env,
            check=False,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        return completed

    sandbox = RunscModelSandboxV2(
        config=config,
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=local_runsc_boundary,
    )
    original_source_bootstrap = model_sandbox_module.model_source_import_bootstrap
    active_source_root = {"value": source_root}
    monkeypatch.setattr(
        model_sandbox_module,
        "model_source_import_bootstrap",
        lambda *_args, **_kwargs: original_source_bootstrap(
            str(active_source_root["value"])
        ),
    )
    runtime_catalog = build_source_add_runtime_catalog_v2([])

    def execute_transition_step(
        step: str,
        *,
        transition_state=None,
        selected_source_root: Path = source_root,
        selected_artifact: PrivateModelArtifactManifest = artifact,
        selected_compatibility_receipt=compatibility_receipt,
    ):
        active_source_root["value"] = selected_source_root
        input_doc = {
            "icp": canonicalize_private_model_icp(
                {
                    "industry": "software",
                    "intent_signal": "hiring",
                    "max_companies": 3,
                }
            ),
            "context": {"transition_step": transition_state or step},
        }
        value = {
            "operation": "run_icp",
            "module_name": "research_lab_adapter",
            "callable_name": "run_icp",
            "input": input_doc,
            "environment": {},
            "provider_cost_scope": sha256_json({"step": step}),
            "provider_evidence_mode": "live",
            "provider_cost_cap_microusd": 0,
            "provider_call_cap": 0,
            "provider_runtime_catalog": runtime_catalog,
            "provider_evidence_cache": {},
        }
        broker_root = visible / ("broker-" + step)
        broker_root.mkdir(mode=0o700)
        os.chown(broker_root, config.uid, config.gid)
        retained_attempts = []
        scope = sandbox._create_provider_scope_v2(
            transport,
            job_id="qualification-transition-" + step,
            purpose="research_lab.private_model_run.v2",
            retry_policy_hashes={"exa": "sha256:" + "b" * 64},
            terminal_sink=lambda attempt: retained_attempts.append(
                deepcopy(dict(attempt))
            ),
            artifact_sink=lambda _artifact: None,
            dynamic_provider_catalog=runtime_catalog,
        )
        server = SandboxProviderSocketServerV2(
            socket_path=broker_root / "provider.sock",
            transport=transport,
            execution_scope=scope,
        )
        server.start()
        run_tmp = tmp_path / ("run-" + step)
        run_tmp.mkdir()
        try:
            envelope, _trace = sandbox._run(
                value,
                artifact=selected_artifact,
                source_root=selected_source_root,
                broker_root=broker_root,
                tmp_root=run_tmp,
                job_id="qualification-transition-" + step,
                provider_snapshot_root=None,
                compatibility_receipt=selected_compatibility_receipt,
            )
        finally:
            server.close()
        scope.assert_accepted_result_is_complete()
        observation = scope.completion_observation()
        _validate_qualification_terminal_observation_v1(
            envelope,
            observation,
        )
        execution_body = {
            "schema_version": "test.strict-execution-boundary.v1",
            "input_hash": sha256_json(input_doc),
            "output_hash": sha256_json(envelope),
            "provider_terminal_observation_hash": sha256_json(observation),
            "attempt_hashes": sorted(
                item["attempt_hash"] for item in retained_attempts
            ),
        }
        execution_root = sha256_json(execution_body)
        authority = _model_qualification_authority_v1(
            envelope=envelope,
            input_doc=input_doc,
            sandbox_result={
                "input_hash": sha256_json(input_doc),
                "provider_terminal_observation": observation,
                "provider_terminal_observation_hash": sha256_json(
                    observation
                ),
            },
            outcome={
                "transport_attempts": retained_attempts,
                "execution_receipt": {"receipt_hash": execution_root},
                "execution_receipt_graph": {
                    "root_receipt_hash": execution_root
                },
            },
            artifact=selected_artifact,
        )
        return envelope, authority, execution_root

    try:
        incomplete_envelope, incomplete_authority, incomplete_root = (
            execute_transition_step("incomplete")
        )
        complete_envelope, complete_authority, complete_root = (
            execute_transition_step("complete")
        )
        successor_envelope, _successor_authority, _successor_root = (
            execute_transition_step(
                "successor-complete",
                transition_state="complete",
                selected_source_root=successor_root,
                selected_artifact=successor_artifact,
                selected_compatibility_receipt=successor_receipt,
            )
        )
    finally:
        transport.restore()
    assert incomplete_envelope["route_completion_receipt"]["disposition"] == (
        "incomplete_retryable"
    )
    assert complete_envelope["route_completion_receipt"]["disposition"] == (
        "complete_confirmed_empty"
    )
    expected_capability_proof = {
        "com.leadpoet.capability-scope-proof": {
            "preserved": True,
            "restored": True,
        }
    }
    assert incomplete_envelope["extensions"] == expected_capability_proof
    assert complete_envelope["extensions"] == expected_capability_proof
    assert successor_envelope["extensions"] == expected_capability_proof
    assert all(
        "x-leadpoet-qualification-route-commitment"
        not in {str(name).lower() for name in request["headers"]}
        for request in coordinator_requests
    )
    assert len(coordinator_requests) == 3
    passed("sandbox_generic_route_hook_and_broker")
    passed("retryable_incomplete_typed_authority")
    passed("complete_confirmed_empty_host_join")
    passed("native_host_capability_scope_preserved_and_restored")
    passed("harmless_successor_artifact_preserves_v2_profile_and_capabilities")
    passed("signed_execution_root_retention")

    incomplete = QualificationOutcomeIncompleteV2Error(
        "qualification incomplete",
        model_qualification_authority=incomplete_authority,
        authority={
            "receipt": {"receipt_hash": incomplete_root},
            "receipt_graph": {"root_receipt_hash": incomplete_root},
        },
    )
    complete = QualificationOutcomeCompleteV2(
        [],
        model_qualification_authority=complete_authority,
    )

    class TransitionRunner:
        def __init__(self):
            self.calls = 0
            self.spec = DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                extra_env={
                    sw.PROVIDER_COST_EVALUATION_SCOPE_ENV: "sha256:" + "c" * 64
                },
            )

        def with_spec(self, spec):
            self.spec = spec
            return self

        async def __call__(self, _icp, _context):
            self.calls += 1
            if self.calls == 1:
                publish_attested_receipt_hash(incomplete_root)
                raise incomplete
            publish_attested_receipt_hash(complete_root)
            return complete

    class NeverScore:
        def __init__(self):
            self.calls = 0

        async def score_with_breakdowns(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("typed incomplete/empty result reached scoring")

    async def no_repo_drift(**_kwargs):
        return None

    monkeypatch.setattr(
        sw,
        "_enforce_baseline_wave_maintenance_boundary",
        no_repo_drift,
    )
    monkeypatch.setattr(sw, "_record_private_baseline_stage", lambda **_: None)
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "production-path-transition-worker"
    worker.config = SimpleNamespace(
        private_baseline_concurrency=1,
        private_baseline_provider_retry_rounds=1,
        private_baseline_retry_concurrency=1,
        scoring_worker_total_workers=1,
    )
    worker._ensure_private_baseline_repo_head_unchanged = no_repo_drift
    runner = TransitionRunner()
    scorer = NeverScore()
    checkpoints = []

    async def checkpoint(row, *, retry_round):
        checkpoints.append(
            sw._baseline_attempt_ledger_entry(
                row,
                retry_round=retry_round,
                gateway_runtime_commit_sha="d" * 40,
            )
        )
        return True

    results, retry_stats = await worker._run_baseline_batch_inner(
        runner=runner,
        retry_runner=runner,
        scorer=scorer,
        window=SimpleNamespace(
            benchmark_items=[
                {
                    "icp": {"industry": "software", "max_companies": 3},
                    "icp_ref": "icp:production-path-transition",
                    "icp_hash": "production-path-transition-hash",
                    "set_id": 1,
                    "day_index": 1,
                    "day_rank": 1,
                }
            ]
        ),
        run_start=time.time(),
        attempt_checkpoint=checkpoint,
        provider_cost_base_scope="sha256:" + "c" * 64,
        benchmark_date="2026-08-20",
    )
    assert retry_stats == {"retried": 1, "recovered": 1, "unresolved": 0}
    assert runner.calls == 2
    assert scorer.calls == 0
    assert len(checkpoints) == 2
    assert checkpoints[0]["result_row"]["_retryable"] is True
    assert checkpoints[0]["result_row"][
        sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD
    ] == [incomplete_root]
    assert checkpoints[1]["result_row"][
        sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD
    ] == [complete_root]
    assert results[0][sw._MODEL_QUALIFICATION_AUTHORITY_FIELD] == (
        complete_authority
    )
    assert "companies" not in json.dumps(checkpoints, sort_keys=True)
    passed("scoring_retry_checkpoint_without_incomplete_scoring")

    assert set(stage_ledger["stages"].values()) == {"passed"}
    print("qualification_transition_stage_ledger=" + json.dumps(
        stage_ledger,
        sort_keys=True,
        separators=(",", ":"),
    ))


def test_harmless_docstring_protocol_mention_routes_to_legacy(monkeypatch, tmp_path):
    root = tmp_path / "docstring-only"
    root.mkdir()
    (root / "research_lab_adapter.py").write_text(
        '"""The qualification_outcome_protocol is documented elsewhere."""\n',
        encoding="utf-8",
    )
    sentinel = {"admission_mode": "legacy-test-sentinel"}
    monkeypatch.setattr(
        contract_check,
        "source_tree_compatibility_admission_v1",
        lambda *_args, **_kwargs: sentinel,
    )

    assert not _qualification_protocol_entrypoint_declared_v2(root)
    assert contract_check.source_tree_compatibility_admission(root) is sentinel


def test_measured_v2_metadata_cannot_execute_under_legacy_admission() -> None:
    policy, policy_hash = semantic_compatibility_policy_identity_v1()
    receipt = {
        "admission_mode": "legacy_exact",
        "consumer_api_version": policy["consumer_api_version"],
        "decision": "accepted",
        "policy_hash": policy_hash,
        "source_tree_hash": "sha256:" + "1" * 64,
        "manifest_hash": "sha256:" + "2" * 64,
        "image_digest": "example.invalid/model@sha256:" + "3" * 64,
    }
    metadata = {"qualification_outcome_protocol": {}}
    probe = _consumer_runtime_probe_v1(
        compatibility_receipt=receipt,
        metadata=metadata,
        expected_module_name="research_lab_adapter",
        expected_callable_name="adapter_metadata",
        invariants={
            "profile": "legacy_exact",
            "qualification_outcome_protocol": {
                "cases": {},
                "nonce_sha256s": {},
            },
        },
    )

    with pytest.raises(ModelSandboxV2Error, match="source admission"):
        validate_consumer_runtime_probe_v1(
            probe,
            compatibility_receipt=receipt,
            metadata=metadata,
            expected_source_tree_hash=receipt["source_tree_hash"],
            expected_manifest_hash=receipt["manifest_hash"],
            expected_image_digest=receipt["image_digest"],
            expected_module_name="research_lab_adapter",
            expected_callable_name="adapter_metadata",
        )


def test_behavior_probe_rejects_failure_to_empty_semantic_drift() -> None:
    complete_nonce = "complete-probe-nonce-0001"
    incomplete_nonce = "incomplete-probe-nonce-01"
    cases = {
        "complete_confirmed_empty": _envelope(
            "complete_confirmed_empty", complete_nonce
        ),
        "incomplete_retryable": _envelope(
            "incomplete_retryable", incomplete_nonce
        ),
    }
    expected_hashes = {
        "complete_confirmed_empty": hashlib.sha256(
            complete_nonce.encode("ascii")
        ).hexdigest(),
        "incomplete_retryable": hashlib.sha256(
            incomplete_nonce.encode("ascii")
        ).hexdigest(),
    }
    validate_qualification_outcome_protocol_probe_cases_v1(
        cases,
        expected_nonce_sha256s=expected_hashes,
    )

    drifted = deepcopy(cases)
    drifted_case = drifted["incomplete_retryable"]
    drifted_case["completion_state"] = "complete"
    receipt = drifted_case["route_completion_receipt"]
    receipt.update(
        {
            "completion_state": "complete",
            "disposition": "complete_confirmed_empty",
            "retryable": False,
            "failure_classes": [],
            "route_summary": {
                "attempted": 1,
                "completed": 0,
                "confirmed_empty": 1,
                "retryable_failed": 0,
                "terminal_failed": 0,
                "skipped": 0,
                "retried": 0,
            },
        }
    )
    receipt_body = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = _plain_hash(receipt_body)
    validate_qualification_outcome_envelope_v2(drifted_case)

    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_protocol_probe_cases_v1(
            drifted,
            expected_nonce_sha256s=expected_hashes,
        )
    with pytest.raises(PrivateModelRuntimeError):
        validate_qualification_outcome_envelope_v2([])


def test_host_join_uses_latest_logical_attempt_and_rejects_latest_http_500() -> None:
    route_commitment = "c" * 64
    failed = _attempt(ordinal=0, terminal="transport_failure")
    recovered = _attempt(
        ordinal=1,
        terminal="authenticated_response",
        http_status=200,
    )
    recovered_observation = {
        "schema_version": "leadpoet.provider-terminal-observation.v1",
        "request_intent_count": 2,
        "terminal_count": 2,
        "latest_operation_count": 1,
        "accepted_latest_terminal_count": 1,
        "successful_latest_terminal_count": 1,
        "failed_latest_terminal_count": 0,
        "unresolved_latest_terminal_count": 0,
        "latest_terminal_attempt_hashes": [recovered["attempt_hash"]],
        "successful_latest_terminal_attempt_hashes": [
            recovered["attempt_hash"]
        ],
        "required_route_commitments": [route_commitment],
        "required_route_count": 1,
        "successful_required_route_count": 1,
        "unresolved_required_route_count": 0,
        "required_route_terminals": [
            {
                "route_commitment": route_commitment,
                "attempt_hash": recovered["attempt_hash"],
                "terminal_status": "authenticated_response",
                "http_status": 200,
            }
        ],
    }
    joined = _host_provider_observation_v1(
        [failed, recovered], recovered_observation
    )
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    envelope_receipt = envelope["route_completion_receipt"]
    envelope_receipt["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes([route_commitment], "confirmed_empty")
        )
    }
    envelope_receipt["receipt_sha256"] = _plain_hash(
        {
            key: value
            for key, value in envelope_receipt.items()
            if key != "receipt_sha256"
        }
    )
    _validate_qualification_terminal_observation_v1(
        envelope, recovered_observation
    )
    assert joined["successful_latest_terminal_count"] == 1
    assert joined["unresolved_latest_terminal_count"] == 0

    latest_500 = _attempt(
        ordinal=2,
        terminal="authenticated_response",
        http_status=500,
    )
    failed_observation = {
        **recovered_observation,
        "request_intent_count": 3,
        "terminal_count": 3,
        "successful_latest_terminal_count": 0,
        "unresolved_latest_terminal_count": 1,
        "latest_terminal_attempt_hashes": [latest_500["attempt_hash"]],
        "successful_latest_terminal_attempt_hashes": [],
        "successful_required_route_count": 0,
        "unresolved_required_route_count": 1,
        "required_route_terminals": [
            {
                "route_commitment": route_commitment,
                "attempt_hash": latest_500["attempt_hash"],
                "terminal_status": "authenticated_response",
                "http_status": 500,
            }
        ],
    }
    joined_failed = _host_provider_observation_v1(
        [failed, recovered, latest_500], failed_observation
    )
    assert joined_failed["unresolved_latest_terminal_count"] == 1
    with pytest.raises(ModelSandboxV2Error):
        _validate_qualification_terminal_observation_v1(
            envelope, failed_observation
        )


@pytest.mark.parametrize(
    ("terminal_status", "http_status", "successful"),
    [
        ("authenticated_response", 200, True),
        ("attested_local_response", 200, True),
        ("authenticated_response", 300, False),
        ("authenticated_response", 400, False),
        ("authenticated_response", 500, False),
        ("transport_failure", None, False),
    ],
)
def test_raw_required_route_success_counter_uses_attested_2xx_only(
    terminal_status,
    http_status,
    successful,
) -> None:
    scope = _provider_scope()
    commitment = "d" * 64
    scope.record_intent("required-route", 0, commitment)
    scope.record_terminal(
        "required-route",
        0,
        terminal_status,
        http_status,
        "sha256:" + "e" * 64,
    )

    observation = scope.completion_observation()

    assert observation["successful_required_route_count"] == int(successful)
    assert observation["unresolved_required_route_count"] == int(not successful)


@pytest.mark.parametrize(
    ("state", "http_status", "accepted"),
    [
        ("completed", 200, True),
        ("completed", 404, False),
        ("completed", 410, False),
        ("confirmed_empty", 200, True),
        ("confirmed_empty", 404, True),
        ("confirmed_empty", 410, True),
        ("confirmed_empty", 403, False),
        ("confirmed_empty", 500, False),
        ("retryable_failed", 200, False),
    ],
)
def test_complete_route_join_is_model_state_and_status_aware(
    state,
    http_status,
    accepted,
) -> None:
    commitment = "c" * 64
    scope = _provider_scope()
    scope.record_intent("state-aware-route", 0, commitment)
    scope.record_terminal(
        "state-aware-route",
        0,
        "authenticated_response",
        http_status,
        "sha256:" + "d" * 64,
    )
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    receipt = envelope["route_completion_receipt"]
    receipt["probe"] = None
    receipt["route_summary"] = {
        "attempted": 1,
        "completed": int(state == "completed"),
        "confirmed_empty": int(state == "confirmed_empty"),
        "retryable_failed": int(state == "retryable_failed"),
        "terminal_failed": int(state == "terminal_failed"),
        "skipped": 0,
        "retried": 0,
    }
    receipt["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes([commitment], state)
        )
    }
    _rehash_receipt(envelope)

    if accepted:
        _validate_qualification_terminal_observation_v1(
            envelope,
            scope.completion_observation(),
        )
    else:
        with pytest.raises(ModelSandboxV2Error):
            _validate_qualification_terminal_observation_v1(
                envelope,
                scope.completion_observation(),
            )


def test_required_route_replacement_supersedes_by_commitment_sequence() -> None:
    scope = _provider_scope()
    commitment = "f" * 64
    scope.record_intent("old-logical-operation", 0, commitment)
    scope.record_terminal(
        "old-logical-operation",
        0,
        "transport_failure",
        None,
        "sha256:" + "1" * 64,
    )
    scope.record_intent("replacement-logical-operation", 0, commitment)
    scope.record_terminal(
        "replacement-logical-operation",
        0,
        "attested_local_response",
        200,
        "sha256:" + "2" * 64,
    )

    observation = scope.completion_observation()

    assert observation["required_route_count"] == 1
    assert observation["successful_required_route_count"] == 1
    assert observation["unresolved_required_route_count"] == 0
    assert observation["required_route_terminals"] == [
        {
            "route_commitment": commitment,
            "attempt_hash": "sha256:" + "2" * 64,
            "terminal_status": "attested_local_response",
            "http_status": 200,
        }
    ]


def test_same_route_commitment_cannot_be_concurrently_in_flight() -> None:
    scope = _provider_scope()
    commitment = "9" * 64
    scope.record_intent("first-operation", 0, commitment)

    with pytest.raises(ProviderClientV2Error, match="in-flight intent"):
        scope.record_intent("racing-operation", 0, commitment)

    scope.record_intent("different-slot", 0, "8" * 64)
    scope.record_terminal(
        "first-operation",
        0,
        "transport_failure",
        None,
        "sha256:" + "7" * 64,
    )
    scope.record_intent("sequential-replacement", 0, commitment)


def test_distinct_required_slots_can_share_one_request_fingerprint() -> None:
    scope = _provider_scope()
    first_commitment = "4" * 64
    second_commitment = "5" * 64
    first_attempt = _attempt(
        ordinal=0,
        terminal="authenticated_response",
        http_status=200,
        logical_operation_id="shared-request-fingerprint",
    )
    second_attempt = _attempt(
        ordinal=1,
        terminal="authenticated_response",
        http_status=200,
        logical_operation_id="shared-request-fingerprint",
    )
    scope.record_intent(
        "shared-request-fingerprint",
        0,
        first_commitment,
    )
    scope.record_intent(
        "shared-request-fingerprint",
        1,
        second_commitment,
    )
    for ordinal, commitment_attempt in enumerate(
        (first_attempt, second_attempt)
    ):
        scope.record_terminal(
            "shared-request-fingerprint",
            ordinal,
            "authenticated_response",
            200,
            commitment_attempt["attempt_hash"],
        )
    observation = scope.completion_observation()
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    receipt = envelope["route_completion_receipt"]
    receipt["route_summary"]["attempted"] = 2
    receipt["route_summary"]["confirmed_empty"] = 2
    receipt["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes(
                [first_commitment, second_commitment],
                "confirmed_empty",
            )
        )
    }
    receipt["receipt_sha256"] = _plain_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )

    assert observation["latest_terminal_attempt_hashes"] == [
        second_attempt["attempt_hash"]
    ]
    assert len(observation["required_route_terminals"]) == 2
    _validate_qualification_terminal_observation_v1(envelope, observation)
    joined = _host_provider_observation_v1(
        [first_attempt, second_attempt],
        observation,
    )
    assert joined["successful_required_route_count"] == 2
    assert joined["unresolved_required_route_count"] == 0


def test_optional_failed_call_does_not_veto_bound_required_route_success() -> None:
    commitment = "b" * 64
    optional_failure = _attempt(
        ordinal=0,
        terminal="transport_failure",
        logical_operation_id="optional-discovery-route",
    )
    required_success = _attempt(
        ordinal=0,
        terminal="attested_local_response",
        http_status=200,
        logical_operation_id="required-adjudication-route",
    )
    latest_hashes = sorted(
        [optional_failure["attempt_hash"], required_success["attempt_hash"]]
    )
    observation = {
        "schema_version": "leadpoet.provider-terminal-observation.v1",
        "request_intent_count": 2,
        "terminal_count": 2,
        "latest_operation_count": 2,
        "accepted_latest_terminal_count": 1,
        "successful_latest_terminal_count": 1,
        "failed_latest_terminal_count": 1,
        "unresolved_latest_terminal_count": 1,
        "latest_terminal_attempt_hashes": latest_hashes,
        "successful_latest_terminal_attempt_hashes": [
            required_success["attempt_hash"]
        ],
        "required_route_commitments": [commitment],
        "required_route_count": 1,
        "successful_required_route_count": 1,
        "unresolved_required_route_count": 0,
        "required_route_terminals": [
            {
                "route_commitment": commitment,
                "attempt_hash": required_success["attempt_hash"],
                "terminal_status": "attested_local_response",
                "http_status": 200,
            }
        ],
    }
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    receipt = envelope["route_completion_receipt"]
    receipt["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes([commitment], "confirmed_empty")
        )
    }
    receipt["receipt_sha256"] = _plain_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )

    _validate_qualification_terminal_observation_v1(envelope, observation)
    joined = _host_provider_observation_v1(
        [optional_failure, required_success], observation
    )
    assert joined["unresolved_latest_terminal_count"] == 1
    assert joined["unresolved_required_route_count"] == 0

    tampered = deepcopy(observation)
    tampered["required_route_terminals"][0]["attempt_hash"] = (
        "sha256:" + "f" * 64
    )
    with pytest.raises(AttestedPrivateModelRunnerV2Error):
        _host_provider_observation_v1(
            [optional_failure, required_success], tampered
        )


@pytest.mark.parametrize(
    ("terminal_status", "http_status", "accepted"),
    [
        ("authenticated_response", 200, True),
        ("transport_failure", None, False),
    ],
)
def test_complete_nonempty_requires_every_bound_route_success(
    terminal_status,
    http_status,
    accepted,
) -> None:
    commitment = "e" * 64
    attempt = _attempt(
        ordinal=0,
        terminal=terminal_status,
        http_status=http_status,
        logical_operation_id="required-nonempty-route",
    )
    scope = _provider_scope()
    scope.record_intent("required-nonempty-route", 0, commitment)
    scope.record_terminal(
        "required-nonempty-route",
        0,
        terminal_status,
        http_status,
        attempt["attempt_hash"],
    )
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    envelope["companies"] = [{"name": "Qualified Co"}]
    receipt = envelope["route_completion_receipt"]
    receipt.update(
        {
            "probe": None,
            "disposition": "complete_nonempty",
            "returned_count": 1,
            "route_summary": {
                "attempted": 1,
                "completed": 1,
                "confirmed_empty": 0,
                "retryable_failed": 0,
                "terminal_failed": 0,
                "skipped": 0,
                "retried": 0,
            },
            "extensions": {
                QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
                    _required_route_outcomes([commitment], "completed")
                )
            },
        }
    )
    _rehash_receipt(envelope)
    validate_qualification_outcome_envelope_v2(envelope)
    observation = scope.completion_observation()

    if accepted:
        _validate_qualification_terminal_observation_v1(
            envelope,
            observation,
        )
        joined = _host_provider_observation_v1([attempt], observation)
        assert joined["successful_required_route_count"] == 1
    else:
        with pytest.raises(
            ModelSandboxV2Error,
            match="qualification outcome required routes differ",
        ):
            _validate_qualification_terminal_observation_v1(
                envelope,
                observation,
            )


def test_complete_authority_cannot_hide_one_failed_required_slot() -> None:
    failed_commitment = "1" * 64
    successful_commitment = "2" * 64
    failed = _attempt(
        ordinal=0,
        terminal="transport_failure",
        logical_operation_id="required-slot-a",
    )
    successful = _attempt(
        ordinal=0,
        terminal="authenticated_response",
        http_status=200,
        logical_operation_id="required-slot-b",
    )
    scope = _provider_scope()
    for operation_id, commitment, attempt in (
        ("required-slot-a", failed_commitment, failed),
        ("required-slot-b", successful_commitment, successful),
    ):
        scope.record_intent(operation_id, 0, commitment)
        scope.record_terminal(
            operation_id,
            0,
            attempt["terminal_status"],
            attempt["http_status"],
            attempt["attempt_hash"],
        )

    input_doc = {"icp": {"industry": "software"}, "context": {}}
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    envelope["companies"] = [{"name": "Qualified Co"}]
    receipt = envelope["route_completion_receipt"]
    receipt.update(
        {
            "probe": None,
            "disposition": "complete_nonempty",
            "returned_count": 1,
            "invocation_sha256": _plain_hash(input_doc),
            "route_summary": {
                "attempted": 2,
                "completed": 2,
                "confirmed_empty": 0,
                "retryable_failed": 0,
                "terminal_failed": 0,
                "skipped": 0,
                "retried": 0,
            },
            "extensions": {
                QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
                    _required_route_outcomes(
                        [failed_commitment, successful_commitment],
                        "completed",
                    )
                )
            },
        }
    )
    _rehash_receipt(envelope)
    observation = scope.completion_observation()
    with pytest.raises(
        ModelSandboxV2Error,
        match="qualification outcome required routes differ",
    ):
        _validate_qualification_terminal_observation_v1(
            envelope,
            observation,
        )

    execution_root = sha256_json({"execution": "failed-required-slot"})
    artifact = PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "3" * 64,
        git_commit_sha="4" * 40,
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:"
            + "5" * 64
        ),
        config_hash="sha256:" + "6" * 64,
        component_registry_version="sourcing-model-components:v2",
        scoring_adapter_version="qualification-company-scorer:v1",
        manifest_uri="s3://private-model/qualification-v2.json",
        manifest_hash="sha256:" + "7" * 64,
        signature_ref="kms:qualification-v2-test",
    )
    sandbox_result = {
        "input_hash": sha256_json(input_doc),
        "provider_terminal_observation": observation,
        "provider_terminal_observation_hash": sha256_json(observation),
    }
    outcome = {
        "transport_attempts": [failed, successful],
        "execution_receipt": {"receipt_hash": execution_root},
        "execution_receipt_graph": {"root_receipt_hash": execution_root},
    }
    with pytest.raises(
        AttestedPrivateModelRunnerV2Error,
        match="qualification outcome lacks exact required-route authority",
    ):
        _model_qualification_authority_v1(
            envelope=envelope,
            input_doc=input_doc,
            sandbox_result=sandbox_result,
            outcome=outcome,
            artifact=artifact,
        )


def test_production_outcome_requires_exact_route_extension() -> None:
    commitment = "f" * 64
    attempt = _attempt(
        ordinal=0,
        terminal="authenticated_response",
        http_status=200,
        logical_operation_id="required-empty-route",
    )
    scope = _provider_scope()
    scope.record_intent("required-empty-route", 0, commitment)
    scope.record_terminal(
        "required-empty-route",
        0,
        "authenticated_response",
        200,
        attempt["attempt_hash"],
    )
    envelope = _envelope(
        "complete_confirmed_empty", "complete-probe-nonce-0001"
    )
    envelope["route_completion_receipt"]["probe"] = None
    _rehash_receipt(envelope)

    with pytest.raises(ModelSandboxV2Error, match="required routes"):
        _validate_qualification_terminal_observation_v1(
            envelope,
            scope.completion_observation(),
        )


def test_incomplete_semantics_may_have_successful_required_transport() -> None:
    commitment = "a" * 64
    attempt = _attempt(
        ordinal=0,
        terminal="attested_local_response",
        http_status=200,
        logical_operation_id="semantic-incomplete-route",
    )
    scope = _provider_scope()
    scope.record_intent("semantic-incomplete-route", 0, commitment)
    scope.record_terminal(
        "semantic-incomplete-route",
        0,
        "attested_local_response",
        200,
        attempt["attempt_hash"],
    )
    envelope = _envelope(
        "incomplete_retryable", "incomplete-probe-nonce-01"
    )
    receipt = envelope["route_completion_receipt"]
    receipt["probe"] = None
    receipt["extensions"] = {
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2: (
            _required_route_outcomes([commitment], "retryable_failed")
        )
    }
    _rehash_receipt(envelope)

    _validate_qualification_terminal_observation_v1(
        envelope,
        scope.completion_observation(),
    )
