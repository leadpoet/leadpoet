from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.code_loop_engine import (
    BuiltCodeEditCandidate,
    CodeEditLoopResult,
)
from gateway.research_lab.git_tree_models import (
    TreeCheckpoint,
    TreePolicy,
    TreeReplacement,
    TreeResult,
    derive_child_slot,
    derive_tree_id,
)
from gateway.research_lab.code_build import (
    CodeEditArtifactMissingError,
    CodeEditBuildError,
    CodeEditBuildResult,
    CodeEditCandidateBuilder,
    CodeEditEmptyOrNoopPatchError,
    CodeEditImageBuildError,
    CodeEditInfraFailureError,
    CodeEditPatchApplyError,
    CodeEditPrivateTestError,
    _copy_source_tree,
    _initialize_temporary_git_repo,
    _prepare_parent_image_workspace,
    _run_git_apply,
    _write_research_lab_build_scaffold,
)
from gateway.research_lab.git_tree_repository import GitTreeRepository
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.autoresearch_runtime import AutoResearchLoopEvent
from gateway.tee.autoresearch_executor_v2 import (
    AUTORESEARCH_REQUEST_SCHEMA_VERSION,
    COMPONENT_REGISTRY_EVIDENCE_PURPOSE_V2,
    HOST_APPEND_EVENT,
    HOST_EVENT_RESULT_SCHEMA_VERSION,
    HOST_GIT_TREE,
    HOST_GIT_TREE_COMMIT_SCHEMA_VERSION,
    HOST_GIT_TREE_RESULT_SCHEMA_VERSION,
    OPENROUTER_GUARD_REQUEST_SCHEMA_VERSION,
    OP_REPAIR_STALE_PARENT,
    OP_RUN_CODE_EDIT_LOOP,
    OP_VERIFY_OPENROUTER_GUARD,
    STALE_PARENT_REPAIR_REQUEST_SCHEMA_VERSION,
    STALE_PARENT_REPAIR_RESULT_SCHEMA_VERSION,
    AutoresearchExecutorV2,
    AutoresearchExecutorV2Error,
    _HostCandidateBuilder,
    _HostGitTreeRepository,
    _candidate_document,
    _raise_code_edit_host_operation_failure,
    _source_context,
    _validate_build_result,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.host_operation_channel_v2 import HostOperationV2Error
from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
from gateway.tee.source_bundle_v2 import (
    build_source_bundle_v2,
    extract_source_bundle_v2,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    ROLE_PURPOSES,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    canonical_json,
    create_boot_identity,
    create_signed_execution_receipt,
    sha256_bytes,
    sha256_json,
)
from research_lab.auto_research_prompt import coerce_component_registry
from research_lab.eval import (
    PrivateModelArtifactManifest,
    private_model_artifact_replay_identity_v2,
)
from research_lab.code_editing import CodeEditDraft, code_edit_candidate_manifest
from research_lab.eval.private_runtime import compute_private_source_tree_hash
from tests.private_model_artifact_fixtures import (
    build_private_artifact_with_adapted_source_admission,
    install_reviewed_consumer_snapshot,
)


class _HostChannel:
    def __init__(self) -> None:
        self.records = []

    def execute(
        self,
        *,
        operation,
        payload,
        expected_state_hash,
        timeout_seconds,
        response_validator,
    ):
        assert timeout_seconds > 0
        assert operation == HOST_APPEND_EVENT
        response = {
            "schema_version": HOST_EVENT_RESULT_SCHEMA_VERSION,
            "event_hash": payload["event_hash"],
            "event_sequence": payload["event_sequence"],
            "next_state_hash": sha256_json(
                {
                    "previous": expected_state_hash,
                    "event_hash": payload["event_hash"],
                }
            ),
        }
        normalized = response_validator(response)
        self.records.append(
            {
                "operation": operation,
                "payload": dict(payload),
                "expected_state_hash": expected_state_hash,
                "response": normalized,
            }
        )
        return normalized

    def complete_ledger(self):
        return ()


class _FakeEngine:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.run_kwargs = None
        self.__class__.instances.append(self)

    async def run(self, **kwargs):
        self.run_kwargs = kwargs
        await self.kwargs["event_sink"](
            AutoResearchLoopEvent(
                event_type="loop_started",
                loop_status="running",
                elapsed_seconds=0.0,
                event_doc={"run_id": kwargs["run_id"]},
            )
        )
        await self.kwargs["event_sink"](
            AutoResearchLoopEvent(
                event_type="loop_failed",
                loop_status="failed",
                elapsed_seconds=1.25,
                event_doc={"run_id": kwargs["run_id"], "candidate_count": 0},
            )
        )
        policy = TreePolicy.from_mapping(
            kwargs["budget_context"]["tree_policy"]["policy"]
        )
        replacement_doc = kwargs["budget_context"].get("tree_replacement")
        replacement = (
            TreeReplacement.from_mapping(replacement_doc)
            if isinstance(replacement_doc, dict)
            else None
        )
        tree_id = derive_tree_id(
            run_id=kwargs["run_id"],
            root_artifact_hash=kwargs["artifact"].model_artifact_hash,
            policy=policy,
            replacement=replacement,
        )
        checkpoint = TreeCheckpoint(
            tree_id=tree_id,
            root_artifact_hash=kwargs["artifact"].model_artifact_hash,
            policy=policy,
            nodes=(),
            frontier_hash="sha256:" + "7" * 64,
            operation_settlement_hash="sha256:" + "8" * 64,
            stop_reason="tree_final_selection_committed",
        )
        tree_result = TreeResult(
            tree_id=tree_id,
            status="failed",
            stop_reason="no_eligible_tree_finalist",
            selected_node_id="",
            nodes=(),
            checkpoint=checkpoint,
        )
        return CodeEditLoopResult(
            selected_candidates=(),
            iterations_completed=1,
            stop_reason="no_eligible_tree_finalist",
            elapsed_seconds=1.25,
            estimated_cost_usd=0.5,
            actual_openrouter_cost_usd=0.0,
            actual_openrouter_cost_microusd=0,
            openrouter_call_count=0,
            tree_result=tree_result,
            status="failed",
            checkpoint_doc={"git_tree_checkpoint": checkpoint.to_dict()},
        )


def _source_and_artifact(tmp_path: Path):
    root = tmp_path / "private-source"
    for directory in (
        "gateway/research_lab",
        "qualification/scoring",
        "sourcing_model",
        "validator_models",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "research_lab_adapter.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("", encoding="utf-8")
    install_reviewed_consumer_snapshot(root)
    manifest = build_private_artifact_with_adapted_source_admission(
        source_path=root,
        git_commit_sha="a" * 40,
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
            + "b" * 64
        ),
        manifest_uri="s3://private/manifests/current.json",
        signature_ref="kms:signature",
        component_registry_version="1",
        scoring_adapter_version="1",
    )
    return build_source_bundle_v2(root), manifest


def _valid_image_build_response(tmp_path: Path):
    _, parent_doc = _source_and_artifact(tmp_path)
    parent = PrivateModelArtifactManifest.from_mapping(parent_doc)
    child_root = tmp_path / "candidate-source"
    shutil.copytree(tmp_path / "private-source", child_root)
    (child_root / "sourcing_model" / "runtime.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    candidate = PrivateModelArtifactManifest.from_mapping(
        build_private_artifact_with_adapted_source_admission(
            source_path=child_root,
            git_commit_sha="c" * 40,
            image_digest=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
                + "d" * 64
            ),
            manifest_uri="s3://private/manifests/candidate.json",
            signature_ref="kms:candidate-signature",
            component_registry_version="1",
            scoring_adapter_version="1",
        )
    )
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    build_payload = {
        "schema_version": "1.1",
        "candidate_kind": "image_build",
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_model_artifact_hash": candidate.model_artifact_hash,
        "candidate_model_manifest_hash": candidate.manifest_hash,
        "source_diff_hash": source_diff_hash,
    }
    build_doc = {
        **build_payload,
        "build_doc_hash": sha256_json(build_payload),
    }
    code_edit_manifest = code_edit_candidate_manifest(
        draft=draft,
        parent_artifact_hash=parent.model_artifact_hash,
        candidate_artifact_hash=candidate.model_artifact_hash,
        candidate_model_manifest_hash=candidate.manifest_hash,
        source_diff_hash=source_diff_hash,
        build_doc_hash=build_doc["build_doc_hash"],
    )
    response = {
        "candidate_model_manifest": candidate.to_dict(),
        "code_edit_manifest": code_edit_manifest,
        "source_diff_hash": source_diff_hash,
        "build_doc": build_doc,
    }
    return response, draft, parent, candidate


def test_v2_build_result_accepts_canonical_image_build_manifest(tmp_path):
    response, draft, parent, candidate = _valid_image_build_response(tmp_path)

    result = _validate_build_result(
        response,
        draft=draft,
        parent_artifact=parent,
        expected_candidate_artifact_hash=candidate.model_artifact_hash,
    )

    assert result.candidate_model_manifest == candidate
    assert result.code_edit_manifest == response["code_edit_manifest"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("patch_type",), "PROMPT_EDIT"),
        (("parent_artifact_hash",), "sha256:" + "e" * 64),
        (("candidate_model_manifest_hash",), "sha256:" + "e" * 64),
        (("candidate_source_diff_hash",), "sha256:" + "e" * 64),
        (("candidate_build_doc_hash",), "sha256:" + "e" * 64),
        (("patch_doc", "target_files"), ["sourcing_model/other.py"]),
    ],
)
def test_v2_build_result_rejects_self_consistent_forged_image_build_manifest(
    tmp_path,
    path,
    replacement,
):
    response, draft, parent, candidate = _valid_image_build_response(tmp_path)
    forged = deepcopy(response)
    target = forged["code_edit_manifest"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    payload = {
        key: value
        for key, value in forged["code_edit_manifest"].items()
        if key != "manifest_hash"
    }
    forged["code_edit_manifest"]["manifest_hash"] = sha256_json(payload)

    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="candidate code-edit manifest differs from measured build inputs",
    ):
        _validate_build_result(
            forged,
            draft=draft,
            parent_artifact=parent,
            expected_candidate_artifact_hash=candidate.model_artifact_hash,
        )


def _payload(tmp_path: Path):
    source_bundle, artifact = _source_and_artifact(tmp_path)
    tree_policy = TreePolicy(mode="active")
    run_id = "run-v2-1"
    queue_event_hash = "sha256:" + "a" * 64
    privacy_proof_doc = {"status": "verified"}
    openrouter_key_ref = "encrypted_ref:openrouter:" + "1" * 32
    runtime_credential_hash = "sha256:" + "5" * 64
    management_credential_hash = "sha256:" + "6" * 64
    guard_result = {
        "schema_version": "leadpoet.openrouter_guard_result.v3",
        "key_ref_hash": sha256_bytes(openrouter_key_ref.encode("utf-8")),
        "miner_hotkey_hash": sha256_bytes(b"miner-hotkey"),
        "runtime_credential_value_hash": runtime_credential_hash,
        "management_credential_value_hash": management_credential_hash,
        "run_state_hash": sha256_json(
            {"run_id": run_id, "queue_event_hash": queue_event_hash}
        ),
        "preflight_status": "passed",
        "preflight_error_type": "",
        "credit_depleted": False,
        "credit_limit_remaining": 1,
        "privacy_proof_doc": privacy_proof_doc,
    }
    guard_graph = _openrouter_guard_graph(guard_result)
    active_model_result = {
        "schema_version": "leadpoet.active_private_model.v2",
        "artifact": private_model_artifact_replay_identity_v2(artifact),
        "active_model": {
            "private_model_version_id": "private-model-v1",
        },
        "source_state_hash": "sha256:" + "f" * 64,
    }
    active_model_graph = _active_model_graph(active_model_result)
    component_metadata = {
        "adapter_version": "adapter:v1",
        "component_registry_version": "components:v1",
        "scoring_adapter_version": "scoring:v1",
        "component_registry": {
            "source_router": {
                "purpose": "Select the source strategy",
                "input_contract": "validated ICP",
                "output_contract": "source query",
                "ablation_leverage": 1.0,
                "allowed_patch_types": ["PROMPT_EDIT"],
                "max_instruction_chars": 800,
                "cost_budget_cents": 10,
            }
        },
    }
    component_registry = coerce_component_registry(component_metadata).to_dict()
    component_result = {
        "schema_version": "leadpoet.model_sandbox_result.v2",
        "operation": "metadata",
        "output": component_metadata,
        "output_hash": sha256_json(component_metadata),
    }
    component_graph = _component_registry_graph(component_result)
    runtime_catalog_body = {
        "schema_version": "leadpoet.source_add_runtime_catalog.v2",
        "routes": [],
    }
    runtime_catalog = {
        **runtime_catalog_body,
        "catalog_hash": sha256_json(runtime_catalog_body),
    }
    catalog_result = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": [],
        "provisioned_sources_hash": sha256_json([]),
        "private_registry_rows": [],
        "private_registry_rows_hash": sha256_json([]),
        "runtime_catalog": runtime_catalog,
        "runtime_catalog_hash": runtime_catalog["catalog_hash"],
    }
    catalog_graph = _provider_catalog_graph(catalog_result)
    provider_outcome_result = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T20:00:00Z"
    ).snapshot()
    provider_outcome_graph = _provider_outcome_graph(provider_outcome_result)
    return {
        "schema_version": AUTORESEARCH_REQUEST_SCHEMA_VERSION,
        "run_id": run_id,
        "ticket": {"ticket_id": "ticket-1", "requested_loop_count": 1},
        "artifact": artifact,
        "component_registry": component_registry,
        "component_registry_evidence": {
            "result": component_result,
            "receipt_graph": component_graph,
            "root_receipt_hash": component_graph["root_receipt_hash"],
        },
        "active_model_evidence": {
            "result": active_model_result,
            "receipt_graph": active_model_graph,
            "root_receipt_hash": active_model_graph["root_receipt_hash"],
        },
        "provider_catalog_evidence": {
            "result": catalog_result,
            "receipt_graph": catalog_graph,
            "root_receipt_hash": catalog_graph["root_receipt_hash"],
        },
        "provider_outcome_evidence": {
            "result": provider_outcome_result,
            "receipt_graph": provider_outcome_graph,
            "root_receipt_hash": provider_outcome_graph["root_receipt_hash"],
        },
        "benchmark_public_summary": {},
        "model_id": "openai/test-model",
        "model_doc": {},
        "budget_context": {
            "requested_compute_budget_usd": 1.0,
            "tree_policy": {
                "schema_version": "research_lab.git_tree_runtime_policy.v2",
                "policy": tree_policy.to_dict(),
                "evaluator_enabled": True,
                "evaluator_commitment": {
                    "schema_version": "research_lab.git_tree_evaluator_commitment.v3",
                    "resolved_snapshot_uri": (
                        "s3://private-dev-snapshots/"
                        + "7" * 64
                    ),
                    "snapshot_pointer_hash": "sha256:" + "6" * 64,
                    "snapshot_manifest_hash": "sha256:" + "7" * 64,
                    "snapshot_ready_hash": "sha256:" + "8" * 64,
                    "dev_set_hash": "sha256:" + "9" * 64,
                    "dev_set_size": tree_policy.live_max_icps_per_node,
                    "snapshot_bank_hash": "sha256:" + "b" * 64,
                    "snapshot_bank_size": 40,
                    "daily_bank_hash": "sha256:" + "c" * 64,
                    "selection_manifest_hash": "sha256:" + "d" * 64,
                    "selection_seed_hash": "sha256:" + "e" * 64,
                    "miner_direction_hash": "sha256:" + "f" * 64,
                    "benchmark_date": "2026-07-10",
                    "benchmark_bundle_id": "private_benchmark:" + "0" * 64,
                    "benchmark_bundle_hash": "sha256:" + "1" * 64,
                    "rolling_window_hash": "sha256:" + "2" * 64,
                    "private_model_manifest_hash": "sha256:" + "3" * 64,
                    "champion_image_digest": artifact["image_digest"],
                    "source_commit": artifact["git_commit_sha"],
                    "model_config_hash": "sha256:" + "a" * 64,
                    "provider_model_ids": [],
                    "miss_policy": "strict",
                    "score_version": "research_lab.dev_eval.v2",
                    "evaluation_timeout_seconds": 300,
                    "live_max_icps_per_node": tree_policy.live_max_icps_per_node,
                    "live_max_provider_calls": 32,
                    "live_cap_microusd": 500000,
                    "minimum_evidence_retention_days": 30,
                },
                "prior_evaluation_provider_call_count": 0,
                "prior_evaluation_cost_microusd": 0,
                "evaluation_provider_call_budget_charge": 0,
                "evaluation_cost_budget_charge_microusd": 0,
                "unsettled_evaluation_operation_count": 0,
                "unsettled_evaluation_operations_hash": sha256_json([]),
                "indeterminate_evaluation_operation_count": 0,
                "indeterminate_evaluation_operations_hash": sha256_json([]),
                "snapshot_age_seconds": 0.0,
            },
        },
        "requested_loop_count": 1,
        "resume_state": {},
        "loop_settings": {
            "min_seconds": 0,
            "max_seconds": 2700,
            "min_iterations": 1,
            "max_iterations": 1,
            "draft_timeout_seconds": 30,
            "reflection_timeout_seconds": 30,
            "estimated_iteration_cost_usd": 0.5,
            "max_candidates": 6,
        },
        "source_bundle": source_bundle,
        "probe_private_window_term_hashes": [],
        "provider_outcome_digest": provider_outcome_result[
            "provider_outcome_digest"
        ],
        "dev_evaluator_enabled": True,
        "openrouter_context": {
            "key_ref": openrouter_key_ref,
            "miner_hotkey": "miner-hotkey",
            "privacy_proof_doc": privacy_proof_doc,
            "privacy_receipt_hash": guard_graph["root_receipt_hash"],
            "runtime_credential_value_hash": runtime_credential_hash,
            "management_credential_value_hash": management_credential_hash,
        },
        "openrouter_guard_evidence": {
            "result": guard_result,
            "receipt_graph": guard_graph,
            "root_receipt_hash": guard_graph["root_receipt_hash"],
            "queue_event_hash": queue_event_hash,
        },
        "expected_event_state_hash": "sha256:" + "3" * 64,
    }


def _stale_parent_payload(tmp_path: Path):
    source_bundle, artifact = _source_and_artifact(tmp_path)
    draft = CodeEditDraft(
        failure_mode="stale implementation",
        mechanism="update measured source",
        expected_improvement="preserve behavior",
        risk="low",
        lane="stale_parent_rebase",
        target_files=("gateway/research_lab/runtime.py",),
        unified_diff=(
            "diff --git a/gateway/research_lab/runtime.py "
            "b/gateway/research_lab/runtime.py\n"
            "--- a/gateway/research_lab/runtime.py\n"
            "+++ b/gateway/research_lab/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="measured stale-parent patch",
        test_plan="run tests",
        rollback_plan="discard candidate",
    )
    return {
        "schema_version": STALE_PARENT_REPAIR_REQUEST_SCHEMA_VERSION,
        "run_id": "run-stale-v2",
        "candidate_id": "candidate:" + "1" * 64,
        "active_artifact": artifact,
        "source_bundle": source_bundle,
        "original_draft": draft.to_dict(),
        "original_source_diff_hash": sha256_json(
            {"unified_diff": draft.unified_diff}
        ),
    }, draft


def _component_registry_graph(
    component_result,
    *,
    parent_graph=None,
    purpose=COMPONENT_REGISTRY_EVIDENCE_PURPOSE_V2,
):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="1" * 32,
            signing_pubkey=public_key,
            transport_pubkey="2" * 64,
            transport_certificate_hash="sha256:" + "3" * 64,
            attestation_user_data_hash="sha256:" + "4" * 64,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode("ascii"),
    )
    parent_receipt_hashes = (
        (str(parent_graph["root_receipt_hash"]),)
        if parent_graph is not None
        else ()
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose=purpose,
            job_id="model-metadata",
            epoch_id=1,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "5" * 64,
            output_root=sha256_bytes(canonical_json(component_result).encode("utf-8")),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=parent_receipt_hashes,
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    parent_boot_identities = (
        tuple(parent_graph["boot_identities"])
        if parent_graph is not None
        else ()
    )
    parent_receipts = (
        tuple(parent_graph["receipts"])
        if parent_graph is not None
        else ()
    )
    parent_transport_attempts = (
        tuple(parent_graph["transport_attempts"])
        if parent_graph is not None
        else ()
    )
    parent_host_operations = (
        tuple(parent_graph["host_operations"])
        if parent_graph is not None
        else ()
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(*parent_boot_identities, boot),
        receipts=(*parent_receipts, receipt),
        transport_attempts=parent_transport_attempts,
        host_operations=parent_host_operations,
    )


def _openrouter_guard_graph(guard_result):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_autoresearch",
            physical_role="gateway_autoresearch",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="3" * 32,
            signing_pubkey=public_key,
            transport_pubkey="4" * 64,
            transport_certificate_hash="sha256:" + "5" * 64,
            attestation_user_data_hash="sha256:" + "6" * 64,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode("ascii"),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_autoresearch",
            purpose="research_lab.openrouter_guard.v2",
            job_id="openrouter-guard",
            epoch_id=1,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "7" * 64,
            output_root=sha256_json(guard_result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
        host_operations=(),
    )


def _provider_catalog_graph(catalog_result):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_coordinator",
            physical_role="gateway_coordinator",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="4" * 32,
            signing_pubkey=public_key,
            transport_pubkey="5" * 64,
            transport_certificate_hash="sha256:" + "6" * 64,
            attestation_user_data_hash="sha256:" + "7" * 64,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(
            "ascii"
        ),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_coordinator",
            purpose="research_lab.source_add_catalog_snapshot.v2",
            job_id="source-add-catalog",
            epoch_id=1,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "8" * 64,
            output_root=sha256_json(catalog_result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
        host_operations=(),
    )


def _active_model_graph(active_model_result):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_coordinator",
            physical_role="gateway_coordinator",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="6" * 32,
            signing_pubkey=public_key,
            transport_pubkey="7" * 64,
            transport_certificate_hash="sha256:" + "8" * 64,
            attestation_user_data_hash="sha256:" + "9" * 64,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(
            "ascii"
        ),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_coordinator",
            purpose="research_lab.active_private_model.v2",
            job_id="active-private-model",
            epoch_id=1,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "a" * 64,
            output_root=sha256_json(active_model_result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
        host_operations=(),
    )


def _provider_outcome_graph(provider_outcome_result):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_coordinator",
            physical_role="gateway_coordinator",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="8" * 32,
            signing_pubkey=public_key,
            transport_pubkey="9" * 64,
            transport_certificate_hash="sha256:" + "a" * 64,
            attestation_user_data_hash="sha256:" + "b" * 64,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(
            "ascii"
        ),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_coordinator",
            purpose="research_lab.provider_outcome_snapshot.v2",
            job_id="provider-outcome-snapshot",
            epoch_id=1,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "c" * 64,
            output_root=sha256_json(provider_outcome_result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
        host_operations=(),
    )


def _parent_receipt_hashes(payload):
    return (
        payload["openrouter_guard_evidence"]["root_receipt_hash"],
        payload["component_registry_evidence"]["root_receipt_hash"],
        payload["active_model_evidence"]["root_receipt_hash"],
        payload["provider_catalog_evidence"]["root_receipt_hash"],
        payload["provider_outcome_evidence"]["root_receipt_hash"],
    )


def _config():
    return replace(
        ResearchLabGatewayConfig.from_env(),
        private_test_cmd="true",
        private_build_cmd="true",
        private_artifact_manifest_output="artifact.json",
    )


def _artifact_seal(*, plaintext, job_id, purpose, artifact_kind):
    plaintext_hash = sha256_bytes(bytes(plaintext))
    descriptor = {
        "status": "sealed",
        "job_id": job_id,
        "purpose": purpose,
        "artifact_kind": artifact_kind,
        "artifact_id": sha256_json(
            {
                "job_id": job_id,
                "purpose": purpose,
                "artifact_kind": artifact_kind,
                "plaintext_hash": plaintext_hash,
            }
        ),
        "plaintext_hash": plaintext_hash,
        "ciphertext_hash": sha256_json(
            {"ciphertext_for": plaintext_hash}
        ),
        "encryption_context_hash": sha256_json(
            {"context_for": plaintext_hash}
        ),
    }
    _artifact_seal.records.append((bytes(plaintext), dict(descriptor)))
    return descriptor


_artifact_seal.records = []


@pytest.mark.parametrize(
    ("failure_code", "expected_type"),
    (
        ("candidate_patch_apply_failed", CodeEditPatchApplyError),
        ("candidate_patch_empty_or_noop", CodeEditEmptyOrNoopPatchError),
        ("candidate_patch_test_failed", CodeEditPrivateTestError),
        ("candidate_image_build_failed", CodeEditImageBuildError),
        ("candidate_artifact_missing", CodeEditArtifactMissingError),
        ("candidate_build_infra_failed", CodeEditInfraFailureError),
        ("candidate_build_failed", CodeEditBuildError),
    ),
)
def test_measured_executor_restores_allowlisted_host_build_failure_type(
    failure_code,
    expected_type,
):
    error = HostOperationV2Error(
        "generic host failure",
        terminal={"failure_code": failure_code},
    )
    with pytest.raises(expected_type) as caught:
        _raise_code_edit_host_operation_failure(error)
    assert caught.value.failure_stage == failure_code
    assert "generic host failure" not in str(caught.value)


def test_measured_executor_preserves_unknown_host_build_failure():
    error = HostOperationV2Error(
        "host operation failed: unknown_stage",
        terminal={"failure_code": "unknown_stage"},
    )
    with pytest.raises(HostOperationV2Error) as caught:
        _raise_code_edit_host_operation_failure(error)
    assert caught.value is error


def test_host_candidate_builder_requeues_signed_infrastructure_failure(tmp_path):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    config = _config()
    source_context = CodeEditCandidateBuilder(
        config
    ).prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host-context",
    )
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )

    class FailedBuildContext:
        @staticmethod
        def execute_host_operation(**kwargs):
            assert kwargs["operation"] == "autoresearch_build_candidate"
            raise HostOperationV2Error(
                "host operation failed",
                terminal={"failure_code": "candidate_build_infra_failed"},
            )

    builder = _HostCandidateBuilder(
        config=config,
        source_context=source_context,
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=FailedBuildContext(),
    )

    with pytest.raises(CodeEditInfraFailureError) as caught:
        builder.build(
            draft=draft,
            parent_artifact=root_artifact,
            run_id="run-host-build-failure-v2",
            candidate_index=1,
            source_context=source_context,
        )
    assert caught.value.retryable is True
    assert caught.value.failure_stage == "candidate_build_infra_failed"


def test_host_candidate_builder_materializes_source_add_derived_artifacts(
    tmp_path,
    monkeypatch,
):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    config = _config()
    source_context = CodeEditCandidateBuilder(
        config
    ).prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host-context",
    )
    draft = CodeEditDraft(
        failure_mode="missing source route",
        mechanism="register the measured source route",
        expected_improvement="make the source available to the model",
        risk="bounded provider integration",
        lane="query_construction",
        target_files=("sourcing_model/routing/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/routing/runtime.py "
            "b/sourcing_model/routing/runtime.py\n"
            "--- a/sourcing_model/routing/runtime.py\n"
            "+++ b/sourcing_model/routing/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="register one measured source route",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    builder = _HostCandidateBuilder(
        config=config,
        source_context=source_context,
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=object(),
    )
    observed = {}

    def materialize(*, draft, source_context):
        observed["draft"] = draft
        observed["source_context"] = source_context
        return draft

    monkeypatch.setattr(
        builder._local,
        "materialize_source_add_derived_artifacts",
        materialize,
    )

    result = builder.materialize_source_add_derived_artifacts(
        draft=draft,
        source_context=source_context,
    )

    assert result is draft
    assert observed == {
        "draft": draft,
        "source_context": source_context,
    }


def test_autoresearch_executor_runs_existing_engine_and_commits_events(tmp_path):
    _FakeEngine.instances.clear()
    _artifact_seal.records.clear()
    channel = _HostChannel()
    payload = _payload(tmp_path)
    context = ExecutionContextV2(
        job_id="autoresearch-v2:test",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=1,
        parent_receipt_hashes=_parent_receipt_hashes(payload),
        provider_credential_ref_hashes={
            "openrouter": "sha256:" + "5" * 64,
            "openrouter_management": "sha256:" + "6" * 64,
        },
        host_operation_channel=channel,
        allowed_purposes=frozenset(ROLE_PURPOSES["gateway_autoresearch"]),
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider must not be called"),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        result = asyncio.run(
            executor(OP_RUN_CODE_EDIT_LOOP, payload, context)
        )
    finally:
        executor.close()

    assert result.output["schema_version"] == "leadpoet.autoresearch_result.v2"
    assert result.output["iterations_completed"] == 1
    assert result.output["selected_candidates"] == []
    assert result.output["tree_result"]["status"] == "failed"
    assert [record["payload"]["event"]["event_type"] for record in channel.records] == [
        "loop_started",
        "loop_failed",
    ]
    assert len(context.stage_receipts) == 1
    assert context.stage_receipts[0].purpose == "research_lab.candidate_decision.v2"
    assert _FakeEngine.instances[0].run_kwargs["run_id"] == "run-v2-1"
    assert _FakeEngine.instances[0].kwargs["builder"].__class__.__name__ == "_HostCandidateBuilder"
    provider_entries, provider_capabilities = _FakeEngine.instances[0].kwargs[
        "provider_registry_loader"
    ]()
    assert provider_entries
    assert provider_capabilities.source_add_provider_count == 0
    assert _FakeEngine.instances[0].kwargs["provider_probe_catalog_loader"]()
    assert len(_artifact_seal.records) == 2
    assert _artifact_seal.records[0][1]["plaintext_hash"] == payload[
        "source_bundle"
    ]["archive_sha256"]
    assert _artifact_seal.records[1][0] == canonical_json(result.output).encode(
        "utf-8"
    )


def test_autoresearch_executor_verifies_input_evidence_with_its_source_role(
    tmp_path,
):
    _FakeEngine.instances.clear()
    _artifact_seal.records.clear()
    payload = _payload(tmp_path)
    component_result = payload["component_registry_evidence"]["result"]
    payload["component_registry_evidence"]["receipt_graph"] = (
        _component_registry_graph(
            component_result,
            parent_graph=payload["active_model_evidence"]["receipt_graph"],
        )
    )
    payload["component_registry_evidence"]["root_receipt_hash"] = payload[
        "component_registry_evidence"
    ]["receipt_graph"]["root_receipt_hash"]
    context = ExecutionContextV2(
        job_id="autoresearch-v2:source-role-verification",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=1,
        parent_receipt_hashes=_parent_receipt_hashes(payload),
        provider_credential_ref_hashes={
            "openrouter": "sha256:" + "5" * 64,
            "openrouter_management": "sha256:" + "6" * 64,
        },
        host_operation_channel=_HostChannel(),
        allowed_purposes=frozenset(ROLE_PURPOSES["gateway_autoresearch"]),
    )
    verified_roles = []

    def verify_scoring(identity):
        assert identity["physical_role"] == "gateway_scoring"
        verified_roles.append("gateway_scoring")

    def verify_coordinator(identity):
        assert identity["physical_role"] == "gateway_coordinator"
        verified_roles.append("gateway_coordinator")

    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider must not be called"),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        scoring_graph_verifier=verify_scoring,
        coordinator_boot_verifier=verify_coordinator,
        artifact_seal=_artifact_seal,
    )
    try:
        result = asyncio.run(executor(OP_RUN_CODE_EDIT_LOOP, payload, context))
    finally:
        executor.close()

    assert result.output["status"] == "failed"
    assert verified_roles == [
        "gateway_coordinator",
        "gateway_scoring",
        "gateway_coordinator",
        "gateway_coordinator",
        "gateway_coordinator",
    ]


def test_autoresearch_executor_rejects_registry_not_derived_from_measured_metadata(
    tmp_path,
):
    payload = _payload(tmp_path)
    payload["component_registry"]["entries"][0]["token_budget"] += 1
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider must not be called"),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="component registry differs from measured model metadata",
        ):
            executor._validate_request(payload)
    finally:
        executor.close()


def test_autoresearch_executor_rejects_obsolete_component_registry_purpose(
    tmp_path,
):
    payload = _payload(tmp_path)
    component_evidence = payload["component_registry_evidence"]
    component_evidence["receipt_graph"] = _component_registry_graph(
        component_evidence["result"],
        purpose="research_lab.private_model_run.v2",
    )
    component_evidence["root_receipt_hash"] = component_evidence[
        "receipt_graph"
    ]["root_receipt_hash"]
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider must not be called"),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            ValueError,
            match="missing required purpose research_lab.model_compatibility.v2",
        ):
            executor._validate_request(payload)
    finally:
        executor.close()


def test_autoresearch_executor_rejects_unsupported_component_ancestry_role(
    tmp_path,
):
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        scoring_graph_verifier=lambda _identity: None,
        coordinator_boot_verifier=lambda _identity: None,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="component registry ancestry contains an unsupported role",
        ):
            executor._verify_component_registry_boot(
                {"physical_role": "gateway_autoresearch"}
            )
    finally:
        executor.close()


def test_autoresearch_executor_requires_component_ancestry_role_verifier(
    tmp_path,
):
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        scoring_graph_verifier=lambda _identity: None,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="component registry ancestry verifier is unavailable",
        ):
            executor._verify_component_registry_boot(
                {"physical_role": "gateway_coordinator"}
            )
    finally:
        executor.close()


def test_autoresearch_executor_accepts_conservative_interrupted_evaluation_recovery(
    tmp_path,
):
    payload = _payload(tmp_path)
    policy = payload["budget_context"]["tree_policy"]
    operation_id = "sha256:" + "f" * 64
    policy.update(
        {
            "evaluation_provider_call_budget_charge": 32,
            "evaluation_cost_budget_charge_microusd": 500_000,
            "unsettled_evaluation_operation_count": 1,
            "unsettled_evaluation_operations_hash": sha256_json([operation_id]),
        }
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        validated = executor._validate_request(payload)
    finally:
        executor.close()

    assert validated["budget_context"]["tree_policy"][
        "evaluation_provider_call_budget_charge"
    ] == 32


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {
                "unsettled_evaluation_operation_count": 1,
                "unsettled_evaluation_operations_hash": sha256_json(
                    ["sha256:" + "f" * 64]
                ),
            },
            "did not exhaust live budget",
        ),
        (
            {
                "evaluation_provider_call_budget_charge": 32,
                "evaluation_cost_budget_charge_microusd": 500_000,
                "unsettled_evaluation_operation_count": 1,
                "unsettled_evaluation_operations_hash": sha256_json([]),
            },
            "counts differ from commitments",
        ),
    ),
)
def test_autoresearch_executor_rejects_unsafe_interrupted_evaluation_recovery(
    tmp_path,
    updates,
    message,
):
    payload = _payload(tmp_path)
    payload["budget_context"]["tree_policy"].update(updates)
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(AutoresearchExecutorV2Error, match=message):
            executor._validate_request(payload)
    finally:
        executor.close()


def test_autoresearch_executor_binds_host_bridge_to_replacement_tree(tmp_path):
    _FakeEngine.instances.clear()
    _artifact_seal.records.clear()
    channel = _HostChannel()
    payload = _payload(tmp_path)
    policy = TreePolicy.from_mapping(
        payload["budget_context"]["tree_policy"]["policy"]
    )
    replacement = TreeReplacement(
        generation=1,
        replaces_tree_id="sha256:" + "1" * 64,
        cancellation_event_hash="sha256:" + "2" * 64,
        prior_root_artifact_hash="sha256:" + "3" * 64,
        prior_root_manifest_hash="sha256:" + "4" * 64,
        prior_policy_hash=policy.policy_hash,
        root_artifact_hash=payload["artifact"]["model_artifact_hash"],
        root_manifest_hash=payload["artifact"]["manifest_hash"],
        policy_hash=policy.policy_hash,
    )
    payload["budget_context"]["tree_replacement"] = replacement.to_dict()
    expected_tree_id = derive_tree_id(
        run_id=payload["run_id"],
        root_artifact_hash=payload["artifact"]["model_artifact_hash"],
        policy=policy,
        replacement=replacement,
    )
    context = ExecutionContextV2(
        job_id="autoresearch-v2:replacement",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=1,
        parent_receipt_hashes=_parent_receipt_hashes(payload),
        provider_credential_ref_hashes={
            "openrouter": "sha256:" + "5" * 64,
            "openrouter_management": "sha256:" + "6" * 64,
        },
        host_operation_channel=channel,
        allowed_purposes=frozenset(ROLE_PURPOSES["gateway_autoresearch"]),
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider must not be called"),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        result = asyncio.run(executor(OP_RUN_CODE_EDIT_LOOP, payload, context))
    finally:
        executor.close()

    engine = _FakeEngine.instances[0]
    assert engine.kwargs["tree_repository"]._tree_id == expected_tree_id
    assert result.output["tree_result"]["tree_id"] == expected_tree_id
    assert (
        engine.run_kwargs["budget_context"]["tree_replacement"]
        == replacement.to_dict()
    )


def test_autoresearch_executor_rejects_replacement_for_another_root(tmp_path):
    payload = _payload(tmp_path)
    policy = TreePolicy.from_mapping(
        payload["budget_context"]["tree_policy"]["policy"]
    )
    payload["budget_context"]["tree_replacement"] = TreeReplacement(
        generation=1,
        replaces_tree_id="sha256:" + "1" * 64,
        cancellation_event_hash="sha256:" + "2" * 64,
        prior_root_artifact_hash="sha256:" + "3" * 64,
        prior_root_manifest_hash="sha256:" + "4" * 64,
        prior_policy_hash=policy.policy_hash,
        root_artifact_hash="sha256:" + "5" * 64,
        root_manifest_hash=payload["artifact"]["manifest_hash"],
        policy_hash=policy.policy_hash,
    ).to_dict()
    context = ExecutionContextV2(
        job_id="autoresearch-v2:wrong-replacement-root",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=1,
        parent_receipt_hashes=_parent_receipt_hashes(payload),
        provider_credential_ref_hashes={
            "openrouter": "sha256:" + "5" * 64,
            "openrouter_management": "sha256:" + "6" * 64,
        },
        host_operation_channel=_HostChannel(),
        allowed_purposes=frozenset(ROLE_PURPOSES["gateway_autoresearch"]),
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider must not be called"),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="replacement authority differs",
        ):
            asyncio.run(executor(OP_RUN_CODE_EDIT_LOOP, payload, context))
    finally:
        executor.close()


def test_host_git_tree_operation_commitment_is_strictly_validated():
    tree_id = "sha256:" + "1" * 64
    observed = {}

    class Context:
        @staticmethod
        def execute_host_operation(**kwargs):
            observed.update(kwargs)
            response = {
                "schema_version": HOST_GIT_TREE_RESULT_SCHEMA_VERSION,
                "action": "operation_settlement_commitment",
                "state_hash": kwargs["expected_state_hash"],
                "result": {
                    "tree_id": tree_id,
                    "action": "operation_settlement_commitment",
                    "operation_count": 3,
                    "settled_cost_microusd": 12_345,
                    "provider_call_count": 2,
                    "operation_settlement_hash": "sha256:" + "2" * 64,
                },
            }
            return kwargs["response_validator"](response)

    result = _HostGitTreeRepository(
        Context(),
        tree_id=tree_id,
        child_source_verifier=lambda **_kwargs: None,
    ).operation_settlement_commitment()

    assert observed["operation"] == HOST_GIT_TREE
    assert observed["payload"] == {
        "action": "operation_settlement_commitment",
        "tree_id": tree_id,
    }
    assert result["operation_count"] == 3
    assert result["settled_cost_microusd"] == 12_345
    assert result["provider_call_count"] == 2


def _host_git_tree_commit_context(*, slot_index_value=0):
    class Context:
        @staticmethod
        def execute_host_operation(**kwargs):
            payload = kwargs["payload"]
            draft = payload["draft"]
            incremental_patch = str(draft["unified_diff"])
            response = {
                "schema_version": HOST_GIT_TREE_RESULT_SCHEMA_VERSION,
                "action": "commit_child",
                "state_hash": kwargs["expected_state_hash"],
                "result": {
                    "schema_version": HOST_GIT_TREE_COMMIT_SCHEMA_VERSION,
                    "tree_id": payload["tree_id"],
                    "node_id": payload["slot"]["node_id"],
                    "parent_node_id": payload["slot"]["parent_node_id"],
                    "root_branch_id": payload["slot"]["root_branch_id"],
                    "depth": payload["slot"]["depth"],
                    "slot_index": slot_index_value,
                    "git_commit": "1" * 64,
                    "parent_git_commit": "2" * 64,
                    "source_tree_hash": "sha256:" + "3" * 64,
                    "draft_patch_hash": sha256_json(
                        {"unified_diff": draft["unified_diff"]}
                    ),
                    "incremental_patch_hash": sha256_json(
                        {"unified_diff": incremental_patch}
                    ),
                    "cumulative_patch_hash": sha256_json(
                        {"unified_diff": incremental_patch}
                    ),
                    "changed_files": list(draft["target_files"]),
                    "incremental_patch": incremental_patch,
                    "cumulative_patch": incremental_patch,
                },
            }
            return kwargs["response_validator"](response)

    return Context()


def test_host_git_tree_commit_accepts_valid_zero_slot_index():
    tree_id = "sha256:" + "4" * 64
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase runtime value",
        expected_improvement="recover valid companies",
        risk="bounded change",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase one bounded runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )

    commit = _HostGitTreeRepository(
        _host_git_tree_commit_context(),
        tree_id=tree_id,
        child_source_verifier=lambda **_kwargs: None,
    ).commit_child(
        slot=slot,
        draft=draft,
        expected_parent_source_tree_hash="sha256:" + "5" * 64,
    )

    assert commit["slot_index"] == 0
    assert commit["node_id"] == slot.node_id


@pytest.mark.parametrize("invalid_slot_index", [False, "0", None, -1])
def test_host_git_tree_commit_rejects_noncanonical_slot_index(
    invalid_slot_index,
):
    tree_id = "sha256:" + "4" * 64
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase runtime value",
        expected_improvement="recover valid companies",
        risk="bounded change",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase one bounded runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="Git-tree commit result topology differs",
    ):
        _HostGitTreeRepository(
            _host_git_tree_commit_context(
                slot_index_value=invalid_slot_index
            ),
            tree_id=tree_id,
            child_source_verifier=lambda **_kwargs: None,
        ).commit_child(
            slot=slot,
            draft=draft,
            expected_parent_source_tree_hash="sha256:" + "5" * 64,
        )


def test_v2_builder_restores_git_tree_parent_from_cumulative_patch(tmp_path):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    source_root = tmp_path / "private-source"
    child_root = tmp_path / "child-source"
    shutil.copytree(source_root, child_root)
    child_file = child_root / "sourcing_model" / "runtime.py"
    child_file.write_text("VALUE = 2\n", encoding="utf-8")
    child_artifact = PrivateModelArtifactManifest.from_mapping(
        build_private_artifact_with_adapted_source_admission(
            source_path=child_root,
            git_commit_sha="c" * 40,
            image_digest=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
                + "d" * 64
            ),
            manifest_uri="s3://private/manifests/child.json",
            signature_ref="kms:child-signature",
            component_registry_version="1",
            scoring_adapter_version="1",
        )
    )
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    candidate = BuiltCodeEditCandidate(
        draft=draft,
        build=CodeEditBuildResult(
            candidate_model_manifest=child_artifact,
            code_edit_manifest={
                "parent_artifact_hash": root_artifact.model_artifact_hash
            },
            source_diff_hash=source_diff_hash,
            build_doc={},
        ),
        node_id="tree-node:" + "1" * 64,
        iteration=1,
        tree_id=derive_tree_id(
            run_id="run-restored-v2",
            root_artifact_hash=root_artifact.model_artifact_hash,
            policy=TreePolicy(mode="active"),
        ),
        tree_parent_node_id="root",
        tree_root_branch_id="tree-node:" + "1" * 64,
        tree_depth=1,
        tree_branch_objective_path_id="bounded-query-path",
        tree_branch_objective_hash="sha256:" + "3" * 64,
        tree_generation_attempt_count=2,
        tree_git_commit="2" * 64,
        tree_root_artifact_hash=root_artifact.model_artifact_hash,
        tree_parent_artifact_hash=root_artifact.model_artifact_hash,
        tree_incremental_source_diff_hash=source_diff_hash,
        tree_cumulative_source_diff_hash=source_diff_hash,
    )
    candidate_doc = _candidate_document(candidate)
    assert candidate_doc["tree_branch_objective_path_id"] == "bounded-query-path"
    assert candidate_doc["tree_branch_objective_hash"] == "sha256:" + "3" * 64
    assert candidate_doc["tree_generation_attempt_count"] == 2
    config = _config()
    builder = _HostCandidateBuilder(
        config=config,
        source_context=_source_context(
            source_root=source_root,
            artifact=root_artifact,
            config=config,
        ),
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=object(),
    )

    restored = builder.restore_rehydrated_candidate_source_context(
        candidate=candidate
    )

    assert restored.source_tree_hash == child_artifact.model_artifact_hash
    assert (restored.source_root / "sourcing_model" / "runtime.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert builder.prepare_parent_source_context(
        parent_artifact=child_artifact,
        workspace_dir=tmp_path / "unused",
    ) is restored

    tampered_builder = _HostCandidateBuilder(
        config=config,
        source_context=_source_context(
            source_root=source_root,
            artifact=root_artifact,
            config=config,
        ),
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=object(),
    )
    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="rehydrated Git-tree candidate commitment differs",
    ):
        tampered_builder.restore_rehydrated_candidate_source_context(
            candidate=replace(
                candidate,
                tree_cumulative_source_diff_hash="sha256:" + "0" * 64,
            )
        )

    structural_diff = draft.unified_diff.replace(
        "--- a/sourcing_model/runtime.py\n",
        "old mode 100644\nnew mode 100755\n"
        "--- a/sourcing_model/runtime.py\n",
    )
    structural_hash = sha256_json({"unified_diff": structural_diff})
    structural_candidate = replace(
        candidate,
        draft=replace(draft, unified_diff=structural_diff),
        build=replace(candidate.build, source_diff_hash=structural_hash),
        tree_incremental_source_diff_hash=structural_hash,
        tree_cumulative_source_diff_hash=structural_hash,
    )
    structural_builder = _HostCandidateBuilder(
        config=config,
        source_context=_source_context(
            source_root=source_root,
            artifact=root_artifact,
            config=config,
        ),
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=object(),
    )
    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="not a content-only Git patch",
    ):
        structural_builder.restore_rehydrated_candidate_source_context(
            candidate=structural_candidate
        )


def test_signed_source_normalization_matches_tree_and_build_hashes(tmp_path):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    config = _config()
    builder = CodeEditCandidateBuilder(config)
    host_context = builder.prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host",
    )

    measured_root = tmp_path / "measured"
    extract_source_bundle_v2(
        source_bundle,
        destination=measured_root,
        expected_source_tree_hash=root_artifact.model_artifact_hash,
    )
    _write_research_lab_build_scaffold(
        measured_root,
        base_image_ref=root_artifact.image_digest,
    )
    measured_context = _source_context(
        source_root=measured_root,
        artifact=root_artifact,
        config=config,
    )

    assert host_context.source_tree_hash == measured_context.source_tree_hash
    assert host_context.source_tree_hash != root_artifact.model_artifact_hash

    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-normalized-v2",
        root_artifact_hash=root_artifact.model_artifact_hash,
        policy=policy,
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=host_context.source_root,
        root_artifact_hash=root_artifact.model_artifact_hash,
        policy_hash=policy.policy_hash,
    )
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    tree_commit = repository.commit_child(
        slot=slot,
        draft=draft,
        expected_parent_source_tree_hash=host_context.source_tree_hash,
    )

    build_root = tmp_path / "build"
    observed_parent_hash, _ = _prepare_parent_image_workspace(
        image_digest=root_artifact.image_digest,
        repo_dir=build_root,
        timeout_seconds=30,
        source_context=host_context,
    )
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(draft.unified_diff, encoding="utf-8")
    _run_git_apply(diff_path, cwd=build_root, timeout_seconds=30, check=True)
    _run_git_apply(diff_path, cwd=build_root, timeout_seconds=30, check=False)

    assert observed_parent_hash == host_context.source_tree_hash
    assert compute_private_source_tree_hash(build_root) == tree_commit.source_tree_hash


@pytest.mark.parametrize(
    "structural_metadata",
    (
        "old mode 100644\nnew mode 100755\n",
        (
            "rename from sourcing_model/runtime.py\n"
            "rename to sourcing_model/runtime.py\n"
        ),
        "GIT binary patch\n",
    ),
)
def test_measured_host_rejects_structural_git_patch_metadata(
    tmp_path,
    structural_metadata,
):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    config = _config()
    source_context = CodeEditCandidateBuilder(
        config
    ).prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host-context",
    )
    patch = (
        "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
        f"{structural_metadata}"
        "--- a/sourcing_model/runtime.py\n"
        "+++ b/sourcing_model/runtime.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )

    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="is not a content-only Git patch",
    ):
        _HostCandidateBuilder._apply_measured_patch(
            context=source_context,
            unified_diff=patch,
            label="Git-tree canonical incremental patch",
        )


def test_measured_host_accepts_content_patch_from_read_only_source(tmp_path):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    source_context = CodeEditCandidateBuilder(
        _config()
    ).prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host-context",
    )
    target = source_context.source_root / "sourcing_model/runtime.py"
    target.chmod(0o444)
    patch = (
        "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
        "--- a/sourcing_model/runtime.py\n"
        "+++ b/sourcing_model/runtime.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )

    source_hash, paths = _HostCandidateBuilder._apply_measured_patch(
        context=source_context,
        unified_diff=patch,
        label="Git-tree generated draft",
    )

    assert source_hash.startswith("sha256:")
    assert paths == frozenset({"sourcing_model/runtime.py"})


def test_measured_host_rejects_content_patch_that_gains_executable_bit(
    tmp_path,
    monkeypatch,
):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    source_context = CodeEditCandidateBuilder(
        _config()
    ).prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host-context",
    )
    patch = (
        "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
        "--- a/sourcing_model/runtime.py\n"
        "+++ b/sourcing_model/runtime.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    real_apply = _run_git_apply

    def _apply_and_change_mode(*args, **kwargs):
        result = real_apply(*args, **kwargs)
        if not kwargs.get("check"):
            target = Path(kwargs["cwd"]) / "sourcing_model/runtime.py"
            target.chmod(target.stat().st_mode | 0o100)
        return result

    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2._run_git_apply",
        _apply_and_change_mode,
    )

    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="changed source file type or mode",
    ):
        _HostCandidateBuilder._apply_measured_patch(
            context=source_context,
            unified_diff=patch,
            label="Git-tree generated draft",
        )


def test_host_git_tree_rejects_semantically_substituted_cumulative_patch(
    tmp_path,
):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    config = _config()
    local_builder = CodeEditCandidateBuilder(config)
    host_context = local_builder.prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host-context",
    )
    measured_builder = _HostCandidateBuilder(
        config=config,
        source_context=host_context,
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=object(),
        root_artifact_hash=root_artifact.model_artifact_hash,
    )
    tree_id = derive_tree_id(
        run_id="run-measured-git-substitution-v2",
        root_artifact_hash=root_artifact.model_artifact_hash,
        policy=TreePolicy(mode="active"),
    )
    repository = GitTreeRepository(
        workspace=tmp_path / "tree-substitution",
        tree_id=tree_id,
    )
    repository.initialize(
        source_root=host_context.source_root,
        root_artifact_hash=root_artifact.model_artifact_hash,
        policy_hash=TreePolicy(mode="active").policy_hash,
    )
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    genuine = repository.commit_child(
        slot=slot,
        draft=draft,
        expected_parent_source_tree_hash=host_context.source_tree_hash,
    )
    measured_builder.verify_git_tree_child_semantics(
        draft=draft,
        canonical_incremental_patch=genuine.incremental_patch,
        cumulative_patch=genuine.cumulative_patch,
        changed_files=genuine.changed_files,
        expected_parent_source_tree_hash=host_context.source_tree_hash,
        expected_child_source_tree_hash=genuine.source_tree_hash,
    )
    substituted_cumulative = (
        "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
        "--- a/sourcing_model/runtime.py\n"
        "+++ b/sourcing_model/runtime.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 999\n"
    )

    class Context:
        @staticmethod
        def execute_host_operation(**kwargs):
            result = {
                **genuine.to_dict(),
                "incremental_patch": genuine.incremental_patch,
                "cumulative_patch": substituted_cumulative,
                "cumulative_patch_hash": sha256_json(
                    {"unified_diff": substituted_cumulative}
                ),
            }
            return kwargs["response_validator"](
                {
                    "schema_version": HOST_GIT_TREE_RESULT_SCHEMA_VERSION,
                    "action": "commit_child",
                    "state_hash": kwargs["expected_state_hash"],
                    "result": result,
                }
            )

    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="Git-tree child patches are not measured-source equivalent",
    ):
        _HostGitTreeRepository(
            Context(),
            tree_id=tree_id,
            child_source_verifier=(
                measured_builder.verify_git_tree_child_semantics
            ),
        ).commit_child(
            slot=slot,
            draft=draft,
            expected_parent_source_tree_hash=host_context.source_tree_hash,
        )


def test_measured_host_rejects_incremental_patch_with_undeclared_extra_path(
    tmp_path,
):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    config = _config()
    local_builder = CodeEditCandidateBuilder(config)
    host_context = local_builder.prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "host-context",
    )
    measured_builder = _HostCandidateBuilder(
        config=config,
        source_context=host_context,
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=object(),
        root_artifact_hash=root_artifact.model_artifact_hash,
    )
    draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    canonical_with_extra_path = draft.unified_diff + (
        "diff --git a/gateway/research_lab/runtime.py "
        "b/gateway/research_lab/runtime.py\n"
        "--- a/gateway/research_lab/runtime.py\n"
        "+++ b/gateway/research_lab/runtime.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )

    with pytest.raises(
        AutoresearchExecutorV2Error,
        match="Git-tree incremental changed-file set differs",
    ):
        measured_builder.verify_git_tree_child_semantics(
            draft=draft,
            canonical_incremental_patch=canonical_with_extra_path,
            cumulative_patch=canonical_with_extra_path,
            changed_files=draft.target_files,
            expected_parent_source_tree_hash=host_context.source_tree_hash,
            expected_child_source_tree_hash="sha256:" + "f" * 64,
        )


def test_host_git_tree_accepts_depth_two_incremental_and_cumulative_paths(
    tmp_path,
):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    config = _config()
    local_builder = CodeEditCandidateBuilder(config)
    root_context = local_builder.prepare_attested_source_context(
        parent_artifact=root_artifact,
        source_bundle=source_bundle,
        workspace_dir=tmp_path / "root-context",
    )
    measured_builder = _HostCandidateBuilder(
        config=config,
        source_context=root_context,
        source_bundle_hash=source_bundle["archive_sha256"],
        execution_context=object(),
        root_artifact_hash=root_artifact.model_artifact_hash,
    )
    tree_id = derive_tree_id(
        run_id="run-measured-depth-two-v2",
        root_artifact_hash=root_artifact.model_artifact_hash,
        policy=TreePolicy(mode="active"),
    )
    repository = GitTreeRepository(
        workspace=tmp_path / "tree-depth-two",
        tree_id=tree_id,
    )
    repository.initialize(
        source_root=root_context.source_root,
        root_artifact_hash=root_artifact.model_artifact_hash,
        policy_hash=TreePolicy(mode="active").policy_hash,
    )

    first_draft = CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("sourcing_model/runtime.py",),
        unified_diff=(
            "diff --git a/sourcing_model/runtime.py b/sourcing_model/runtime.py\n"
            "--- a/sourcing_model/runtime.py\n"
            "+++ b/sourcing_model/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    first_slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    first = repository.commit_child(
        slot=first_slot,
        draft=first_draft,
        expected_parent_source_tree_hash=root_context.source_tree_hash,
    )

    first_source = tmp_path / "first-child-source"
    _copy_source_tree(root_context.source_root, first_source)
    _initialize_temporary_git_repo(first_source)
    first_diff = tmp_path / "first-child.diff"
    first_diff.write_text(first_draft.unified_diff, encoding="utf-8")
    _run_git_apply(first_diff, cwd=first_source, timeout_seconds=30, check=True)
    _run_git_apply(first_diff, cwd=first_source, timeout_seconds=30, check=False)
    first_artifact = replace(
        root_artifact,
        model_artifact_hash=first.source_tree_hash,
    )
    first_context = _source_context(
        source_root=first_source,
        artifact=first_artifact,
        config=config,
    )
    assert first_context.source_tree_hash == first.source_tree_hash
    measured_builder._source_contexts[first.source_tree_hash] = first_context

    second_draft = CodeEditDraft(
        failure_mode="bounded routing",
        mechanism="increase an independent gateway runtime value",
        expected_improvement="improve source routing",
        risk="bounded runtime increase",
        lane="source_routing",
        target_files=("gateway/research_lab/runtime.py",),
        unified_diff=(
            "diff --git a/gateway/research_lab/runtime.py "
            "b/gateway/research_lab/runtime.py\n"
            "--- a/gateway/research_lab/runtime.py\n"
            "+++ b/gateway/research_lab/runtime.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 3\n"
        ),
        redacted_summary="increase an independent gateway runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    second_slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id=first_slot.node_id,
        root_branch_id=first_slot.root_branch_id,
        depth=2,
        slot_index=0,
    )
    second = repository.commit_child(
        slot=second_slot,
        draft=second_draft,
        expected_parent_source_tree_hash=first.source_tree_hash,
    )

    assert second.changed_files == ("gateway/research_lab/runtime.py",)
    assert "sourcing_model/runtime.py" in second.cumulative_patch
    assert "gateway/research_lab/runtime.py" in second.cumulative_patch
    measured_builder.verify_git_tree_child_semantics(
        draft=second_draft,
        canonical_incremental_patch=second.incremental_patch,
        cumulative_patch=second.cumulative_patch,
        changed_files=second.changed_files,
        expected_parent_source_tree_hash=first.source_tree_hash,
        expected_child_source_tree_hash=second.source_tree_hash,
    )


def test_autoresearch_executor_rejects_tampered_provider_catalog(tmp_path):
    payload = _payload(tmp_path)
    catalog_evidence = payload["provider_catalog_evidence"]
    catalog_evidence["result"] = dict(catalog_evidence["result"])
    catalog_evidence["result"]["runtime_catalog"] = dict(
        catalog_evidence["result"]["runtime_catalog"]
    )
    catalog_evidence["result"]["runtime_catalog"]["catalog_hash"] = (
        "sha256:" + "9" * 64
    )
    context = ExecutionContextV2(
        job_id="autoresearch-v2:tampered-catalog",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=1,
        parent_receipt_hashes=_parent_receipt_hashes(payload),
        provider_credential_ref_hashes={
            "openrouter": "sha256:" + "5" * 64,
            "openrouter_management": "sha256:" + "6" * 64,
        },
        host_operation_channel=_HostChannel(),
        allowed_purposes=frozenset(ROLE_PURPOSES["gateway_autoresearch"]),
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="runtime catalog is invalid",
        ):
            asyncio.run(executor(OP_RUN_CODE_EDIT_LOOP, payload, context))
    finally:
        executor.close()


def test_autoresearch_executor_rejects_source_bundle_not_matching_parent(tmp_path):
    payload = _payload(tmp_path)
    payload["source_bundle"] = dict(payload["source_bundle"])
    payload["source_bundle"]["source_tree_hash"] = "sha256:" + "9" * 64
    context = ExecutionContextV2(
        job_id="autoresearch-v2:test",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=1,
        parent_receipt_hashes=_parent_receipt_hashes(payload),
        provider_credential_ref_hashes={
            "openrouter": "sha256:" + "5" * 64,
            "openrouter_management": "sha256:" + "6" * 64,
        },
        host_operation_channel=_HostChannel(),
        allowed_purposes=frozenset(ROLE_PURPOSES["gateway_autoresearch"]),
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(ValueError, match="declared tree differs"):
            asyncio.run(executor(OP_RUN_CODE_EDIT_LOOP, payload, context))
    finally:
        executor.close()


def test_autoresearch_executor_rejects_tampered_provider_outcome_digest(tmp_path):
    payload = _payload(tmp_path)
    payload["provider_outcome_digest"] = dict(payload["provider_outcome_digest"])
    payload["provider_outcome_digest"]["sidecar_sequence"] = 99
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="differs from measured snapshot",
        ):
            asyncio.run(
                executor(
                    OP_RUN_CODE_EDIT_LOOP,
                    payload,
                    ExecutionContextV2(
                        job_id="autoresearch-v2:tampered-outcome",
                        purpose="research_lab.candidate_decision.v2",
                        epoch_id=1,
                        parent_receipt_hashes=_parent_receipt_hashes(payload),
                    ),
                )
            )
    finally:
        executor.close()


def test_autoresearch_executor_requires_provider_outcome_ancestry(tmp_path):
    payload = _payload(tmp_path)
    parents = _parent_receipt_hashes(payload)[:-1]
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="provider outcome receipt is missing",
        ):
            asyncio.run(
                executor(
                    OP_RUN_CODE_EDIT_LOOP,
                    payload,
                    ExecutionContextV2(
                        job_id="autoresearch-v2:missing-outcome-parent",
                        purpose="research_lab.candidate_decision.v2",
                        epoch_id=1,
                        parent_receipt_hashes=parents,
                    ),
                )
            )
    finally:
        executor.close()


def test_autoresearch_executor_rejects_uncommitted_privacy_context(tmp_path):
    payload = _payload(tmp_path)
    payload["openrouter_context"]["privacy_receipt_hash"] = "missing"
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="privacy receipt hash",
        ):
            asyncio.run(
                executor(
                    OP_RUN_CODE_EDIT_LOOP,
                    payload,
                    ExecutionContextV2(
                        job_id="autoresearch-v2:test",
                        purpose="research_lab.candidate_decision.v2",
                        epoch_id=1,
                    ),
                )
            )
    finally:
        executor.close()


def test_autoresearch_executor_requires_guard_evidence_ancestry(tmp_path):
    payload = _payload(tmp_path)
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="OpenRouter guard receipt is missing",
        ):
            asyncio.run(
                executor(
                    OP_RUN_CODE_EDIT_LOOP,
                    payload,
                    ExecutionContextV2(
                        job_id="autoresearch-v2:missing-guard-parent",
                        purpose="research_lab.candidate_decision.v2",
                        epoch_id=1,
                    ),
                )
            )
    finally:
        executor.close()


@pytest.mark.parametrize("substitution", ["run", "queue_head"])
def test_autoresearch_executor_rejects_cross_run_guard_substitution(
    tmp_path,
    substitution,
):
    payload = _payload(tmp_path)
    if substitution == "run":
        payload["run_id"] = "run-v2-other"
    else:
        payload["openrouter_guard_evidence"]["queue_event_hash"] = (
            "sha256:" + "b" * 64
        )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="guard evidence differs from autoresearch run",
        ):
            asyncio.run(
                executor(
                    OP_RUN_CODE_EDIT_LOOP,
                    payload,
                    ExecutionContextV2(
                        job_id="autoresearch-v2:cross-run-guard",
                        purpose="research_lab.candidate_decision.v2",
                        epoch_id=1,
                        parent_receipt_hashes=_parent_receipt_hashes(payload),
                    ),
                )
            )
    finally:
        executor.close()


def test_openrouter_guard_returns_only_committed_redacted_key_evidence(monkeypatch):
    runtime_hash = "sha256:" + "5" * 64
    management_hash = "sha256:" + "6" * 64
    key_ref = "encrypted_ref:openrouter:" + "1" * 32
    run_id = "run-credential-transition"
    queue_event_hash = "sha256:" + "a" * 64
    run_state_hash = sha256_json(
        {"run_id": run_id, "queue_event_hash": queue_event_hash}
    )
    observed_preflight_kwargs = {}
    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2.preflight_openrouter_key",
        lambda _key, **kwargs: observed_preflight_kwargs.update(kwargs) or {
            "limit_remaining": "0.00",
            "usage": 12,
        },
    )
    observed_privacy_kwargs = {}
    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2.verify_openrouter_workspace_privacy",
        lambda **kwargs: observed_privacy_kwargs.update(kwargs) or {
            "workspace_id_hash": "workspace-hash",
            "runtime_key_hash": runtime_hash.split(":", 1)[1],
            "management_key_hash": "placeholder-hash",
            "proof_hash": "sha256:" + "7" * 64,
        },
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("mock guard makes no request"),
        retry_policy_hashes={
            "openrouter": "sha256:" + "4" * 64,
            "openrouter_management": "sha256:" + "8" * 64,
        },
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        result = asyncio.run(
            executor(
                OP_VERIFY_OPENROUTER_GUARD,
                {
                    "schema_version": OPENROUTER_GUARD_REQUEST_SCHEMA_VERSION,
                    "key_ref": key_ref,
                    "key_ref_hash": sha256_bytes(key_ref.encode("utf-8")),
                    "miner_hotkey_hash": "sha256:" + "9" * 64,
                    "runtime_credential_value_hash": runtime_hash,
                    "management_credential_value_hash": management_hash,
                    "stage": "autoresearch_v2_authority",
                    "request_policy": {
                        "data_collection": "deny",
                        "allow_fallbacks": False,
                    },
                    "run_id": run_id,
                    "queue_event_hash": queue_event_hash,
                    "run_state_hash": run_state_hash,
                },
                ExecutionContextV2(
                    job_id="autoresearch-v2:guard",
                    purpose="research_lab.openrouter_guard.v2",
                    epoch_id=1,
                    allowed_purposes=frozenset(
                        ROLE_PURPOSES["gateway_autoresearch"]
                    ),
                    provider_credential_ref_hashes={
                        "openrouter": runtime_hash,
                        "openrouter_management": management_hash,
                    },
                ),
            )
        )
    finally:
        executor.close()
    assert result.output["credit_depleted"] is True
    assert result.output["management_credential_value_hash"] == management_hash
    assert result.output["run_state_hash"] == run_state_hash
    assert result.output["privacy_proof_doc"]["management_key_hash"] == (
        management_hash.split(":", 1)[1]
    )
    assert observed_privacy_kwargs["expected_runtime_key_hash"] == (
        runtime_hash.split(":", 1)[1]
    )
    assert observed_preflight_kwargs["expected_key_hash"] == (
        runtime_hash.split(":", 1)[1]
    )
    assert "sk-or-v1-" not in str(result.output)


def test_loop_privacy_verifier_binds_placeholder_to_committed_runtime_key_hash(
    monkeypatch,
):
    runtime_hash = "sha256:" + "5" * 64
    management_hash = "sha256:" + "6" * 64
    observed_privacy_kwargs = {}

    def verify_privacy(**kwargs):
        observed_privacy_kwargs.update(kwargs)
        return {
            "workspace_id_hash": "workspace-hash",
            "runtime_key_hash": kwargs["expected_runtime_key_hash"],
            "runtime_key_label_hash": "label-hash",
            "runtime_key_creator_user_id_hash": "creator-hash",
            "management_key_hash": "placeholder-hash",
            "proof_hash": "sha256:" + "7" * 64,
        }

    class FakeWorker:
        def __init__(self, _config, *, worker_ref):
            assert worker_ref == "enclave:autoresearch-v2"

        def _auto_research_max_tokens_for_call(self, **_kwargs):
            return 32

        async def _call_openrouter(self, **kwargs):
            kwargs["privacy_verifier"](
                runtime_key=kwargs["api_key"],
                management_key=kwargs["privacy_management_key"],
                stage=kwargs["capture_stage"],
                request_policy={
                    "data_collection": "deny",
                    "allow_fallbacks": False,
                },
            )
            return SimpleNamespace(
                content="ok",
                provider_usage={},
                cost_microusd=1,
            )

    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2.verify_openrouter_workspace_privacy",
        verify_privacy,
    )
    monkeypatch.setattr(
        "gateway.research_lab.worker.ResearchLabHostedWorker", FakeWorker
    )
    monkeypatch.setattr(
        "gateway.research_lab.worker._resolve_code_edit_loop_stage_model_request",
        lambda *_args, **_kwargs: {
            "stage": "code_edit_draft",
            "model_id": "model/test",
            "model_ids": ("model/test",),
            "reasoning_effort": "low",
            "max_tokens": 32,
            "temperature": 0.0,
            "allow_non_zdr": False,
        },
    )
    context = ExecutionContextV2(
        job_id="autoresearch-v2:loop-privacy",
        purpose="research_lab.patch_draft.v2",
        epoch_id=1,
    )
    openrouter_context = {
        "key_ref": "encrypted_ref:openrouter:" + "1" * 32,
        "miner_hotkey": "miner-hotkey",
        "runtime_credential_value_hash": runtime_hash,
        "management_credential_value_hash": management_hash,
        "privacy_proof_doc": {
            "workspace_id_hash": "workspace-hash",
            "runtime_key_hash": runtime_hash.split(":", 1)[1],
            "runtime_key_label_hash": "label-hash",
            "runtime_key_creator_user_id_hash": "creator-hash",
            "management_key_hash": management_hash.split(":", 1)[1],
        },
    }
    executor = AutoresearchExecutorV2.__new__(AutoresearchExecutorV2)
    caller = executor._loop_model_caller(
        context=context,
        config=_config(),
        run_id="run-loop-privacy",
        model_id="model/test",
        model_doc={},
        openrouter_context=openrouter_context,
    )

    result = asyncio.run(caller([{"role": "user", "content": "test"}], 30, 32))

    assert result.content == "ok"
    assert observed_privacy_kwargs["expected_runtime_key_hash"] == (
        runtime_hash.split(":", 1)[1]
    )


def test_openrouter_guard_rejects_unleased_credential_commitment(monkeypatch):
    runtime_hash = "sha256:" + "5" * 64
    management_hash = "sha256:" + "6" * 64
    key_ref = "encrypted_ref:openrouter:" + "1" * 32
    run_id = "run-wrong-credential-commitment"
    queue_event_hash = "sha256:" + "a" * 64
    payload = {
        "schema_version": OPENROUTER_GUARD_REQUEST_SCHEMA_VERSION,
        "key_ref": key_ref,
        "key_ref_hash": sha256_bytes(key_ref.encode("utf-8")),
        "miner_hotkey_hash": "sha256:" + "9" * 64,
        "runtime_credential_value_hash": runtime_hash,
        "management_credential_value_hash": management_hash,
        "stage": "autoresearch_v2_authority",
        "request_policy": {
            "data_collection": "deny",
            "allow_fallbacks": False,
        },
        "run_id": run_id,
        "queue_event_hash": queue_event_hash,
        "run_state_hash": sha256_json(
            {"run_id": run_id, "queue_event_hash": queue_event_hash}
        ),
    }
    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2.verify_openrouter_workspace_privacy",
        lambda **_kwargs: pytest.fail("provider call must not begin"),
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider call must not begin"),
        retry_policy_hashes={
            "openrouter": "sha256:" + "4" * 64,
            "openrouter_management": "sha256:" + "8" * 64,
        },
        config_supplier=_config,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        with pytest.raises(
            AutoresearchExecutorV2Error,
            match="openrouter credential commitment differs",
        ):
            asyncio.run(
                executor(
                    OP_VERIFY_OPENROUTER_GUARD,
                    payload,
                    ExecutionContextV2(
                        job_id="autoresearch-v2:guard-wrong-commitment",
                        purpose="research_lab.openrouter_guard.v2",
                        epoch_id=1,
                        provider_credential_ref_hashes={
                            "openrouter": "sha256:" + "f" * 64,
                            "openrouter_management": management_hash,
                        },
                        allowed_purposes=frozenset(
                            ROLE_PURPOSES["gateway_autoresearch"]
                        ),
                    ),
                )
            )
    finally:
        executor.close()


def _stale_parent_context():
    return ExecutionContextV2(
        job_id="autoresearch-v2:stale-parent",
        purpose="research_lab.stale_parent_repair.v2",
        epoch_id=1,
        parent_receipt_hashes=("sha256:" + "a" * 64,),
        provider_credential_profile="stale_parent_repair",
        provider_credential_ref_hashes={"openrouter": "sha256:" + "b" * 64},
        allowed_purposes=frozenset(ROLE_PURPOSES["gateway_autoresearch"]),
    )


def test_stale_parent_direct_rebase_preserves_exact_draft_without_provider(
    tmp_path,
    monkeypatch,
):
    payload, draft = _stale_parent_payload(tmp_path)
    monkeypatch.setattr(
        CodeEditCandidateBuilder,
        "check_patch_applies",
        lambda *_args, **_kwargs: None,
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail(
            "direct stale-parent rebase must not call a provider"
        ),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=ResearchLabGatewayConfig,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        result = asyncio.run(
            executor(
                OP_REPAIR_STALE_PARENT,
                payload,
                _stale_parent_context(),
            )
        )
    finally:
        executor.close()

    assert result.output["schema_version"] == STALE_PARENT_REPAIR_RESULT_SCHEMA_VERSION
    assert result.output["repair_used"] is False
    assert result.output["draft"] == draft.to_dict()
    assert result.output["result_source_diff_hash"] == payload[
        "original_source_diff_hash"
    ]


def test_stale_parent_repair_runs_existing_prompt_and_parser_in_measured_scope(
    tmp_path,
    monkeypatch,
):
    from gateway.research_lab import scoring_worker

    payload, draft = _stale_parent_payload(tmp_path)
    checks = []
    calls = []

    def check_patch(_self, **_kwargs):
        checks.append(True)
        if len(checks) == 1:
            raise CodeEditPatchApplyError("patch does not apply")

    async def call_operator(**kwargs):
        calls.append(kwargs)
        return '{"candidates":[]}'

    monkeypatch.setattr(CodeEditCandidateBuilder, "check_patch_applies", check_patch)
    monkeypatch.setattr(scoring_worker, "_call_operator_openrouter_json", call_operator)
    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2.parse_code_edit_repair_response",
        lambda _raw, *, original_draft: (original_draft,),
    )
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: pytest.fail(
            "mocked OpenRouter helper makes no transport request"
        ),
        retry_policy_hashes={"openrouter": "sha256:" + "4" * 64},
        config_supplier=ResearchLabGatewayConfig,
        engine_factory=_FakeEngine,
        artifact_seal=_artifact_seal,
    )
    try:
        result = asyncio.run(
            executor(
                OP_REPAIR_STALE_PARENT,
                payload,
                _stale_parent_context(),
            )
        )
    finally:
        executor.close()

    assert result.output["repair_used"] is True
    assert result.output["draft"] == draft.to_dict()
    assert len(checks) == 2
    assert len(calls) == 1
    assert calls[0]["api_key"].startswith("sk-or-v1-")
    assert calls[0]["messages"]
