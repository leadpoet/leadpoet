from __future__ import annotations

import asyncio
import base64
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.code_loop_engine import (
    BuiltCodeEditCandidate,
    CodeEditLoopResult,
)
from gateway.research_lab.git_tree_models import (
    TreeCheckpoint,
    TreePolicy,
    TreeResult,
    derive_tree_id,
)
from gateway.research_lab.code_build import (
    CodeEditBuildResult,
    CodeEditCandidateBuilder,
    CodeEditPatchApplyError,
)
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.autoresearch_runtime import AutoResearchLoopEvent
from gateway.tee.autoresearch_executor_v2 import (
    AUTORESEARCH_REQUEST_SCHEMA_VERSION,
    HOST_APPEND_EVENT,
    HOST_EVENT_RESULT_SCHEMA_VERSION,
    HOST_GIT_TREE,
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
    _source_context,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
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
from research_lab.eval import (
    PrivateModelArtifactManifest,
    build_local_private_artifact_manifest,
)
from research_lab.code_editing import CodeEditDraft


PRIVACY_RECEIPT_HASH = "sha256:" + "2" * 64


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
        tree_id = derive_tree_id(
            run_id=kwargs["run_id"],
            root_artifact_hash=kwargs["artifact"].model_artifact_hash,
            policy=policy,
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
    manifest = build_local_private_artifact_manifest(
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


def _payload(tmp_path: Path):
    source_bundle, artifact = _source_and_artifact(tmp_path)
    tree_policy = TreePolicy(mode="active")
    active_model_result = {
        "schema_version": "leadpoet.active_private_model.v2",
        "artifact": artifact,
        "active_model": {
            "private_model_version_id": "private-model-v1",
        },
        "source_state_hash": "sha256:" + "f" * 64,
    }
    active_model_graph = _active_model_graph(active_model_result)
    component_registry = {"schema_version": "1.0", "components": []}
    component_result = {
        "schema_version": "leadpoet.model_sandbox_result.v2",
        "operation": "metadata",
        "output": component_registry,
        "output_hash": sha256_json(component_registry),
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
        "run_id": "run-v2-1",
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
                "schema_version": "research_lab.git_tree_runtime_policy.v1",
                "policy": tree_policy.to_dict(),
                "evaluator_enabled": True,
                "evaluator_commitment": {
                    "schema_version": "research_lab.git_tree_evaluator_commitment.v1",
                    "snapshot_manifest_hash": "sha256:" + "7" * 64,
                    "snapshot_ready_hash": "sha256:" + "8" * 64,
                    "dev_set_hash": "sha256:" + "9" * 64,
                    "dev_set_size": tree_policy.live_max_icps_per_node,
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
            "key_ref": "encrypted_ref:openrouter:" + "1" * 32,
            "miner_hotkey": "miner-hotkey",
            "privacy_proof_doc": {"status": "verified"},
            "privacy_receipt_hash": PRIVACY_RECEIPT_HASH,
            "runtime_credential_value_hash": "sha256:" + "5" * 64,
            "management_credential_value_hash": "sha256:" + "6" * 64,
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


def _component_registry_graph(component_result):
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
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.private_model_run.v2",
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
        PRIVACY_RECEIPT_HASH,
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
        Context(), tree_id=tree_id
    ).operation_settlement_commitment()

    assert observed["operation"] == HOST_GIT_TREE
    assert observed["payload"] == {
        "action": "operation_settlement_commitment",
        "tree_id": tree_id,
    }
    assert result["operation_count"] == 3
    assert result["settled_cost_microusd"] == 12_345
    assert result["provider_call_count"] == 2


def test_v2_builder_restores_git_tree_parent_from_cumulative_patch(tmp_path):
    source_bundle, root_artifact_doc = _source_and_artifact(tmp_path)
    root_artifact = PrivateModelArtifactManifest.from_mapping(root_artifact_doc)
    source_root = tmp_path / "private-source"
    child_root = tmp_path / "child-source"
    shutil.copytree(source_root, child_root)
    child_file = child_root / "sourcing_model" / "runtime.py"
    child_file.write_text("VALUE = 2\n", encoding="utf-8")
    child_artifact = PrivateModelArtifactManifest.from_mapping(
        build_local_private_artifact_manifest(
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
        with pytest.raises(AutoresearchExecutorV2Error, match="privacy receipt hash"):
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


def test_openrouter_guard_returns_only_committed_redacted_key_evidence(monkeypatch):
    runtime_hash = "sha256:" + "5" * 64
    management_hash = "sha256:" + "6" * 64
    key_ref = "encrypted_ref:openrouter:" + "1" * 32
    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2.preflight_openrouter_key",
        lambda _key: {"limit_remaining": "0.00", "usage": 12},
    )
    monkeypatch.setattr(
        "gateway.tee.autoresearch_executor_v2.verify_openrouter_workspace_privacy",
        lambda **_kwargs: {
            "workspace_id_hash": "workspace-hash",
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
                },
                ExecutionContextV2(
                    job_id="autoresearch-v2:guard",
                    purpose="research_lab.openrouter_guard.v2",
                    epoch_id=1,
                    allowed_purposes=frozenset(
                        ROLE_PURPOSES["gateway_autoresearch"]
                    ),
                ),
            )
        )
    finally:
        executor.close()
    assert result.output["credit_depleted"] is True
    assert result.output["management_credential_value_hash"] == management_hash
    assert result.output["privacy_proof_doc"]["management_key_hash"] == (
        management_hash.split(":", 1)[1]
    )
    assert "sk-or-v1-" not in str(result.output)


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
