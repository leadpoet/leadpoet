"""Hash and verify protected Research Lab business-logic ASTs.

The manifest deliberately hashes selected function/class bodies rather than
whole files. I/O adapters and imports can move around those bodies while CI
continues to fail if scoring, autoresearch, promotion, accounting, allocation,
or weight behavior changes unintentionally.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA_VERSION = "leadpoet.protected_workflows.v2"
DEFAULT_MANIFEST = Path(__file__).with_name("protected_workflows.json")

PROTECTED_SYMBOLS = {
    "leadpoet_canonical/kms_recipient.py": (
        "decrypt_kms_recipient_ciphertext",
    ),
    "gateway/tee/coordinator_epoch_cutover_v2.py": (
        "attest_subnet_epoch_cutover_v2",
    ),
    "gateway/tee/execution_job_manager_v2.py": (
        "ExecutionJobManagerV2",
    ),
    "gateway/research_lab/tee_protocol.py": (
        "normalize_tee_protocol",
    ),
    "gateway/api/weights.py": (
        "get_weight_inputs_v2",
        "submit_weights_v2",
        "finalize_weights_v2",
        "get_attested_weights_v2",
        "submit_weights",
    ),
    "research_lab/code_editing.py": (
        "build_loop_direction_planner_messages",
        "build_loop_direction_reference_repair_messages",
        "build_code_edit_source_inspection_messages",
        "build_code_edit_auto_research_messages",
        "build_code_edit_fallback_messages",
        "build_plan_alignment_judge_messages",
        "build_code_edit_repair_messages",
        "parse_loop_direction_plan_response",
        "parse_plan_alignment_judge_response",
        "parse_code_edit_source_inspection_response",
        "parse_code_edit_response",
        "parse_code_edit_repair_response",
        "validate_code_edit_draft",
        "code_edit_candidate_manifest",
    ),
    "gateway/research_lab/autoresearch_runtime.py": (
        "AutoResearchRuntimeSettings",
        "AutoResearchLoopEvent",
        "would_exceed_budget",
    ),
    "gateway/research_lab/code_loop_engine.py": (
        "CodeEditLoopEngine",
        "_bind_loop_direction_plan",
    ),
    "gateway/research_lab/git_tree_models.py": (
        "TreePolicy",
        "TreeReplacement",
        "TreeEvaluation",
        "TreeNode",
        "TreeCheckpoint",
        "TreeResult",
        "derive_tree_id",
        "derive_frontier_commitment_hash",
        "derive_child_slot",
        "generation_operation_id",
        "build_operation_id",
        "cohort_evaluation_operation_id",
        "tree_rank_key",
        "select_finalist",
        "next_child_slot",
    ),
    "gateway/research_lab/git_tree_scheduler.py": (
        "GitTreeScheduler",
        "sanitized_branch_context",
    ),
    "gateway/research_lab/git_tree_repository.py": (
        "GitTreeRepository",
    ),
    "gateway/research_lab/git_tree_store.py": (
        "GitTreeStore",
    ),
    "gateway/research_lab/git_tree_evaluator.py": (
        "TreeEvaluationPlan",
        "classify_tree_evaluation",
        "classify_candidate_tree_evaluation",
    ),
    "gateway/research_lab/dev_eval_runner.py": (
        "snapshot_readiness",
        "DockerReplayDevEvaluator",
        "AttestedReplayDevEvaluatorV2",
    ),
    "gateway/research_lab/champion_settlement_v2.py": (
        "validate_finalized_allocation_authorities_v2",
        "validate_legacy_settlement_migrations_v2",
        "validate_legacy_allocation_nonfinalizations_v2",
        "merge_finalized_allocation_histories_v2",
        "load_finalized_allocation_history_v2",
        "load_legacy_allocation_nonfinalizations_v2",
        "champion_v2_cutover_readiness",
    ),
    "gateway/research_lab/attested_v2_store.py": (
        "load_receipt_graph_v2",
        "load_receipt_graphs_v2",
        "load_business_artifact_graph_by_ref_v2",
        "load_business_artifact_graphs_by_ref_v2",
        "persist_legacy_finalized_allocation_migration_v2",
        "persist_legacy_allocation_nonfinalization_v2",
    ),
    "gateway/research_lab/autoresearch_authority_v2.py": (
        "run_authoritative_autoresearch_v2",
    ),
    "gateway/research_lab/snapshot_refresh.py": (
        "maybe_refresh_dev_snapshot",
    ),
    "gateway/tee/autoresearch_executor_v2.py": (
        "AutoresearchExecutorV2",
    ),
    "research_lab/eval/evaluator.py": (
        "evaluate_private_model_pair",
        "score_private_model_pair_items",
        "_score_single_icp",
        "_run_candidate_with_retries",
        "build_holdout_gate_result",
        "_score_with_private_holdout_gate",
        "benchmark_icp_score_from_company_scores",
        "_benchmark_icp_score",
        "build_score_bundle_from_scored_icps",
        "build_scoring_health_doc",
        "prepare_autoresearch_scoring_payload",
    ),
    "research_lab/eval/dev_eval.py": (
        "build_current_day_dev_bank",
        "select_current_day_dev_icps",
        "select_snapshot_dev_icps",
        "build_dev_icp_set",
        "DevEvalResult",
        "evaluate_dev",
        "_score_dev_items",
    ),
    "research_lab/eval/snapshot_store.py": (
        "build_snapshot_pointer_document",
        "verify_snapshot_pointer_document",
        "ProviderSnapshotStore",
    ),
    "research_lab/eval/baseline_summary.py": (
        "build_baseline_health",
        "daily_noise_budget_doc",
        "build_baseline_score_summary",
    ),
    "research_lab/eval/promotion_metric.py": (
        "_paired_lcb_promotion_metric",
        "promotion_improvement_metric",
        "promotion_gate_decision",
    ),
    "research_lab/eval/provider_evidence_cache.py": (
        "canonical_request_fingerprint",
        "icp_evidence_cache_key",
        "build_evidence_cache_from_trace_entries",
        "merge_evidence_caches",
    ),
    "research_lab/eval/provider_costs.py": (
        "estimate_provider_cost",
        "ProviderCostLedger",
        "summarize_provider_cost_events",
        "summarize_provider_cost_trace_entries",
    ),
    "research_lab/source_add_rewards.py": (
        "SourceAddRewardRecord",
        "validate_source_add_reward_record",
        "create_leg1_reward",
        "create_leg2_reward",
        "stop_reward_forward",
    ),
    "gateway/research_lab/source_add_provenance.py": (
        "sanitize_source_add_precheck_doc",
        "evaluate_source_add_provenance",
    ),
    "gateway/research_lab/source_add_workflow.py": (
        "build_automatic_probe_config",
        "process_source_add_work_item",
        "_process_provenance",
        "_process_functional_probe",
        "_process_leg1_reward",
        "_retry_allowed",
        "_retry_at",
    ),
    "gateway/research_lab/source_add_llm_judge.py": (
        "SourceAddJudgeVerdict",
        "judge_source_add_implementation",
        "_parse_verdict",
    ),
    "gateway/research_lab/allocations.py": (
        "build_research_lab_allocation_bundle",
        "_champion_finalized_paid_alpha_to_date",
        "_champion_obligation_caps",
        "_champion_paid_alpha_to_date_from_snapshots",
        "_champion_replay_obligation",
        "_epoch_active",
    ),
    "gateway/research_lab/maintenance.py": (
        "reconcile_champion_reward_statuses",
        "backfill_champion_reward_v2_authority",
        "backfill_champion_settlement_v2_authority",
        "backfill_source_add_reward_v2_authority",
        "champion_v2_cutover_readiness_report",
    ),
    "gateway/research_lab/arweave_audit.py": (
        "_verified_rebuffer_event",
        "record_research_lab_checkpointed_events",
        "rebuffer_research_lab_buffered_audit_events",
        "recover_research_lab_checkpointed_audit_epochs",
    ),
    "gateway/research_lab/v2_authority.py": (
        "attest_historical_champion_reward_v2",
        "attest_historical_champion_settlement_v2",
        "attest_historical_source_add_reward_v2",
        "classify_historical_champion_allocation_v2",
        "build_allocation_v2",
        "_load_allocation_parent_graphs_v2",
    ),
    "gateway/research_lab/promotion.py": (
        "confirmation_min_delta",
        "confirmation_attempt_budget",
        "_baseline_aggregate_excluding_icps",
        "ResearchLabPromotionController.process_scored_candidate",
        "_candidate_icp_score",
    ),
    "leadpoet_canonical/weight_computation.py": (
        "weight_config_document",
        "normalize_to_u16_with_uids_pure",
        "research_lab_uid_weights_from_allocation",
        "compute_final_weights",
    ),
    "leadpoet_canonical/weight_authority_v2.py": (
        "gateway_weight_input_value_documents_v2",
        "weight_input_value_documents_v2",
        "validate_weight_input_source_evidence_v2",
        "build_weight_snapshot_v2",
        "validate_published_weight_bundle_v2",
        "validate_weight_finalization_submission_v2",
    ),
    "leadpoet_canonical/chain_source_v2.py": (
        "weights_storage_key",
        "decode_weights_storage",
        "validate_arweave_checkpoint_event",
    ),
    "leadpoet_canonical/legacy_settlement_v2.py": (
        "validate_legacy_weight_bundle_v2",
        "legacy_chain_vector_matches_bundle_v2",
        "validate_legacy_audit_event_v2",
        "validate_legacy_allocation_nonfinalization_v2",
        "validate_legacy_nonfinalization_document_v2",
        "validate_legacy_finalized_settlement_v2",
        "validate_legacy_settlement_document_v2",
    ),
    "leadpoet_verifier/economics.py": (
        "compute_reimbursement_award",
        "build_reimbursement_schedule",
        "build_champion_reward_obligation",
        "allocate_research_lab_epoch",
        "_allocate_research_lab_epoch_existing",
        "cap_reimbursement_schedules_by_epoch",
        "compose_final_weight_vector",
        "_allocate_reimbursements_no_champions",
        "_allocate_reimbursements_with_champions",
        "_allocate_champions",
        "_allocate_source_add",
        "_allocate_capped_pro_rata",
    ),
    "gateway/tee/coordinator_reward_source_v2.py": (
        "CoordinatorRewardSourceV2",
    ),
    "gateway/tee/coordinator_allocation_source_v2.py": (
        "CoordinatorAllocationSourceV2",
    ),
    "gateway/tee/coordinator_legacy_settlement_v2.py": (
        "CoordinatorLegacySettlementSourceV2",
    ),
    "gateway/tee/coordinator_executor_v2.py": (
        "CoordinatorExecutorV2",
    ),
    "gateway/tee/coordinator_active_model_source_v2.py": (
        "CoordinatorActiveModelSourceV2",
    ),
    "gateway/tee/coordinator_chain_source_v2.py": (
        "CoordinatorChainSourceV2",
    ),
    "gateway/tee/coordinator_source_add_v2.py": (
        "CoordinatorSourceAddProvenanceV2",
        "CoordinatorSourceAddFunctionalProbeV2",
    ),
    "gateway/tee/source_add_runtime_v2.py": (
        "validate_source_add_credential_envelope_v2",
        "validate_source_add_sealed_job_envelope_v2",
        "build_source_add_probe_route_v2",
        "validate_source_add_runtime_route_v2",
        "build_source_add_probe_job_envelope_v2",
    ),
    "gateway/tee/coordinator_weight_source_v2.py": (
        "CoordinatorWeightSourceV2",
    ),
    "gateway/tee/model_sandbox_v2.py": (
        "RunscModelSandboxV2",
    ),
    "gateway/tee/provider_broker_v2.py": (
        "_extract_tls_metadata",
        "HTTPXProviderTransport",
        "ProviderBrokerV2",
        "provider_registry_document",
    ),
    "gateway/tee/provider_semantics_v2.py": (
        "ProviderSemanticsAuthorityV2",
    ),
    "gateway/tee/tee_service.py": (
        "acknowledge_checkpoint",
        "build_checkpoint",
    ),
    "gateway/tee/rpc_authority.py": (
        "allowed_exact_methods",
        "rpc_method_allowed",
    ),
    "gateway/utils/arweave_client.py": (
        "checkpoint_payload_bytes",
        "upload_checkpoint",
        "wait_for_confirmation",
    ),
    "gateway/tasks/hourly_batch.py": (
        "build_arweave_checkpoint_log_event",
        "hourly_batch_task",
    ),
    "gateway/tee/reward_executor_v2.py": (
        "reward_receipt_projection_v2",
        "champion_reward_row_projection_v2",
        "source_add_reward_row_projection_v2",
        "reimbursement_reward_row_projection_v2",
        "execute_reward_decision_v2",
        "_source_add_migration",
    ),
    "gateway/tee/verify_weight_submission_ready_v2.py": (
        "verify_weight_submission_ready_v2",
    ),
    "gateway/tee/release_lineage_v2.py": (
        "load_approved_release_lineage_v2",
        "build_release_lineage_boot_verifier_v2",
    ),
    "gateway/tee/scoring_executor_v2.py": (
        "ScoringExecutorV2",
    ),
}


class ProtectedWorkflowError(RuntimeError):
    """A protected symbol is absent or has changed from the baseline."""


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node: Any) -> Any:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), (ast.Str, ast.Constant))
            and isinstance(getattr(body[0].value, "s", getattr(body[0].value, "value", None)), str)
        ):
            node.body = body[1:]
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        return self._strip(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return self._strip(node)


def _symbol_index(tree: ast.Module) -> Dict[str, ast.AST]:
    index = {}  # type: Dict[str, ast.AST]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            index[node.name] = node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    index[node.name + "." + child.name] = child
    return index


def _symbol_hash(node: ast.AST) -> str:
    normalized = _StripDocstrings().visit(ast.fix_missing_locations(node))
    encoded = ast.dump(normalized, annotate_fields=True, include_attributes=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _manifest_hash(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_path(root: Path, relative_path: str) -> Path:
    direct = root / relative_path
    if direct.is_file():
        return direct
    if relative_path.startswith("gateway/"):
        gateway_relative = root / relative_path.split("/", 1)[1]
        if gateway_relative.is_file():
            return gateway_relative
    staged = root / "_attested_runtime" / relative_path
    if staged.is_file():
        return staged
    return direct


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().lower()
    except Exception as exc:
        raise ProtectedWorkflowError("cannot resolve baseline Git commit") from exc


def build_manifest(
    root: Path,
    *,
    baseline_commit: str = "",
    protected_source_commit: str = "",
) -> Dict[str, Any]:
    root = root.resolve()
    entries = []
    for relative_path, symbols in sorted(PROTECTED_SYMBOLS.items()):
        path = _source_path(root, relative_path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            raise ProtectedWorkflowError("cannot parse protected file %s" % relative_path) from exc
        index = _symbol_index(tree)
        for symbol in symbols:
            if symbol not in index:
                raise ProtectedWorkflowError(
                    "protected symbol %s:%s is missing" % (relative_path, symbol)
                )
            entries.append(
                {
                    "path": relative_path,
                    "symbol": symbol,
                    "ast_sha256": _symbol_hash(index[symbol]),
                }
            )
    entries.sort(key=lambda item: (item["path"], item["symbol"]))
    body = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": baseline_commit or _git_commit(root),
        "protected_source_commit": protected_source_commit or _git_commit(root),
        "entries": entries,
    }
    return {**body, "manifest_hash": _manifest_hash(body)}


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    encoded = json.dumps(
        dict(manifest),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    ) + "\n"
    path.write_text(encoded, encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProtectedWorkflowError("cannot read protected workflow manifest") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("entries"), list)
        or set(value)
        != {
            "schema_version",
            "baseline_commit",
            "protected_source_commit",
            "entries",
            "manifest_hash",
        }
    ):
        raise ProtectedWorkflowError("protected workflow manifest schema is invalid")
    body = {
        key: value[key]
        for key in (
            "schema_version",
            "baseline_commit",
            "protected_source_commit",
            "entries",
        )
    }
    if value.get("manifest_hash") != _manifest_hash(body):
        raise ProtectedWorkflowError("protected workflow manifest hash is invalid")
    return dict(value)


def verify_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = build_manifest(
        root,
        baseline_commit=str(manifest.get("baseline_commit") or ""),
        protected_source_commit=str(
            manifest.get("protected_source_commit") or ""
        ),
    )
    if dict(manifest) != expected:
        expected_by_key = {
            (item["path"], item["symbol"]): item["ast_sha256"]
            for item in expected["entries"]
        }
        observed_by_key = {
            (item.get("path"), item.get("symbol")): item.get("ast_sha256")
            for item in manifest.get("entries", [])
            if isinstance(item, dict)
        }
        changed = sorted(
            "%s:%s" % key
            for key in set(expected_by_key) | set(observed_by_key)
            if expected_by_key.get(key) != observed_by_key.get(key)
        )
        raise ProtectedWorkflowError(
            "protected workflow manifest mismatch: %s" % ", ".join(changed)
        )


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--baseline-commit", default="")
    parser.add_argument("--protected-source-commit", default="")
    args = parser.parse_args(list(argv) if argv else None)
    if args.write:
        write_manifest(
            build_manifest(
                args.root,
                baseline_commit=args.baseline_commit,
                protected_source_commit=args.protected_source_commit,
            ),
            args.manifest,
        )
    else:
        verify_manifest(args.root, load_manifest(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
