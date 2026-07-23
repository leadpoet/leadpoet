"""Read-only PostgREST schema gate for the selected gateway V2 release."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REQUIRED_SUPABASE_V2_SCHEMA = (
    (
        "scripts/92-validator-sourcing-attested-v2.sql",
        "validator_sourcing_epoch_inputs_v2",
        ("epoch_id", "epoch_hash", "decision_root", "receipt_hash"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_trees",
        ("tree_id", "run_id", "root_artifact_hash"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_nodes",
        ("tree_id", "node_id", "parent_node_id"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_events",
        ("tree_id", "seq", "event_hash"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_operation_settlements",
        ("logical_operation_id", "seq", "tree_id"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_frontier_commitments",
        ("tree_id", "round_index", "frontier_hash"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_handoffs",
        ("tree_id", "node_id", "handoff_hash"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_candidate_artifacts",
        (
            "git_tree_id",
            "git_tree_node_id",
            "git_tree_root_commit",
            "git_tree_node_commit",
            "git_tree_lineage_hash",
        ),
    ),
    (
        "scripts/97-research-lab-conditional-validation.sql",
        "research_lab_conditional_validation_events",
        ("event_id", "assignment_hash", "policy_hash"),
    ),
    (
        "scripts/97-research-lab-conditional-validation.sql",
        "research_lab_scoring_category_results",
        ("category_result_id", "source_kind", "category", "assignment_hash"),
    ),
    (
        "scripts/97-research-lab-conditional-validation.sql",
        "research_lab_scoring_job_candidate",
        (
            "conditional_total",
            "baseline_preliminary_score",
            "threshold_points",
            "preliminary_gate_status",
            "category_assignment_hash",
            "conditional_policy_hash",
        ),
    ),
    (
        "scripts/99-research-lab-v2-champion-settlement.sql",
        "research_lab_legacy_finalized_allocation_migrations_v2",
        ("netuid", "epoch_id", "allocation_hash", "settlement_receipt_hash"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_candidates_v1",
        ("snapshot_hash", "netuid", "subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_cutovers_v1",
        ("cutover_authority_hash", "netuid", "first_subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_boundaries_v1",
        ("boundary_hash", "netuid", "subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_snapshots_v1",
        ("snapshot_hash", "netuid", "subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_cutover_state_v1",
        ("lifecycle_state", "mapping_hash", "netuid", "updated_at"),
    ),
    (
        "scripts/103-research-lab-legacy-allocation-nonfinalization.sql",
        "research_lab_legacy_allocation_nonfinalizations_v2",
        ("netuid", "epoch_id", "allocation_hash", "finding_receipt_hash"),
    ),
    (
        "scripts/117-research-lab-maintenance-lease.sql",
        "research_lab_maintenance_lease",
        ("lease_name", "holder_ref", "expires_at"),
    ),
    (
        "scripts/120-research-lab-atomic-candidate-claim.sql",
        "research_lab_candidate_claim",
        ("candidate_id", "holder_ref", "claimed_at", "expires_at"),
    ),
    (
        "scripts/121-research-lab-atomic-run-claim.sql",
        "research_loop_run_claim",
        ("run_id", "holder_ref", "claimed_at", "expires_at"),
    ),
    (
        "scripts/122-research-lab-corpus-completeness.sql",
        "research_lab_corpus_complete",
        ("trajectory_id", "run_id", "source_watermark", "completed_at"),
    ),
)

REQUIRED_SUPABASE_V2_RPCS = (
    (
        "scripts/116-research-lab-trajectory-antijoin.sql",
        "research_lab_missing_trajectory_ids",
    ),
    (
        "scripts/117-research-lab-maintenance-lease.sql",
        "research_lab_acquire_maintenance_lease",
    ),
    (
        "scripts/118-research-lab-provider-usage-batch-insert.sql",
        "insert_research_lab_provider_usage_ledger_rows",
    ),
    (
        "scripts/119-research-lab-trajectory-delta.sql",
        "research_lab_next_unprojected_terminal_runs",
    ),
    (
        "scripts/119-research-lab-trajectory-delta.sql",
        "research_lab_terminal_runs_missing_traces",
    ),
    (
        "scripts/120-research-lab-atomic-candidate-claim.sql",
        "claim_next_research_lab_candidate",
    ),
    (
        "scripts/121-research-lab-atomic-run-claim.sql",
        "claim_next_research_loop_run",
    ),
    (
        "scripts/122-research-lab-corpus-completeness.sql",
        "research_lab_corpus_source_watermark",
    ),
    (
        "scripts/122-research-lab-corpus-completeness.sql",
        "research_lab_mark_corpus_complete",
    ),
    (
        "scripts/122-research-lab-corpus-completeness.sql",
        "research_lab_terminal_runs_needing_corpus",
    ),
)


class SupabaseSchemaPreflightV2Error(RuntimeError):
    """The selected V2 release cannot use the live PostgREST schema."""


def verify_required_supabase_v2_schema(
    parent_environment: Mapping[str, str],
    *,
    opener: Any = urlopen,
    timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    supabase_url = str(parent_environment.get("SUPABASE_URL") or "").rstrip("/")
    service_role_key = str(parent_environment.get("SUPABASE_SERVICE_ROLE_KEY") or "")
    if not supabase_url or not service_role_key:
        raise SupabaseSchemaPreflightV2Error(
            "prepared parent environment lacks Supabase V2 schema credentials"
        )
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
    }
    migrations = set()
    for migration, table, columns in REQUIRED_SUPABASE_V2_SCHEMA:
        query = urlencode({"select": ",".join(columns), "limit": "0"})
        request = Request(
            f"{supabase_url}/rest/v1/{table}?{query}",
            headers=headers,
        )
        try:
            with opener(request, timeout=timeout_seconds) as response:
                status = int(response.getcode())
                response.read(1)
        except HTTPError as exc:
            raise SupabaseSchemaPreflightV2Error(
                "required Supabase V2 schema is unavailable for "
                f"{table}; apply {migration} before restart (HTTP {exc.code})"
            ) from exc
        except Exception as exc:
            raise SupabaseSchemaPreflightV2Error(
                f"Supabase V2 schema probe failed for {table}"
            ) from exc
        if status < 200 or status >= 300:
            raise SupabaseSchemaPreflightV2Error(
                "required Supabase V2 schema is unavailable for "
                f"{table}; apply {migration} before restart (HTTP {status})"
            )
        migrations.add(migration)
    # PostgREST returns 200 to OPTIONS even for a nonexistent /rpc path, so an
    # OPTIONS probe cannot prove function availability. The service-role OpenAPI
    # document lists only functions present in the active schema cache and
    # executable by that role; inspect it once without executing any RPC.
    schema_request = Request(
        f"{supabase_url}/rest/v1/",
        headers={**headers, "Accept": "application/openapi+json"},
    )
    try:
        with opener(schema_request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded_schema = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            f"Supabase V2 RPC schema probe failed (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "Supabase V2 RPC schema probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            f"Supabase V2 RPC schema probe failed (HTTP {status})"
        )
    try:
        schema_document = json.loads(encoded_schema.decode("utf-8"))
        schema_paths = schema_document["paths"]
        if not isinstance(schema_paths, Mapping):
            raise TypeError("OpenAPI paths must be an object")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "Supabase V2 RPC schema document is invalid"
        ) from exc
    for migration, function_name in REQUIRED_SUPABASE_V2_RPCS:
        if f"/rpc/{function_name}" not in schema_paths:
            raise SupabaseSchemaPreflightV2Error(
                "required Supabase V2 RPC is unavailable for "
                f"{function_name}; apply {migration} before restart"
            )
        migrations.add(migration)
    return {
        "status": "ready",
        "probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA)
        + len(REQUIRED_SUPABASE_V2_RPCS),
        "table_probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA),
        "rpc_probe_count": len(REQUIRED_SUPABASE_V2_RPCS),
        "schema_document_probe_count": 1,
        "migration_files": sorted(migrations),
    }
