"""Read-only PostgREST schema gate for the selected gateway V2 release."""

from __future__ import annotations

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
    migrations = set()
    for migration, table, columns in REQUIRED_SUPABASE_V2_SCHEMA:
        query = urlencode({"select": ",".join(columns), "limit": "0"})
        request = Request(
            f"{supabase_url}/rest/v1/{table}?{query}",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {service_role_key}",
                "apikey": service_role_key,
            },
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
    return {
        "status": "ready",
        "probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA),
        "migration_files": sorted(migrations),
    }
