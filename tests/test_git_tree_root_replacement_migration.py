from pathlib import Path


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "115-research-lab-git-tree-root-replacement.sql"
).read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    return SQL.split(f"CREATE OR REPLACE FUNCTION public.{name}", 1)[1].split(
        "CREATE OR REPLACE FUNCTION",
        1,
    )[0]


def test_replacement_migration_preserves_history_and_serializes_generations():
    assert "DROP CONSTRAINT IF EXISTS research_lab_autoresearch_trees_run_id_key" in SQL
    assert "tree_generation INTEGER NOT NULL DEFAULT 0" in SQL
    assert "replaces_tree_id TEXT NULL" in SQL
    assert "cancellation_event_hash TEXT NULL" in SQL
    assert "replacement_hash TEXT NULL" in SQL
    assert "UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_run_generation" in SQL
    assert "ON public.research_lab_autoresearch_trees(run_id, tree_generation)" in SQL
    assert "UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_one_successor" in SQL
    assert "WHERE replaces_tree_id IS NOT NULL" in SQL
    assert "ON DELETE RESTRICT" in SQL
    assert "UPDATE public.research_lab_autoresearch_trees" not in SQL
    assert "DELETE FROM public.research_lab_autoresearch_trees" not in SQL


def test_tree_create_rpc_requires_latest_cancelled_predecessor():
    body = _function_body("create_research_lab_autoresearch_tree")
    for marker in (
        "pg_advisory_xact_lock",
        "one_active_version",
        "current_version_status = 'active'",
        "research_lab_git_tree_create_stale_active_root",
        "ORDER BY tree_generation DESC",
        "requested_generation IS DISTINCT FROM latest.tree_generation + 1",
        "requested_replaces_tree_id IS DISTINCT FROM latest.tree_id",
        "tree_cancelled_root_changed",
        "requested_cancellation_event_hash",
        "research_lab_git_tree_replacement_predecessor_active",
        "research_lab_git_tree_replacement_lineage_conflict",
    ):
        assert marker in body


def test_run_current_view_and_usage_rpc_cover_all_tree_generations():
    assert "CREATE OR REPLACE VIEW public.research_lab_autoresearch_run_tree_current" in SQL
    assert "SELECT DISTINCT ON (tree.run_id)" in SQL
    assert "tree.tree_generation DESC" in SQL
    usage = _function_body("research_lab_autoresearch_run_evaluation_usage")
    assert "JOIN public.research_lab_autoresearch_trees tree" in usage
    assert "tree.run_id = requested_run_id" in usage
    assert "settlement.operation_kind = 'evaluation'" in usage
    assert "'succeeded', 'failed', 'indeterminate'" in usage
    assert "operation_status = 'reserved'" in usage
    assert "'unsettled_operation_ids'" in usage
    assert "'indeterminate_operation_ids'" in usage


def test_candidate_handoff_is_atomic_and_fenced_to_the_active_latest_root():
    assert (
        "CREATE OR REPLACE FUNCTION public.create_research_lab_git_tree_candidate_handoff"
        in SQL
    )
    candidate_handoff = _function_body(
        "create_research_lab_git_tree_candidate_handoff"
    )
    for marker in (
        "INSERT INTO public.research_lab_candidate_artifacts",
        "public.record_research_lab_autoresearch_tree_handoff",
        "requested_candidate_doc->>'git_tree_id'",
        "research_lab_git_tree_candidate_content_conflict",
        "candidate_row.candidate_artifact_hash",
        "candidate_row.candidate_source_diff_hash",
        "candidate_row.candidate_patch_hash",
        "candidate_row.private_model_manifest_hash",
        "'candidate', pg_catalog.to_jsonb(candidate_row)",
    ):
        assert marker in candidate_handoff

    guard = _function_body("guard_research_lab_git_tree_handoff_active_root")
    for marker in (
        "pg_advisory_xact_lock",
        "one_active_version",
        "ORDER BY tree_generation DESC",
        "current_version_status = 'active'",
        "active_artifact_hash IS DISTINCT FROM tree_row.root_artifact_hash",
        "research_lab_git_tree_handoff_stale_active_root",
    ):
        assert marker in guard
    assert "BEFORE INSERT ON public.research_lab_autoresearch_tree_handoffs" in SQL


def test_every_private_model_status_transition_shares_the_handoff_lock():
    guard = _function_body(
        "guard_research_lab_one_active_private_model_version"
    )
    lock_index = guard.index("pg_advisory_xact_lock")
    early_return_index = guard.index("IF NEW.version_status <> 'active'")
    assert lock_index < early_return_index


def test_replacement_authority_remains_service_role_only():
    assert (
        "REVOKE ALL ON TABLE public.research_lab_autoresearch_run_tree_current"
        in SQL
    )
    assert "FROM PUBLIC, anon, authenticated" in SQL
    assert (
        "GRANT SELECT ON TABLE public.research_lab_autoresearch_run_tree_current"
        in SQL
    )
    assert (
        "REVOKE ALL ON FUNCTION public.research_lab_autoresearch_run_evaluation_usage(UUID)"
        in SQL
    )
    assert (
        "REVOKE ALL ON FUNCTION public.create_research_lab_git_tree_candidate_handoff"
        in SQL
    )
    assert "TO service_role" in SQL
