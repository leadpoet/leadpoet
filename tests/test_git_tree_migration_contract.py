from pathlib import Path


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "95-research-lab-git-tree-autoresearch.sql"
).read_text(encoding="utf-8")


def test_tree_migration_is_append_only_and_service_role_only():
    for table in (
        "research_lab_autoresearch_trees",
        "research_lab_autoresearch_tree_nodes",
        "research_lab_autoresearch_tree_events",
        "research_lab_autoresearch_operation_settlements",
        "research_lab_autoresearch_frontier_commitments",
        "research_lab_autoresearch_tree_handoffs",
    ):
        assert f"CREATE TABLE IF NOT EXISTS public.{table}" in SQL
        assert "prevent_research_lab_attested_v2_mutation" in SQL
        assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "FROM anon, authenticated" in SQL
    assert "FROM anon, authenticated, service_role" in SQL
    assert "GRANT SELECT, INSERT ON TABLE public.%I TO service_role" not in SQL
    assert "GRANT SELECT ON TABLE public.%I TO service_role" in SQL


def test_tree_migration_has_concurrency_and_exactly_one_finalist_guards():
    assert "pg_advisory_xact_lock" in SQL
    assert "research_lab_git_tree_operation_already_terminal" in SQL
    assert "uq_research_lab_tree_operation_terminal" in SQL
    assert "uq_research_lab_tree_final_selected" in SQL
    assert "uq_research_lab_tree_node_generated" in SQL
    assert "uq_research_lab_tree_terminal" in SQL
    assert "uq_research_lab_candidate_one_per_git_tree" in SQL
    assert "record_research_lab_autoresearch_tree_handoff" in SQL
    assert "paid_finalist_count' IS DISTINCT FROM '1'" in SQL
    assert "expected_previous_hash" in SQL
    assert "'tree_completed'" in SQL
    assert "'tree_failed'" in SQL
    assert "requested_completed_event_hash" in SQL
    assert "selected_candidate_artifact_hash" in SQL
    assert "selected_node_git_commit" in SQL
    assert "selected_lineage_hash" in SQL
    assert "candidate_row.candidate_artifact_hash" in SQL
    assert "selected_node_event.event_doc->>'candidate_artifact_hash'" in SQL
    assert "selected_node_event.event_doc->>'lineage_hash'" in SQL
    assert "research_lab_git_tree_final_selection_authority_conflict" in SQL
    final_selector = SQL.split(
        "CREATE OR REPLACE FUNCTION public.select_research_lab_autoresearch_tree_final",
        1,
    )[1].split("CREATE OR REPLACE FUNCTION", 1)[0]
    assert "hashtext('research_lab_git_tree_final')" in final_selector
    assert "hashtext('research_lab_git_tree_event')" in final_selector
    generic_append = SQL.split(
        "CREATE OR REPLACE FUNCTION public.append_research_lab_autoresearch_tree_event",
        1,
    )[1].split("CREATE OR REPLACE FUNCTION", 1)[0]
    assert "requested_selection_doc" not in generic_append


def test_tree_migration_repairs_complete_legacy_loop_event_allowlist():
    for event_type in (
        "checkpoint_saved",
        "source_inspection_requested",
        "source_inspection_seeded",
        "candidate_generation_fallback_drafted",
        "allocator_decision",
        "probe_resolved",
        "loop_paused",
    ):
        assert f"'{event_type}'" in SQL


def test_tree_migration_retries_are_semantically_idempotent():
    assert "AND event_type = requested_event_type" in SQL
    assert "AND event_doc = requested_event_doc" in SQL
    assert "AND frontier_hash = requested_frontier_hash" in SQL
    assert "research_lab_git_tree_frontier_identity_conflict" in SQL
    assert "research_lab_git_tree_event_identity_conflict" in SQL
    assert "research_lab_git_tree_frontier_commitment_conflict" in SQL
    assert "existing.event_hash IS DISTINCT FROM requested_event_hash" in SQL
    assert "inserted.tree_id IS DISTINCT FROM requested_tree_id" in SQL
    assert "latest.node_id IS DISTINCT FROM NULLIF(requested_node_id, '')" in SQL
