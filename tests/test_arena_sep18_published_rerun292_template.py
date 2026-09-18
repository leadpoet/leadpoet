from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "scripts/292-arena-2026-09-18-published-baseline-rerun.sql.template"


def placeholders(value: str) -> set[str]:
    return set(re.findall(r"__[A-Z0-9_]+__", value))


def test_template_requires_all_live_seals_and_no_guessed_source_identity():
    sql = TEMPLATE.read_text()
    assert placeholders(sql) == {
        "__FORWARD_SCHEDULE_JSON__",
        "__SCORING_DEFINITION_SHA256__",
        "__NEW_SOURCE_COMMIT__", "__NEW_SOURCE_SHA256__",
        "__NEW_SOURCE_SIZE_BYTES__",
        "__TERMINAL_ROUND_JSON__", "__TERMINAL_BASELINE_JSON__",
        "__TERMINAL_SOURCE_COMMIT__", "__TERMINAL_SOURCE_SHA256__",
        "__TERMINAL_SOURCE_SIZE_BYTES__",
        "__TERMINAL_BASELINE_RUN_COUNT__", "__TERMINAL_BASELINE_RUNS_HASH__",
        "__TERMINAL_BASELINE_LEDGER_COUNT__", "__TERMINAL_BASELINE_LEDGER_HASH__",
        "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__",
        "__TERMINAL_NONBASELINE_SUBMISSIONS_HASH__",
        "__TERMINAL_NONBASELINE_RUN_COUNT__", "__TERMINAL_NONBASELINE_RUNS_HASH__",
        "__TERMINAL_NONBASELINE_LEDGER_COUNT__",
        "__TERMINAL_NONBASELINE_LEDGER_HASH__",
    }
    assert "325e8d315a74cca89f72f14f119aa9e7759846b2" not in sql


def test_current_scorer_changes_only_admitted_baseline_namespace():
    sql = TEMPLATE.read_text()
    assert "pg_get_functiondef" in sql
    assert "lab_arena_open_scoring_v2(text,smallint,jsonb)" in sql
    assert "lab_arena_judgment_cache_source_invalid" in sql
    assert sql.count("v_definition := pg_catalog.replace") == 1
    assert "THEN ':rerun286'" in sql
    assert "THEN ':rerun292' ELSE '' END" in sql
    assert "NEW.submission_id = 'baseline-2026-09-18'" in sql
    assert "':score:rerun292'" in sql
    assert "lab_arena_open_sep18_baseline_scoring" not in sql


def test_prepare_is_one_private_atomic_archive_and_preserves_activated_authority():
    sql = TEMPLATE.read_text()
    assert sql.count("CREATE TABLE") == 0
    assert "IN ACCESS EXCLUSIVE MODE" in sql
    assert sql.count("DISABLE TRIGGER USER") == 4
    assert sql.count("ENABLE TRIGGER USER") == 4
    assert "arena-2026-09-18-rerun291archive" in sql
    assert "authorized_sep18_rerun291_baseline_archive" in sql
    assert "'mode', 'shadow', 'rewards_enabled', FALSE" in sql
    assert "assignment_id LIKE '%:rerun292'" in sql
    assert "assignment_id NOT LIKE '%:score:rerun292'" in sql
    for field in (
        "reward_basis_hash", "reward_basis_doc", "signing_key_doc",
        "effective_reward_epoch", "reward_activated_at", "king_outcome",
        "king_hotkey", "king_start_epoch", "promotion_doc",
        "baseline_promoted_at", "champion_funding_frozen",
        "champion_submission_id", "champion_hotkey",
        "champion_fallback_providers",
    ):
        assert field in sql
    assert "inflight_calls" in sql and "success_unresolved_calls" in sql
    assert "final_score')::NUMERIC > 0" not in sql
    assert "source_sha256', p_source_sha256" in sql
    assert "source_commit', p_source_commit" in sql
    assert "FROM PUBLIC, anon, authenticated, service_role, lab_arena_service" in sql
    assert ") TO lab_arena_service;" in sql


def test_no_sep17_archive_chain_and_all_nonbaseline_evidence_remains_sealed():
    sql = TEMPLATE.read_text()
    assert "recovery284" not in sql
    assert "rerun284archive" not in sql
    assert "arena-2026-09-17-rerun285archive" in sql
    assert "lab_arena_sep18_published_rerun292_nonbaseline_valid_v1()" in sql
    assert sql.count(
        "OR public.lab_arena_sep18_published_rerun292_nonbaseline_valid_v1()"
    ) == 1  # terminal seal applies only at the archive/reopen boundary
    assert "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__" in sql
    assert "__TERMINAL_NONBASELINE_RUN_COUNT__" in sql
    assert "__TERMINAL_NONBASELINE_LEDGER_COUNT__" in sql


def test_natural_source_optional_hashes_conflict_but_need_not_exist():
    sql = TEMPLATE.read_text()
    assert "NOT submission_doc ? 'source_sha256'" in sql
    assert "NOT submission_doc ? 'source_commit'" in sql
    assert "6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91" in sql
    assert "successful_calls_per_icp_v1" in sql
    assert "execution_icp_cap_microusd" in sql


def test_publication_guard_seals_only_winner_identity_not_new_scores_or_source():
    sql = TEMPLATE.read_text()
    for field in (
        "outcome", "king_submission_id", "king_hotkey", "winner_submission_id",
    ):
        assert (
            f"NEW.publication_doc #>> '{{king_decision,{field}}}' IS DISTINCT FROM"
            in sql
        )
    guard = sql[sql.index(
        "CREATE OR REPLACE FUNCTION public.lab_arena_sep18_published_rerun292_publication_guard_v1"
    ):]
    assert "NEW.publication_doc IS DISTINCT FROM" not in guard
    assert "NEW.publication_doc #>> '{final_ranking" not in guard
    assert "NEW_SOURCE_SHA256" not in guard
    assert "NEW.configuration_doc ->> 'baseline_source_url' IS DISTINCT FROM" in guard
    assert "NEW.configuration_doc - 'schedule' - 'baseline_source_url'" in guard
    assert "leadpoet-sales-agent/archive/refs/heads/lab.tar.gz" in guard


def test_prepare_starts_each_active_icp_with_an_independent_zero_cost_state():
    sql = TEMPLATE.read_text()
    assert "pg_catalog.generate_series(0, 19)" in sql
    assert "lab_arena__successful_icp_cost_state(" in sql
    assert "success_unresolved_microusd" in sql


def test_unrelated_row_preservation_uses_compact_count_and_digest_summaries():
    sql = TEMPLATE.read_text()
    start = sql.index("-- Preserve every unrelated round")
    end = sql.index("  v_expected_round :=", start)
    protected = sql[start:end]
    assert protected.count(
        "'count', pg_catalog.count(*), 'sha256', 'sha256:'"
    ) == 10
    assert "jsonb_agg" not in protected


def test_ledger_preservation_scan_is_bounded_to_neighbor_rounds():
    sql = TEMPLATE.read_text()
    bounded = (
        "WHERE round_id IN (v_round_id, 'arena-2026-09-19',\n"
        "                         'arena-2026-09-17-rerun285archive')"
    )
    assert sql.count(bounded) == 2
    start = sql.index("-- Preserve every unrelated round")
    end = sql.index("  v_expected_round :=", start)
    protected = sql[start:end]
    assert "FROM public.lab_arena_ledger AS row_value\n      WHERE NOT (" not in protected
