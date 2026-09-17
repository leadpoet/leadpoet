from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "scripts/286-arena-2026-09-17-published-baseline-rerun.sql.template"


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
    assert "THEN ':rerun286' ELSE '' END" in sql
    assert "NEW.submission_id = 'baseline-2026-09-17'" in sql
    assert "':score:rerun286'" in sql
    assert "lab_arena_open_sep17_baseline_scoring" not in sql


def test_prepare_is_one_private_atomic_archive_and_preserves_reward_authority():
    sql = TEMPLATE.read_text()
    assert sql.count("CREATE TABLE") == 0
    assert "IN ACCESS EXCLUSIVE MODE" in sql
    assert sql.count("DISABLE TRIGGER USER") == 4
    assert sql.count("ENABLE TRIGGER USER") == 4
    assert "arena-2026-09-17-rerun285archive" in sql
    assert "authorized_sep17_recovery285_baseline_archive" in sql
    assert "'mode', 'shadow', 'rewards_enabled', FALSE" in sql
    assert "assignment_id LIKE '%:rerun286'" in sql
    assert "assignment_id NOT LIKE '%:score:rerun286'" in sql
    for field in (
        "reward_basis_hash", "reward_basis_doc", "signing_key_doc",
        "effective_reward_epoch", "reward_activated_at",
    ):
        assert field in sql
    assert "inflight_calls" in sql and "success_unresolved_calls" in sql
    assert "final_score')::NUMERIC > 0" not in sql
    assert "source_sha256', p_source_sha256" in sql
    assert "source_commit', p_source_commit" in sql
    assert "FROM PUBLIC, anon, authenticated, service_role, lab_arena_service" in sql
    assert ") TO lab_arena_service;" in sql


def test_prior_archive_and_nonbaseline_evidence_remain_sealed():
    sql = TEMPLATE.read_text()
    assert sql.count("WHERE round_id IN (v_round_id, 'arena-2026-09-18',") == 2
    assert sql.count("'arena-2026-09-17-rerun284archive')") == 2
    assert "lab_arena_sep17_recovery284_archive_valid_v1()" in sql
    assert "lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()" in sql
    assert "lab_arena_sep17_rerun286_nonbaseline_valid_v1()" in sql
    assert "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__" in sql
    assert "__TERMINAL_NONBASELINE_RUN_COUNT__" in sql
    assert "__TERMINAL_NONBASELINE_LEDGER_COUNT__" in sql
