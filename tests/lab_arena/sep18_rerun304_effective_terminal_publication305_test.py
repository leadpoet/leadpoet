"""Public contract for the narrow rerun304 effective-terminal guard repair."""
from pathlib import Path
import hashlib


SQL = (
    Path(__file__).parents[2]
    / "scripts/305-arena-2026-09-18-rerun304-effective-terminal-publication.sql"
)


def test_migration_is_sealed_and_only_replaces_the_publication_guard():
    body = SQL.read_text()
    assert hashlib.sha256(SQL.read_bytes()).hexdigest() == (
        "f841feeaafb8763b342dae7c896be8f0078dbaba73ba0bcc683c737533a735ab"
    )
    assert body.count(
        "CREATE OR REPLACE FUNCTION public."
        "lab_arena_sep18_newjudge_rerun304_publication_guard_v1()"
    ) == 1
    assert "ALTER FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1() OWNER TO lab_arena_owner" in body
    assert "CREATE TABLE" not in body
    assert "ALTER TABLE" not in body
    assert "INSERT INTO" not in body
    assert "UPDATE public." not in body
    assert "DELETE FROM" not in body
    assert "lab_arena_prepare_sep18" not in body


def test_guard_requires_exact_effective_positions_and_current_zero_policy():
    body = SQL.read_text()
    assert "count(*) FROM effective)<>100" in body
    assert "count(DISTINCT submission_id||':'||icp_position::TEXT) FROM effective)<>100" in body
    assert "work_count+zero_count<>100" in body
    assert "w.submission_id<>'baseline-2026-09-18')<>80" in body
    for cause in (
        "model_timeout", "invalid_output", "budget_exhausted",
        "credential_error", "model_error",
    ):
        assert cause in body
    assert "provider_error" not in body
    assert "latest.per_icp_score IS DISTINCT FROM 0::DOUBLE PRECISION" in body
    assert "accepted.status='accepted'" in body


def test_guard_binds_each_accepted_output_to_one_fresh_judgment():
    body = SQL.read_text()
    assert "w.scored_run_id=e.run_id" in body
    assert "NOT EXISTS(SELECT 1 FROM work w WHERE w.scored_run_id=e.run_id)" in body
    assert "s.scored_run_id=w.scored_run_id" in body
    assert ":score:rerun304" in body
    assert "sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c" in body
    assert "count(DISTINCT s.scored_run_id)" in body
    assert "<>(SELECT pg_catalog.count(*) FROM work)" in body
    assert "status NOT IN('accepted','failed')" in body


def test_source_archive_cost_and_reward_authority_guards_remain():
    body = SQL.read_text()
    assert "lab_arena_sep18_rerun303_archive_valid304_v1() IS NOT TRUE" in body
    assert "lab_arena_sep18_newjudge_rerun304_active_valid_v1() IS NOT TRUE" in body
    assert "lab_arena__successful_call_cost_state" in body
    assert "OR unsettled<>0" in body
    for field in (
        "reward_basis_hash", "reward_basis_doc", "signing_key_doc",
        "effective_reward_epoch", "reward_activated_at", "king_outcome",
        "champion_funding_frozen", "winner_submission_id",
    ):
        assert field in body


def test_old_all_accepted_only_predicates_are_removed():
    body = SQL.read_text()
    assert "status='accepted' AND terminal_cause='accepted')<>100" not in body
    assert "assignment_id LIKE '%:score:rerun304')<>100" not in body
    assert "NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id=e.round_id" not in body
