"""Public boundary checks for the sealed Sep18 scoring-only recovery.

The literal migration and complete service lifecycle are additionally exercised
against the private, hash-bound terminal302 snapshot in disposable PostgreSQL.
"""
from pathlib import Path
import re

SQL=Path(__file__).parents[2]/"scripts/303-arena-2026-09-18-cancelled-scoring-recovery.sql"

def test_recovery_keeps_source_and_uses_fresh_judgments():
    body=SQL.read_text()
    assert "SET status='stage1',status_generation" in body
    assert "stage1_scoring_plan_doc=NULL,stage2_scoring_plan_doc=NULL" in body
    assert "THEN ':rerun303'" in body
    assert "SET per_icp_score=NULL,qualification_doc=NULL" in body
    assert "archived_execution_judgments" in body
    assert "9996" in body
    assert "sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2" in body
    assert "6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91" in body
    assert "stage_1_scoring_close')::TIMESTAMPTZ<=pg_catalog.clock_timestamp()" in body
    assert "b83c135347d6ea80fb3e21ebbc5bbf992afc87a5e04ed6404a4831707ef3d03d" in body
    assert body.count("SELECT public.lab_arena_prepare_sep18_cancelled_rerun303_v1(")==1

def test_publication_allows_failed_attempt_followed_by_complete_recovery():
    body=SQL.read_text().split("CREATE OR REPLACE FUNCTION public.lab_arena_sep18_cancelled_rerun303_publication_guard_v1()",1)[1]
    assert "status NOT IN('accepted','failed')" in body
    assert "count(DISTINCT scored_run_id)" in body
    assert "status='accepted' AND terminal_cause='accepted'" in body
    assert "kind='score')<>100" not in body

def test_migration_contains_no_private_lead_payload_or_new_sourcing_work():
    body=SQL.read_text()
    assert re.search(r"__[A-Z0-9_]+__",body) is None
    for forbidden in ('company_identity_key','company_name','linkedin.com','intent_details"'+':','INSERT INTO public.lab_arena_ledger','DELETE FROM','TRUNCATE '):
        assert forbidden not in body
    assert "UPDATE public.lab_arena_runs SET round_id=" in body
    assert "WHERE round_id='arena-2026-09-18' AND kind='score';" in body
