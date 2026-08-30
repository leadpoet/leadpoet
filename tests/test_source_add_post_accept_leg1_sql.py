"""Fail-closed checks for post-acceptance SOURCE_ADD Leg 1."""

from pathlib import Path

from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/167-research-lab-source-add-post-accept-leg1.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")


def test_post_accept_leg1_migration_is_additive_and_transactional() -> None:
    assert SQL.startswith("-- Create SOURCE_ADD Leg 1 only after accepted")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "SET LOCAL lock_timeout = '5s'" in SQL
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()
    assert "UPDATE public.research_lab_source_add_reward_obligations" not in SQL


def test_leg1_requires_acceptance_provisioning_and_both_receipt_parents() -> None:
    for marker in (
        "trg_source_add_leg1_intent_after_acceptance",
        "trg_source_add_leg1_work_after_acceptance",
        "trg_source_add_leg1_catalog_binding",
        "accepted.stage = 'accepted'",
        "accepted.precheck_status = 'provenance_precheck_passed'",
        "provision.provision_status = 'provisioned_autoresearch_eligible'",
        "smoke.evaluation_mode = 'provisioning_smoke'",
        "smoke.result_status = 'passed'",
        "edge.parent_receipt_hash = functional.receipt_hash",
        "edge.parent_receipt_hash = smoke.receipt_hash",
        "NEW.catalog_id := v_catalog_id",
    ):
        assert marker in SQL


def test_post_accept_finalizer_is_restart_gated_and_service_role_only() -> None:
    required = {
        function
        for migration, function in REQUIRED_SUPABASE_V2_RPCS
        if migration == MIGRATION
    }
    assert required == {
        "research_lab_source_add_finalize_provision_smoke_v2",
        "research_lab_source_add_post_accept_leg1_contract_v1",
    }
    assert "FROM PUBLIC, anon, authenticated" in SQL
    assert "TO service_role" in SQL
    assert "NOTIFY pgrst, 'reload schema'" in SQL
