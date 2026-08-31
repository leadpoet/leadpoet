"""Fail-closed checks for the SOURCE_ADD admission-control repair."""

from pathlib import Path

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
SQL = (
    ROOT / "scripts" / "145-research-lab-source-add-admission-control.sql"
).read_text(encoding="utf-8")


def test_source_add_admission_control_is_transactional_and_idempotent():
    assert SQL.startswith("-- Linearize SOURCE_ADD miner admission")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "CREATE OR REPLACE FUNCTION" in SQL
    assert "DROP TRIGGER IF EXISTS trg_source_add_work_admission_control" in SQL
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()


def test_initial_miner_admission_and_pause_share_one_transaction_lock():
    assert SQL.count("hashtextextended('source-add-control', 0)") == 2
    assert "NEW.work_kind = 'provenance'" in SQL
    assert "NEW.job_doc->>'admission_kind' = 'miner_submission'" in SQL
    assert "COALESCE((" in SQL
    assert "), TRUE)" in SQL
    assert "RAISE EXCEPTION 'SOURCE_ADD admission is paused'" in SQL
    assert "BEFORE INSERT ON public.research_lab_source_add_work_items" in SQL


def test_operator_rechecks_are_not_disabled_and_missing_control_fails_closed():
    assert "operator provenance rechecks remain" in SQL.lower()
    assert "IF NOT FOUND THEN" in SQL
    assert "RAISE EXCEPTION 'SOURCE_ADD control row is unavailable'" in SQL


def test_control_functions_remain_service_role_only():
    assert (
        "REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_admission_control()"
        in SQL
    )
    assert (
        "REVOKE ALL ON FUNCTION public.research_lab_source_add_set_paused(BOOLEAN, TEXT, TEXT)"
        in SQL
    )
    assert (
        "GRANT EXECUTE ON FUNCTION public.research_lab_source_add_set_paused(BOOLEAN, TEXT, TEXT)"
        in SQL
    )
    assert "research_lab_source_add_admission_control_contract_v1" in SQL
    assert "'control_row_present', EXISTS" in SQL
    assert "'trigger_enabled', COALESCE" in SQL
    assert "TO service_role" in SQL


def test_restart_preflight_requires_complete_source_add_v2_schema():
    migration = "scripts/96-research-lab-source-add-functional-workflow.sql"
    relations = {
        relation
        for declared_migration, relation, _columns in REQUIRED_SUPABASE_V2_SCHEMA
        if declared_migration == migration
    }
    assert relations == {
        "research_lab_source_add_submission_current",
        "research_lab_source_add_control",
        "research_lab_source_add_work_items",
        "research_lab_source_add_probe_config_current",
        "research_lab_source_add_functional_probe_current",
        "research_lab_source_add_provisioning_smoke_current",
        "research_lab_source_add_reward_intents",
        "research_lab_source_add_reward_slots",
        "research_lab_source_add_identity_current",
    }
    assert {
        relation
        for declared_migration, relation, _columns in REQUIRED_SUPABASE_V2_SCHEMA
        if declared_migration
        in {
            "scripts/72-research-lab-source-experiments.sql",
            "scripts/78-research-lab-source-add-catalog-provisioning.sql",
        }
    } >= {
        "research_lab_source_catalog",
        "research_lab_source_add_reward_current",
        "research_lab_source_add_provisioning_current",
    }
    functions = {
        function
        for declared_migration, function in REQUIRED_SUPABASE_V2_RPCS
        if declared_migration == migration
    }
    assert functions == {
        "research_lab_source_add_admit",
        "research_lab_source_add_begin_provider_execution",
        "research_lab_source_add_claim_work",
        "research_lab_source_add_finish_work",
        "research_lab_source_add_requeue_provenance",
        "research_lab_source_add_set_paused",
        "research_lab_source_add_enqueue_provision_smoke",
    }
    assert {
        function
        for declared_migration, function in REQUIRED_SUPABASE_V2_RPCS
        if declared_migration
        == "scripts/169-research-lab-source-add-post-accept-leg1.sql"
    } == {
        "research_lab_source_add_configure_probe_v2",
        "research_lab_source_add_finalize_provision_v2",
        "research_lab_source_add_reject_current_builtin_v2",
        "research_lab_source_add_reserve_leg1_slot_v2",
        "research_lab_source_add_finalize_leg1_v2",
        "research_lab_source_add_finalize_provision_smoke_v2",
        "research_lab_source_add_post_accept_leg1_contract_v1",
    }
    assert (
        "scripts/145-research-lab-source-add-admission-control.sql",
        "research_lab_source_add_admission_control_contract_v1",
    ) in REQUIRED_SUPABASE_V2_RPCS
