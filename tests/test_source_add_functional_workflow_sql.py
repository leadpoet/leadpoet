"""Structural fail-closed checks for the SOURCE_ADD functional workflow migration."""

from pathlib import Path
import re

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES


ROOT = Path(__file__).resolve().parents[1]
SQL = (
    ROOT / "scripts" / "96-research-lab-source-add-functional-workflow.sql"
).read_text(encoding="utf-8")


def _function(name: str, next_name: str) -> str:
    return SQL.split(f"FUNCTION public.{name}", 1)[1].split(
        f"FUNCTION public.{next_name}", 1
    )[0]


def test_migration_is_transactional_additive_and_disabled_by_default():
    assert SQL.startswith("-- SOURCE_ADD measured functional workflow")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "migration_96_disabled_by_default" in SQL
    assert "paused BOOLEAN NOT NULL DEFAULT TRUE" in SQL
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()


def test_atomic_admission_reserves_identity_limits_hotkey_and_creates_work():
    section = _function(
        "research_lab_source_add_admit",
        "research_lab_source_add_begin_provider_execution",
    )
    assert "source-add-identity:" in section
    assert "source-add-hotkey:" in section
    assert section.count("pg_advisory_xact_lock") >= 2
    assert "p_max_open" in section
    assert "p_max_day" in section
    assert "p_max_30d" in section
    assert "v_start_seq INTEGER := 0" in section
    assert "credential_envelope" in section
    assert "credential_policy', '') <> 'no_credentials'" in section
    assert "'submitted', v_start_seq" in section
    assert "'manifest_validated', v_start_seq + 1" in section
    assert "'provenance_queued', v_start_seq + 2" in section
    assert "'provenance', 'queued'" in section
    assert "admission_kind', 'miner_submission'" in section
    assert "v_existing_terminal" in section
    assert "rejected_precheck" in section


def test_queue_claim_is_fifo_skip_locked_and_restart_safe():
    section = _function(
        "research_lab_source_add_claim_work", "research_lab_source_add_finish_work"
    )
    assert "FOR UPDATE SKIP LOCKED" in section
    assert "ORDER BY w.priority ASC, w.available_at ASC, w.created_at ASC, w.work_id ASC" in section
    assert "w.work_status = 'leased' AND w.lease_expires_at <= NOW()" in section
    assert "WHEN v_row.work_status = 'leased' THEN attempt_count" in section
    assert "active.job_doc->>'host_hash' = w.job_doc->>'host_hash'" in section
    assert "source-add-host:" in section
    assert "'functional_probe', 'provisioning_smoke'" in section


def test_provider_execution_fence_prevents_ambiguous_restart_replay():
    begin = _function(
        "research_lab_source_add_begin_provider_execution",
        "research_lab_source_add_claim_work",
    )
    claim = _function(
        "research_lab_source_add_claim_work", "research_lab_source_add_finish_work"
    )
    finish = _function(
        "research_lab_source_add_finish_work",
        "research_lab_source_add_configure_probe",
    )
    assert "FOR UPDATE" in begin
    assert "v_work.lease_token IS DISTINCT FROM p_lease_token" in begin
    assert "provider_execution_state', 'started'" in begin
    assert "provider_execution_attempt', attempt_count" in begin
    assert "uncertain_after_lease_expiry" in claim
    assert "v_row.work_status = 'leased'" in claim
    assert "v_row.job_doc->>'provider_execution_state' = 'started'" in claim
    assert "job_doc = job_doc" in finish
    assert "- 'provider_execution_state'" in finish
    assert "- 'provider_execution_recovery'" in finish


def test_finish_work_persists_exact_probe_then_queues_one_reward_intent():
    section = _function(
        "research_lab_source_add_finish_work",
        "research_lab_source_add_configure_probe",
    )
    assert "SOURCE_ADD functional attempt binding is invalid" in section
    assert "ON CONFLICT (work_id, attempt_number) DO NOTHING" in section
    assert "SOURCE_ADD functional attempt idempotency differs" in section
    assert "ON CONFLICT (adapter_id, leg) DO NOTHING" in section
    assert "SOURCE_ADD reward intent idempotency differs" in section
    assert "'provenance_precheck_passed'" in section
    assert "'functional_probe_passed'" in section
    assert "'leg1_queued'" in section
    assert "'released', v_identity.seq + 1, 'terminal_rejection'" in section


def test_current_probe_views_prefer_new_configuration_over_old_retry_count():
    assert (
        "ORDER BY submission_id, created_at DESC, attempt_number DESC, attempt_ref DESC"
        in SQL
    )
    assert SQL.count(
        "ORDER BY submission_id, created_at DESC, attempt_number DESC, attempt_ref DESC"
    ) == 2


def test_leg1_slot_counts_legacy_and_functional_rewards_with_fifo_overflow():
    section = _function(
        "research_lab_source_add_reserve_leg1_slot",
        "research_lab_source_add_finalize_leg1",
    )
    assert "source-add-leg1-day:" in section
    assert "leg1_provenance_precheck_passed" in section
    assert "leg1_functional_probe_passed" in section
    assert "v_created + v_reserved >= p_daily_cap" in section
    assert "'daily_cap_fifo'" in section
    assert "(v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'" in section
    assert "ORDER BY number LIMIT 1" in section
    assert (
        "WHERE intent_id = p_intent_id AND slot_status = 'reserved'" in section
    )


def test_leg1_finalization_binds_current_slot_functional_receipt_and_null_catalog():
    section = _function(
        "research_lab_source_add_finalize_leg1",
        "research_lab_source_add_finalize_provision",
    )
    assert "slot_status = 'reserved'" in section
    assert "lease_token = p_slot_lease_token" in section
    assert "research_lab.source_add_functional_probe.v2" in section
    assert "source_add_functional_probe" in section
    assert "research_lab.reward_decision.v2" in section
    assert "source_add_reward_decision" in section
    assert "decision_receipt_hash" in section
    assert "decision_artifact_hash" in section
    assert "functional_probe_result_hash" in section
    assert "leg1_functional_probe_passed" in section
    assert "v_intent.adapter_id, NULL, v_intent.miner_hotkey" in section
    assert "'leg1_created'" in section
    assert (
        "VALUES (p_reward->>'reward_ref', 0, p_reward->>'state', "
        "'leg1_functional_probe_passed')"
    ) in section


def test_provisioning_requires_functional_and_exact_smoke_receipts():
    section = SQL.split(
        "FUNCTION public.research_lab_source_add_finalize_provision", 1
    )[1].split("-- The original V2 migration", 1)[0]
    assert "functional_probe_required" in section
    assert "current_probe_config_required" in section
    assert "provision_config_differs_from_test" in section
    assert "provision_doc'->'request_headers'" in section
    assert section.count("jsonb_array_elements(v_config.probe_doc->'probes')") >= 2
    assert section.count(
        "jsonb_array_elements(\n                  p_provision_row->'provision_doc'->'probe_endpoints'"
    ) >= 1
    assert "smoke_test_required" in section
    assert "source_add_provisioning_smoke" in section
    assert "provisioned_autoresearch_eligible" in section
    assert "measured_trial_yield" in section


def test_provisioning_smoke_is_durable_queue_work_and_lease_bound_at_finalize():
    enqueue = _function(
        "research_lab_source_add_enqueue_provision_smoke",
        "research_lab_source_add_finalize_provision",
    )
    finalize = _function(
        "research_lab_source_add_finalize_provision_smoke",
        "prevent_research_lab_source_add_identity_mutation",
    )
    assert "approved_pending_provision" in enqueue
    assert "current_probe_config_required" in enqueue
    assert "'provisioning_smoke', 'queued'" in enqueue
    assert "SOURCE_ADD provisioning smoke idempotency differs" in enqueue
    assert "FOR UPDATE" in finalize
    assert "v_work.work_status <> 'leased'" in finalize
    assert "v_work.lease_token IS DISTINCT FROM p_lease_token" in finalize
    assert "SOURCE_ADD provisioning smoke lease binding differs" in finalize
    assert "research_lab_source_add_finalize_provision(" in finalize
    assert "work_status = 'completed'" in finalize


def test_new_tables_and_functions_are_service_role_only_with_rls():
    tables = (
        "research_lab_source_add_identity_events",
        "research_lab_source_add_probe_config_events",
        "research_lab_source_add_functional_probe_attempts",
        "research_lab_source_add_work_items",
        "research_lab_source_add_reward_intents",
        "research_lab_source_add_reward_slots",
        "research_lab_source_add_control",
    )
    for table in tables:
        assert f"REVOKE ALL ON TABLE public.{table} FROM PUBLIC, anon, authenticated" in SQL
        assert f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY" in SQL
    assert "REVOKE ALL ON FUNCTION public.research_lab_source_add_admit" in SQL
    assert "GRANT EXECUTE ON FUNCTION public.research_lab_source_add_admit" in SQL
    assert "REVOKE ALL ON FUNCTION public.research_lab_source_add_begin_provider_execution" in SQL
    assert "GRANT EXECUTE ON FUNCTION public.research_lab_source_add_begin_provider_execution" in SQL
    assert "REVOKE ALL ON FUNCTION public.research_lab_source_add_enqueue_provision_smoke" in SQL
    assert "GRANT EXECUTE ON FUNCTION public.research_lab_source_add_enqueue_provision_smoke" in SQL
    assert "REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision_smoke" in SQL
    assert "GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_provision_smoke" in SQL


def test_v2_receipt_allowlist_adds_source_add_without_removing_protected_purposes():
    assert "research_lab.source_add_provenance.v2" in SQL
    assert "research_lab.source_add_functional_probe.v2" in SQL
    assert "research_lab.source_add_reward_input.v2" in SQL
    assert "research_lab.promotion_decision.v2" in SQL
    assert "research_lab.allocation.v2" in SQL
    assert "gateway.weights.publication.v2" in SQL
    assert "validator.weights.finalized.v2" in SQL


def test_v2_receipt_allowlist_exactly_matches_canonical_role_contract():
    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            SQL,
            re.DOTALL,
        )
        assert match is not None, role
        migrated_purposes = set(re.findall(r"'([^']+)'", match.group(1)))
        assert migrated_purposes == set(expected_purposes), role
