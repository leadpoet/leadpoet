"""Structural gates for post-accept SOURCE_ADD Leg 1 authority."""

from pathlib import Path

from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS


ROOT = Path(__file__).resolve().parents[1]
SQL = (
    ROOT / "scripts" / "169-research-lab-source-add-post-accept-leg1.sql"
).read_text(encoding="utf-8")


def test_migration_is_transactional_and_requires_a_quiet_pause():
    assert SQL.startswith("-- Make SOURCE_ADD Leg 1 contingent")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "SOURCE_ADD must be paused" in SQL
    assert "work_status = 'leased'" in SQL
    assert "SET LOCAL lock_timeout = '5s'" in SQL
    quiet = SQL.split("DO $$", 1)[1].split("$$;", 1)[0]
    control_lock = (
        "LOCK TABLE public.research_lab_source_add_control\n"
        "        IN ACCESS EXCLUSIVE MODE NOWAIT"
    )
    assert control_lock in quiet
    for table in (
        "research_lab_source_add_work_items",
        "research_lab_source_add_submissions",
        "research_lab_source_add_functional_probe_attempts",
        "research_lab_source_catalog",
        "research_lab_source_add_provisioning_events",
        "research_lab_source_add_reward_intents",
        "research_lab_source_add_reward_slots",
        "research_lab_source_add_reward_obligations",
    ):
        assert f"public.{table}" in quiet
    assert "IN SHARE ROW EXCLUSIVE MODE NOWAIT" in quiet
    assert quiet.index(control_lock) < quiet.index(
        "public.research_lab_source_add_work_items"
    )
    assert SQL.count("LOCK TABLE") == 2
    assert "pre-final acceptance requiring adjudication" in SQL
    assert "pre-accept Leg 1 intent requiring adjudication" in SQL
    assert "pre-accept Leg 1 obligation requiring adjudication" in SQL
    assert "provision.created_at <= accepted.created_at" in SQL
    assert "smoke.created_at <= accepted.created_at" in SQL
    assert "provision.created_at <= intent.created_at" not in SQL
    assert "smoke.created_at <= intent.created_at" not in SQL
    assert "accepted.created_at <= intent.created_at" not in SQL
    assert "reward.catalog_id = provision.catalog_id" in SQL
    assert "provision.created_at <= reward.created_at" in SQL
    assert "smoke.created_at <= reward.created_at" in SQL
    assert "accepted.created_at <= reward.created_at" in SQL
    assert SQL.count("accepted.stage = 'accepted'") >= 2
    preflight = SQL.split("-- Return the accepted catalog", 1)[0]
    assert "research_lab_source_add_provisioning_current" not in preflight
    assert "research_lab_source_add_provisioning_events provision" in preflight
    assert "research_lab_source_add_functional_probe_attempts smoke" in preflight
    assert "smoke.evaluation_mode = 'provisioning_smoke'" in preflight
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()


def test_pending_approval_cannot_append_final_acceptance():
    assert "enforce_research_lab_source_add_acceptance_v2" in SQL
    assert "v_provision.provision_status = 'approved_pending_provision'" in SQL
    assert "RETURN NULL;" in SQL
    assert "provisioned_autoresearch_eligible" in SQL
    assert "acceptance requires a passed provisioning smoke" in SQL
    assert "acceptance requires its Leg 1 intent" in SQL


def test_eligible_transition_and_reward_gates_are_independent():
    assert "trg_source_add_eligible_v2" in SQL
    assert "trg_source_add_leg1_work_v2" in SQL
    assert "trg_source_add_leg1_slot_v2" in SQL
    assert "trg_source_add_leg1_obligation_v2" in SQL
    assert SQL.count("research_lab_source_add_final_approval_catalog_v2") >= 5
    assert "source_add_provisioning_smoke_current" in SQL
    assert "source_add_functional_probe_current" in SQL
    assert "source_add_provisioning_current" in SQL
    assert "source_add_provisioning_smoke" in SQL
    assert "research_lab.source_add_functional_probe.v2" in SQL
    assert "source_add_functional_probe'" in SQL
    assert SQL.count("receipt.role = 'gateway_coordinator'") >= 3


def test_leg1_obligation_binds_exact_smoke_and_decision_parent_edges():
    obligation = SQL.split(
        "CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_leg1_obligation_v2()",
        1,
    )[1].split("REVOKE ALL ON FUNCTION", 1)[0]
    for field in (
        "provisioning_smoke_passed",
        "provisioning_smoke_attempt_ref",
        "provisioning_smoke_receipt_hash",
        "provisioning_smoke_business_artifact_hash",
        "provisioning_smoke_result_hash",
        "final_acceptance_stage",
        "provision_ref",
        "catalog_id",
        "registry_provider_id",
        "provision_status",
    ):
        assert f"'{field}'" in obligation
    assert "NEW.trigger_evidence_doc IS DISTINCT FROM v_expected_trigger" in obligation
    assert "research_lab_attested_business_artifact_links_v2" in obligation
    assert "research_lab_attested_receipt_edges_v2" in obligation
    assert "v_decision.receipt_doc->'parent_receipt_hashes'" in obligation
    assert "IS DISTINCT FROM v_expected_parents" in obligation
    assert "research_lab_source_add_jsonb_hash_v2(v_expected_projection)" in obligation
    assert "link.artifact_hash = v_expected_decision_hash" in obligation
    assert "reward decision ancestry differs" in obligation


def test_post_accept_finalizer_is_lease_bound_atomic_and_idempotent():
    assert "research_lab_source_add_finalize_provision_smoke_v2" in SQL
    assert "v_work.work_status <> 'leased'" in SQL
    assert "v_work.lease_token IS DISTINCT FROM p_lease_token" in SQL
    assert "SOURCE_ADD post-accept smoke lease binding differs" in SQL
    assert "'prefix', 'source_add_probe_attempt'" in SQL
    assert "v_smoke.work_id <> p_work_id" in SQL
    assert "v_smoke.attempt_number <> v_work.attempt_count" in SQL
    assert "v_smoke.business_artifact_hash <>" in SQL
    assert "SOURCE_ADD post-accept persisted smoke differs from lease" in SQL
    assert "ON CONFLICT (adapter_id, leg) DO NOTHING" in SQL
    assert "SOURCE_ADD post-accept reward intent idempotency differs" in SQL
    assert "research_lab_source_add_finalize_provision(" in SQL
    assert "SOURCE_ADD post-accept provisioning failed" in SQL
    assert "ON CONFLICT (work_id) DO NOTHING" in SQL
    assert "SOURCE_ADD post-accept reward work idempotency differs" in SQL
    assert "work_status = 'completed'" in SQL
    assert "leg1_intent_id" in SQL
    assert "leg1_work_id" in SQL


def test_terminal_provisioning_smoke_can_only_be_explicitly_requeued_safely():
    enqueue = SQL.split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_enqueue_provision_smoke(",
        1,
    )[1].split("COMMENT ON FUNCTION", 1)[0]
    assert "FOR UPDATE" in enqueue
    assert enqueue.index("'source-add-work:' || p_work_id") < enqueue.index(
        "FOR UPDATE"
    ) < enqueue.index("'source-add-submission:' || p_submission_id")
    assert "v_work.work_status = 'completed'" in enqueue
    for result_status in (
        "'failed'",
        "'manual_review'",
        "'awaiting_operator'",
        "'retryable'",
    ):
        assert result_status in enqueue
    assert "attempt.evaluation_mode = 'provisioning_smoke'" in enqueue
    assert "attempt.attempt_number = v_work.attempt_count" in enqueue
    assert "v_terminal_status = 'worker_exception_dead_letter'" in enqueue
    assert "v_terminal_status = 'current_model_catalog_unavailable'" in enqueue
    assert "attempt.result_status = 'passed'" in enqueue
    assert "'not_eligible'" not in enqueue
    assert "v_work.attempt_count < 20" in enqueue
    assert "provider_execution_outcome_unknown_after_worker_loss" not in enqueue
    assert "accepted.stage IN ('accepted', 'leg1_queued', 'leg1_created')" in enqueue
    assert "provision_status = 'provisioned_autoresearch_eligible'" in enqueue
    assert "SET work_status = 'queued'" in enqueue
    assert "job_doc = v_job_doc" in enqueue
    assert "completed_at = NULL" in enqueue
    assert "'requeued', TRUE" in enqueue
    assert "'terminal_retry_not_allowed'" in enqueue
    assert "attempt_count =" not in enqueue


def test_only_service_role_can_call_the_candidate_finalizer():
    signature = (
        "public.research_lab_source_add_finalize_provision_smoke_v2(\n"
        "    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB\n"
        ")"
    )
    assert f"REVOKE ALL ON FUNCTION {signature}" in SQL
    assert f"GRANT EXECUTE ON FUNCTION {signature}" in SQL
    assert "TO service_role" in SQL
    assert (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_finalize_provision_smoke_v2",
    ) in REQUIRED_SUPABASE_V2_RPCS


def test_v2_rpc_boundary_freezes_final_approval_and_owns_leg1_policy():
    for name in (
        "research_lab_source_add_configure_probe_v2",
        "research_lab_source_add_finalize_provision_v2",
        "research_lab_source_add_reject_current_builtin_v2",
        "research_lab_source_add_reserve_leg1_slot_v2",
        "research_lab_source_add_finalize_leg1_v2",
    ):
        assert f"CREATE OR REPLACE FUNCTION public.{name}(" in SQL
        assert f"GRANT EXECUTE ON FUNCTION public.{name}(" in SQL
    for legacy in (
        "research_lab_source_add_configure_probe",
        "research_lab_source_add_finalize_provision",
        "research_lab_source_add_reserve_leg1_slot",
        "research_lab_source_add_finalize_leg1",
        "research_lab_source_add_finalize_provision_smoke",
    ):
        assert f"REVOKE ALL ON FUNCTION public.{legacy}(" in SQL
    assert "'status', 'final_approval_frozen'" in SQL
    assert "accepted.stage = 'accepted'" in SQL
    assert "intent.intent_status = 'finalized'" in SQL
    assert "p_reward->>'state' <> 'active'" in SQL
    assert "p_reward->>'alpha_percent')::NUMERIC, 0) <> 1.0" in SQL
    assert "p_reward->>'reward_epochs')::INTEGER, 0) <> 20" in SQL
    assert SQL.count("p_work_lease_token,\n        10,") >= 1


def test_current_builtin_rejection_is_one_atomic_terminal_transition():
    rejection = SQL.split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_reject_current_builtin_v2(",
        1,
    )[1].split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_reserve_leg1_slot_v2(",
        1,
    )[0]
    assert "p_disabled_provision_row->>'provision_status' <> 'disabled'" in rejection
    assert "'functional_probe_failed'" in rejection
    assert "jsonb_build_object('status', 'not_eligible')" in rejection
    assert "research_lab_source_add_finish_work(" in rejection
    assert "research_lab_source_add_finalize_provision(" in rejection
    assert "v_smoke.evaluation_mode <> 'provisioning_smoke'" in rejection
    assert "v_smoke.result_status <> 'passed'" in rejection
    assert "SOURCE_ADD current-provider persisted smoke differs" in rejection
    assert "'status', 'not_eligible'" in rejection


def test_leg1_slot_reservation_is_server_capped_and_fifo():
    reserve = SQL.split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_reserve_leg1_slot_v2(",
        1,
    )[1].split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_leg1_v2(",
        1,
    )[0]
    assert "source-add-leg1-day:" in reserve
    assert "candidate.work_status = 'leased'" in reserve
    assert "candidate.work_status IN ('queued', 'retry_wait')" in reserve
    assert "candidate.available_at <= NOW()" in reserve
    assert "ORDER BY candidate.priority ASC" in reserve
    assert "candidate.created_at ASC" in reserve
    assert "candidate.work_id ASC" in reserve
    assert "v_oldest_work_id <> p_work_id" in reserve
    assert "'status', 'fifo_wait'" in reserve
    assert "research_lab_source_add_reserve_leg1_slot(" in reserve
    assert "p_work_lease_token,\n        10," in reserve


def test_leg1_initial_event_and_restart_contract_fail_closed():
    assert "trg_source_add_leg1_initial_event_v2" in SQL
    assert "NEW.seq <> 0" in SQL
    assert "NEW.reward_status <> 'active'" in SQL
    assert "NEW.reason <> 'leg1_functional_probe_passed'" in SQL
    assert "research_lab_source_add_post_accept_leg1_contract_v1" in SQL
    for trigger in (
        "trg_source_add_acceptance_v2",
        "trg_source_add_eligible_v2",
        "trg_source_add_leg1_work_v2",
        "trg_source_add_leg1_slot_v2",
        "trg_source_add_leg1_obligation_v2",
        "trg_source_add_leg1_initial_event_v2",
    ):
        assert f"trigger_row.tgname = '{trigger}'" in SQL
    assert "'daily_cap', 10" in SQL
    assert "'leg1_alpha_percent', 1.0" in SQL
    assert "'leg1_reward_epochs', 20" in SQL
    assert "'legacy_not_callable'" in SQL
