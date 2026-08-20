from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "157-research-lab-routing-experiment-authority.sql"
PURPOSE_MIGRATION = ROOT / "scripts" / "158-research-lab-routing-experiment-purposes.sql"
TRANSITION_MIGRATION = ROOT / "scripts" / "161-research-lab-exact-model-transitions.sql"
BEHAVIOR = ROOT / "tests" / "sql" / "test_routing_experiment_authority_v2.sql"


def test_routing_authority_migration_has_fenced_append_only_security_contract():
    sql = MIGRATION.read_text()
    assert "research_lab_routing_append_fenced_event_v2" in sql
    assert "research_lab_routing_assert_claim_v3" in sql
    assert "research_lab_routing_claim_experiment_v3" in sql
    assert "research_lab_routing_recover_claim_v3" in sql
    assert "research_lab_routing_experiment_claims_v3" in sql
    assert "research_lab_routing_append_fenced_event_v3" in sql
    assert "research_lab_routing_event_claim_fence_required" not in sql
    assert "claim_generation BIGINT NOT NULL" in sql
    assert "request_hash TEXT NOT NULL" in sql
    assert "lease_hash TEXT NOT NULL" in sql
    assert "lease_generation BIGINT NOT NULL" in sql
    assert "research_lab_routing_renew_claim_v3" in sql
    assert "research_lab_routing_close_claim_v3" in sql
    assert "research_lab_routing_experiment_claim_heartbeats_v3" in sql
    assert "effective_expiry" in sql
    assert "research_lab_routing_experiment_claim_closures_v3" in sql
    assert "billing_state', 'uncertain'" in sql
    # The active v3 claim authority has no bearer-capability argument.  Legacy
    # v2 definitions remain only as non-destructively retired replay shims.
    claim_v3_start = sql.index(
        "CREATE OR REPLACE FUNCTION public.research_lab_routing_claim_experiment_v3"
    )
    claim_v3_end = sql.index(
        "CREATE OR REPLACE FUNCTION public.research_lab_routing_recover_claim_v3",
        claim_v3_start,
    )
    assert "p_claim_capability TEXT" not in sql[claim_v3_start:claim_v3_end]
    for active_name, next_name in (
        (
            "research_lab_routing_claim_experiment_v3",
            "research_lab_routing_recover_claim_v3",
        ),
        (
            "research_lab_routing_recover_claim_v3",
            "research_lab_routing_renew_claim_v3",
        ),
        (
            "research_lab_routing_append_provider_attempt_v3",
            "REVOKE ALL ON FUNCTION",
        ),
    ):
        active_start = sql.index(
            f"CREATE OR REPLACE FUNCTION public.{active_name}"
        )
        active_end = sql.index(next_name, active_start + 20)
        assert "claim_capability" not in sql[active_start:active_end]
    assert "Retire bearer-capability claim entry points" in sql
    assert "FROM PUBLIC, anon, authenticated, service_role" in sql
    assert "reservation_id TEXT NOT NULL" in sql
    assert "action_id TEXT NOT NULL" in sql
    assert "research_lab_routing_attempt_reservation_chain_mismatch" in sql
    assert "budget_head.event_type IS DISTINCT FROM 'settle'" in sql
    assert "rl_route_reservation_global_uq" in sql
    assert "research_lab_routing_evaluation_receipt_set_is_not_complete" in sql
    assert "research_lab_routing_promote_event_missing" in sql
    assert "research_lab_routing_recover_claim_event_missing" in sql
    assert "research_lab_routing_list_unresolved_budget_reservations_v2" in sql
    assert "provider_dispatch_started" in sql
    assert "research_lab_routing_dispatch_event_v3_reservation_mismatch" in sql
    assert "reserve_event.event_doc = p_event_doc" in sql
    assert "budget_head.event_type <> 'settle'" in sql
    assert "client[_-]?secret" in sql
    assert "private[_-]?key" in sql
    assert "service[_-]?role" in sql
    assert "REVOKE TRUNCATE" in sql
    assert "FROM PUBLIC, anon, authenticated, service_role;" in sql
    assert "GRANT SELECT ON TABLE" in sql
    assert "DROP TRIGGER IF EXISTS" in sql
    assert "SET search_path = pg_catalog, public" in sql
    assert "research_lab_routing_canonical_jsonb_v2" in sql
    assert "research_lab_routing_jsonb_hash_v2(p_spec_doc)" in sql
    assert "research_lab_routing_jsonb_hash_v2(p_execution_envelope_doc)" in sql
    assert "claim_state = 'recovered'" in sql
    assert "claim_recovered_unknown_billing" in sql
    assert "model_binding_observation_receipt_hash" in sql
    # Provider attempts carry explicit, redacted links to all three signed
    # execution receipts and both terminal projections.  The SQL RPC must
    # receive and compare these values instead of trusting nested JSON only.
    for field in (
        "terminal_receipt_hash",
        "protected_release_receipt_hash",
        "admission_bundle_hash",
        "terminal_provider_record_hash",
        "terminal_billing_projection_hash",
    ):
        assert field in sql
    assert "research_lab_routing_assert_provider_receipt_chain_v2" in sql
    assert "research_lab_routing_assert_provider_receipt_chain_v3" in sql
    assert "research_lab_routing_append_provider_attempt_v3" in sql
    assert "routing_provider_attempt.v3" in sql
    for function_name in (
        "research_lab_routing_append_decision_receipt_v3",
        "research_lab_routing_append_evaluation_v3",
        "research_lab_routing_reserve_budget_v3",
        "research_lab_routing_settle_budget_v3",
        "research_lab_routing_mark_budget_uncertain_v3",
        "research_lab_routing_recover_budget_v3",
        "research_lab_routing_list_expired_budget_reservations_v3",
        "research_lab_routing_list_unresolved_budget_reservations_v3",
        "research_lab_routing_assert_promotion_receipt_chain_v3",
        "research_lab_routing_assert_promotion_reconciliation_v3",
        "research_lab_routing_promote_v3",
    ):
        assert f"CREATE OR REPLACE FUNCTION public.{function_name}" in sql
    assert "authorization_request_hash" in sql
    assert "authorization_job_id" in sql
    assert "admission_job_id" in sql
    assert "model_binding_observation_receipt_hash" in sql
    assert "parent_receipt_hashes" in sql
    assert "receipt.receipt_doc = p_attempt_doc->'call_grant_receipt'" in sql
    assert "receipt.receipt_doc = p_attempt_doc->'protected_release_receipt'" in sql
    assert "receipt.receipt_doc = terminal_doc" in sql
    assert "terminal_result" in sql
    assert "terminal_request_hash" in sql
    assert "jsonb_hash_v2(terminal_result)" in sql
    assert "leadpoet.routing_provider_dispatch_job.v3" in sql
    assert "expected_terminal_job_id := 'routing-dispatch:'" in sql
    assert "terminal_job_id IS DISTINCT FROM expected_terminal_job_id" in sql
    assert "leadpoet.research_lab.routing_claim_fence.v3" in sql
    assert "leadpoet.research_lab.routing_budget_reservation_proof.v3" in sql
    assert "leadpoet.research_lab.routing_budget_reservation_result.v3" in sql
    assert "budget_reservation->>'response_hash' NOT IN" in sql
    assert "research_lab_routing_attempt_v3_reservation_proof_mismatch" in sql
    for reservation_binding in (
        "budget_reservation->>'experiment_hash' IS DISTINCT FROM p_experiment_hash",
        "budget_reservation->>'binding_id' IS DISTINCT FROM p_binding_id",
        "budget_reservation->>'claim_key' IS DISTINCT FROM reserve_event.claim_key",
        "budget_reservation->>'event_key'",
        "budget_reservation->>'reservation_id'",
    ):
        assert reservation_binding in sql
    assert "jsonb_object_keys(budget_reservation)" in sql
    assert "terminal_result->>'transport_attempt_hash'" in sql
    assert "budget_reservation->>'transport_attempt_hash'" in sql
    assert not any(
        comparison in sql
        for comparison in (
            "terminal_result->>'transport_attempt_hash' IS DISTINCT FROM budget_reservation->>'transport_attempt_hash'",
            "budget_reservation->>'transport_attempt_hash' IS DISTINCT FROM terminal_result->>'transport_attempt_hash'",
            "terminal_result->>'transport_attempt_hash' = budget_reservation->>'transport_attempt_hash'",
            "budget_reservation->>'transport_attempt_hash' = terminal_result->>'transport_attempt_hash'",
        )
    )
    assert "GRANT EXECUTE ON FUNCTION public.research_lab_routing_assert_provider_receipt_chain_v3" in sql
    assert "research_lab_routing_assert_promotion_receipt_chain_v2" in sql
    assert "research_lab_routing_promote_receipt_chain_missing" in sql
    # Promotion must independently rebuild the three Python authority roots
    # from durable rows and bind the signed attestation output to them.
    assert "research_lab_routing_promote_v3_durable_root_mismatch" in sql
    assert "research_lab_routing_promote_v3_attestation_output_mismatch" in sql
    assert "research_lab_routing_promote_v3_attestation_ancestry_mismatch" in sql
    for projected_field in (
        "decision_doc",
        "provider_receipt_ref",
        "binding_catalog_manifest_hash",
        "authorization_proof_hash",
        "request_fingerprint",
        "authoritative_billed_credit_microunits",
        "event_doc",
    ):
        assert projected_field in sql
    assert "expected_output_root := public.research_lab_routing_jsonb_hash_v2(expected_output)" in sql
    assert "'input_root', p_reconciliation_doc->>'authority_input_root'" in sql
    assert "receipt.receipt_doc->>'input_root' = receipt.input_root" in sql
    assert (
        "REVOKE ALL ON FUNCTION public.research_lab_routing_assert_promotion_reconciliation_v3"
        in sql
    )
    helper_start = sql.index(
        "CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_promotion_reconciliation_v3"
    )
    helper_end = sql.index(
        "CREATE OR REPLACE FUNCTION public.research_lab_routing_promote_v3",
        helper_start,
    )
    helper = sql[helper_start:helper_end]
    assert "SECURITY DEFINER" in helper
    assert "SET search_path = pg_catalog, public" in helper
    assert "pg_advisory_xact_lock" in helper
    # The native behavior script must remain the place for positive and
    # adversarial receipt-chain calls; keep a visible marker so it cannot be
    # dropped when the fixtures are refreshed.
    behavior = BEHAVIOR.read_text()
    assert "receipt_chain" in behavior
    assert "receipt_chain_v3_full_path" in behavior
    assert "exact v3 terminal attempt replay was not idempotent" in behavior
    assert "forged budget reservation response hash unexpectedly succeeded" in behavior
    assert "v3 provider dispatch marker did not append" in behavior
    assert "forged v3 provider dispatch marker unexpectedly succeeded" in behavior
    assert "heartbeat-extended v3 claim was treated as stale" in behavior
    assert "function_acl_cutover" in behavior
    assert "promotion reconciliation helper is directly callable" in behavior
    # Boundary-aware body checks permit hash-only provenance fields while
    # retaining rejection of raw request/response bodies.
    assert "response[_-]?body([^[:alnum:]_]|$)" in sql
    assert "request[_-]?body([^[:alnum:]_]|$)" in sql
    assert "credential|service[_-]?role" in sql
    assert (
        "authorization_hash|authorization_proof_hash|authorization_request_hash|request_body_hash|response_body_hash|claim_fence_hash"
        in sql
    )
    assert "claim_nonce_commitment|claim_capability_commitment" not in sql
    assert sql.count("ADD COLUMN IF NOT EXISTS claim_capability_commitment") == 3
    assert '"(authorization_job_id|job_id)"' in sql
    assert "sha256:[0-9a-f]{64}" in sql


def test_routing_authority_purpose_migration_is_exact_and_replay_safe():
    sql = PURPOSE_MIGRATION.read_text()
    assert "DROP CONSTRAINT IF EXISTS" in sql
    assert "research_lab.routing_experiment.v2" in sql
    assert "research_lab.routing_provider_evidence.v2" in sql
    assert "research_lab.routing_model_binding_observation.v2" in sql
    assert "role = 'gateway_scoring'" in sql
    assert "NOT VALID" in sql
    assert "VALIDATE CONSTRAINT" in sql


def test_exact_model_transition_migration_is_redacted_and_retires_v2_mutations():
    sql = TRANSITION_MIGRATION.read_text()
    assert "model_transition_completed" in sql
    assert "research_lab_routing_experiment_events_v2_event_type_check" in sql
    assert "VALIDATE CONSTRAINT" in sql
    assert "leadpoet.research_lab.model_transition.v1" in sql
    assert "provider_response_sha256" in sql
    assert "provider_response'" not in sql
    assert "p_event_doc->'provider_receipt'" in sql
    assert "jsonb_object_length" not in sql
    assert "jsonb_object_keys(p_event_doc)" in sql
    assert ") <> 13" in sql
    for replay_field in (
        "protected_dispatch_job_id",
        "terminal_receipt_hash",
        "model_completion_contract_hash",
        "model_provider_response_sha256",
    ):
        assert replay_field in sql
    assert (
        "attempt.attempt_doc->'terminal_result'->>'model_provider_response_sha256'"
        in sql
    )
    assert sql.count("REVOKE ALL ON FUNCTION public.research_lab_routing_") == 12
    for preserved in (
        "research_lab_routing_submit_experiment_v2",
        "research_lab_routing_request_execution_v2",
        "research_lab_routing_recover_claim_v2",
        "research_lab_routing_claim_execution_requests_v2",
        "research_lab_routing_claim_execution_v3",
    ):
        assert f"REVOKE ALL ON FUNCTION public.{preserved}" not in sql
    behavior = BEHAVIOR.read_text()
    assert "exact Model transition did not append" in behavior
    assert "exact Model transition replay was not idempotent" in behavior
    assert "forged exact Model transition unexpectedly succeeded" in behavior


@pytest.mark.skipif(
    not os.getenv("ROUTING_EXPERIMENT_TEST_PG_DSN"),
    reason="set ROUTING_EXPERIMENT_TEST_PG_DSN for disposable PostgreSQL behavior test",
)
def test_routing_authority_disposable_postgres_behavior():
    psql = shutil.which("psql")
    if not psql:
        pytest.skip("psql is unavailable")
    result = subprocess.run(
        [
            psql,
            os.environ["ROUTING_EXPERIMENT_TEST_PG_DSN"],
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            str(BEHAVIOR),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
