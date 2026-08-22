-- Admit the exact attestation purposes used by measured routing experiments.
-- This is append-only schema policy: it changes no receipt rows.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_attested_execution_receipts_v2
    DROP CONSTRAINT IF EXISTS
        research_lab_attested_execution_receipts_v2_role_purpose_check;

ALTER TABLE public.research_lab_attested_execution_receipts_v2
    ADD CONSTRAINT research_lab_attested_execution_receipts_v2_role_purpose_check
    CHECK (
        (role = 'gateway_coordinator' AND purpose IN (
            'research_lab.admission.v2',
            'research_lab.provider_evidence.v2',
            'research_lab.provider_outcome_snapshot.v2',
            'research_lab.provider_outcome_state.v2',
            'research_lab.active_private_model.v2',
            'leadpoet.artifact_persistence.v2',
            'research_lab.ranking.v2',
            'research_lab.promotion_decision.v2',
            'research_lab.reward_decision.v2',
            'research_lab.legacy_finalized_allocation.v2',
            'research_lab.chain_weight_observation.v1',
            'research_lab.chain_realized_epoch_settlement.v1',
            'research_lab.subnet_epoch_cutover.v2',
            'research_lab.source_add_provenance.v2',
            'research_lab.source_add_functional_probe.v2',
            'research_lab.source_add_catalog_snapshot.v2',
            'research_lab.source_add_credential.v2',
            'research_lab.openrouter_credential.v2',
            'research_lab.openrouter_credit_preflight.v2',
            'research_lab.allocation.v2',
            'research_lab.champion_input.v2',
            'research_lab.reimbursement_input.v2',
            'research_lab.source_add_reward_input.v2',
            'research_lab.sourcing_input.v2',
            'research_lab.fulfillment_input.v2',
            'research_lab.leaderboard_input.v2',
            'research_lab.ban_input.v2',
            'research_lab.anomaly_adjustment_input.v2',
            'research_lab.ancestry_checkpoint_bootstrap.v2',
            'research_lab.allocation_settlement_frontier_bootstrap.v2',
            'gateway.weights.publication.v2'
        )) OR
        (role = 'gateway_scoring' AND purpose IN (
            'research_lab.private_model_run.v2',
            'research_lab.candidate_model_run.v2',
            'research_lab.provider_evidence_tape.v2',
            'research_lab.routing_experiment.v2',
            'research_lab.routing_provider_evidence.v2',
            'research_lab.candidate_test.v2',
            'research_lab.candidate_hybrid_test.v2',
            'research_lab.candidate_hybrid_discovery.v2',
            'research_lab.model_compatibility.v2',
            'research_lab.routing_model_binding_observation.v2',
            'research_lab.company_score.v2',
            'research_lab.provider_preflight.v2',
            'research_lab.candidate_score.v2',
            'research_lab.baseline_score.v2',
            'research_lab.benchmark.v2',
            'research_lab.rebenchmark.v2',
            'research_lab.confirmation_score.v2',
            'research_lab.source_add_judge.v2',
            'qualification.lead_decision.v2',
            'qualification.email_evidence.v2',
            'qualification.sourcing_epoch.v2'
        )) OR
        (role = 'gateway_autoresearch' AND purpose IN (
            'research_lab.source_inspection.v2',
            'research_lab.research_plan.v2',
            'research_lab.patch_draft.v2',
            'research_lab.patch_validation.v2',
            'research_lab.candidate_test.v2',
            'research_lab.candidate_build.v2',
            'research_lab.candidate_decision.v2',
            'research_lab.stale_parent_repair.v2',
            'research_lab.checkpoint.v2',
            'research_lab.openrouter_guard.v2'
        )) OR
        (role = 'validator_weights' AND purpose IN (
            'validator.weight_snapshot.v2',
            'validator.weights.computed.v2',
            'validator.chain_state.v2',
            'validator.subnet_epoch_snapshot.v2',
            'validator.metagraph_state.v2',
            'validator.burn_ownership.v2',
            'validator.feature_flags.v2',
            'validator.constants.v2',
            'validator.hotkey_signature.v2',
            'validator.serve_axon_extrinsic.v2',
            'validator.set_weights_extrinsic.v2',
            'validator.weights.finalized.v2'
        ))
    ) NOT VALID;

ALTER TABLE public.research_lab_attested_execution_receipts_v2
    VALIDATE CONSTRAINT
        research_lab_attested_execution_receipts_v2_role_purpose_check;

NOTIFY pgrst, 'reload schema';

COMMIT;
