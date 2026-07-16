-- Champion obligation V2 cutover and finalized-chain settlement authority.
--
-- Additive only. This view exposes an exact, service-role-only join across the
-- immutable V2 bundle, publication, and finalized-chain records. Allocation
-- snapshots that did not reach this join are intentionally not payment proof.
-- Apply after scripts/86-research-lab-attested-v2-authority.sql.

BEGIN;

-- Migration 96 owns the canonical role/purpose constraint for clean installs.
-- Widen that same constraint here so already-deployed databases can persist
-- measured pre-V2 settlement receipts before V2 allocation cutover.
DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.research_lab_attested_execution_receipts_v2'::REGCLASS
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%gateway_coordinator%'
          AND pg_get_constraintdef(oid) LIKE '%purpose%'
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_attested_execution_receipts_v2 DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

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
            'gateway.weights.publication.v2'
        )) OR
        (role = 'gateway_scoring' AND purpose IN (
            'research_lab.private_model_run.v2',
            'research_lab.candidate_model_run.v2',
            'research_lab.provider_evidence_tape.v2',
            'research_lab.candidate_test.v2',
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
    VALIDATE CONSTRAINT research_lab_attested_execution_receipts_v2_role_purpose_check;

CREATE OR REPLACE VIEW public.research_lab_finalized_allocation_epochs_v2
WITH (security_invoker = true) AS
SELECT
    b.bundle_hash,
    b.schema_version,
    b.netuid,
    b.epoch_id,
    b.block,
    b.validator_hotkey,
    b.root_receipt_hash,
    b.weights_hash,
    b.snapshot_hash,
    b.bundle_doc,
    p.weight_submission_event_hash,
    p.publication_receipt_hash,
    p.transparency_event_hash,
    p.durable_readback_hash,
    p.publication_doc,
    f.weight_finalization_event_hash,
    f.finalization_receipt_hash,
    f.extrinsic_authorization_hash,
    f.extrinsic_hash,
    f.finalized_block,
    f.finalized_block_hash,
    f.state_transition_hash,
    f.finalization_doc
FROM public.research_lab_attested_weight_bundles_v2 b
JOIN public.research_lab_attested_publication_events_v2 p
  ON p.bundle_hash = b.bundle_hash
JOIN public.research_lab_attested_weight_finalizations_v2 f
  ON f.bundle_hash = b.bundle_hash
 AND f.weight_submission_event_hash = p.weight_submission_event_hash;

REVOKE ALL ON TABLE public.research_lab_finalized_allocation_epochs_v2
    FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_finalized_allocation_epochs_v2
    TO service_role;

COMMENT ON VIEW public.research_lab_finalized_allocation_epochs_v2 IS
    'Service-role-only immutable V2 allocation inputs whose exact validator weight extrinsic reached finalized chain state.';

CREATE TABLE IF NOT EXISTS public.research_lab_legacy_finalized_allocation_migrations_v2 (
    netuid                      INTEGER     NOT NULL CHECK (netuid > 0),
    epoch_id                    INTEGER     NOT NULL CHECK (epoch_id >= 0),
    schema_version              TEXT        NOT NULL
                                           CHECK (schema_version = 'leadpoet.legacy_finalized_allocation.v2'),
    allocation_hash             TEXT        NOT NULL UNIQUE
                                           CHECK (allocation_hash ~ '^sha256:[0-9a-f]{64}$'),
    settlement_hash             TEXT        NOT NULL UNIQUE
                                           CHECK (settlement_hash ~ '^sha256:[0-9a-f]{64}$'),
    settlement_receipt_hash     TEXT        NOT NULL UNIQUE
                                           REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                           ON DELETE RESTRICT,
    allocation_doc              JSONB       NOT NULL CHECK (
                                           jsonb_typeof(allocation_doc) = 'object'
                                           AND allocation_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                           ),
    settlement_doc              JSONB       NOT NULL CHECK (
                                           jsonb_typeof(settlement_doc) = 'object'
                                           AND settlement_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                           ),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (netuid, epoch_id),
    CHECK ((allocation_doc->>'epoch')::INTEGER = epoch_id),
    CHECK ((allocation_doc->>'netuid')::INTEGER = netuid),
    CHECK (allocation_doc->>'allocation_hash' = allocation_hash),
    CHECK (settlement_doc->>'schema_version' = schema_version),
    CHECK ((settlement_doc->>'epoch_id')::INTEGER = epoch_id),
    CHECK ((settlement_doc->>'netuid')::INTEGER = netuid),
    CHECK (settlement_doc->>'allocation_hash' = allocation_hash),
    CHECK (settlement_doc->>'settlement_hash' = settlement_hash)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_legacy_settlement_receipt_v2
    ON public.research_lab_legacy_finalized_allocation_migrations_v2(settlement_receipt_hash);

DROP TRIGGER IF EXISTS prevent_research_lab_legacy_settlement_v2_mutation
    ON public.research_lab_legacy_finalized_allocation_migrations_v2;
CREATE TRIGGER prevent_research_lab_legacy_settlement_v2_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_legacy_finalized_allocation_migrations_v2
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL ON TABLE public.research_lab_legacy_finalized_allocation_migrations_v2
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_legacy_finalized_allocation_migrations_v2
    TO service_role;

ALTER TABLE public.research_lab_legacy_finalized_allocation_migrations_v2
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_legacy_finalized_allocation_migrations_v2;
CREATE POLICY service_role_read
    ON public.research_lab_legacy_finalized_allocation_migrations_v2
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_legacy_finalized_allocation_migrations_v2;
CREATE POLICY service_role_insert
    ON public.research_lab_legacy_finalized_allocation_migrations_v2
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_legacy_finalized_allocation_migrations_v2 IS
    'Append-only measured migration receipts proving pre-V2 Research Lab allocations reached finalized epoch-end chain state.';

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Verify after applying:
-- SELECT netuid, epoch_id, validator_hotkey, bundle_hash,
--        weight_finalization_event_hash
-- FROM public.research_lab_finalized_allocation_epochs_v2
-- ORDER BY epoch_id DESC, validator_hotkey;
--
-- SELECT netuid, epoch_id, allocation_hash, settlement_hash,
--        settlement_receipt_hash
-- FROM public.research_lab_legacy_finalized_allocation_migrations_v2
-- ORDER BY epoch_id;
