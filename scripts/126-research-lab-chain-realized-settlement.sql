-- Chain-realized Research Lab settlement credits.
--
-- Additive only. These append-only ledgers let the allocator settle Lab
-- obligations from finalized chain-observed emission, rather than from weight
-- submission intent alone.  A complete per-epoch settlement marker is required
-- before per-obligation credit rows can replace finalized allocation history.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Permit the coordinator enclave to attest chain-realized settlement evidence.
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
            'research_lab.chain_realized_epoch_settlement.v1',
            'research_lab.chain_realized_obligation_credit.v1',
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
            'validator.subnet_epoch_snapshot.v2',
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

CREATE TABLE IF NOT EXISTS public.research_lab_chain_realized_epoch_settlements_v1 (
    netuid                   INTEGER     NOT NULL CHECK (netuid > 0),
    epoch_id                 INTEGER     NOT NULL CHECK (epoch_id >= 0),
    schema_version           TEXT        NOT NULL
                                      CHECK (schema_version = 'leadpoet.research_lab_chain_realized_epoch_settlement.v1'),
    settlement_hash          TEXT        NOT NULL UNIQUE
                                      CHECK (settlement_hash ~ '^sha256:[0-9a-f]{64}$'),
    settlement_receipt_hash  TEXT        NOT NULL UNIQUE
                                      REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                      ON DELETE RESTRICT,
    settlement_doc           JSONB       NOT NULL CHECK (
                                      jsonb_typeof(settlement_doc) = 'object'
                                      AND settlement_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                      ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (netuid, epoch_id),
    CHECK (settlement_doc->>'schema_version' = schema_version),
    CHECK ((settlement_doc->>'netuid')::INTEGER = netuid),
    CHECK ((settlement_doc->>'epoch_id')::INTEGER = epoch_id),
    CHECK (jsonb_typeof(settlement_doc->'credit_hashes') = 'array')
);

CREATE INDEX IF NOT EXISTS idx_research_lab_chain_settlement_receipt_v1
    ON public.research_lab_chain_realized_epoch_settlements_v1(settlement_receipt_hash);
CREATE INDEX IF NOT EXISTS idx_research_lab_chain_settlement_epoch_v1
    ON public.research_lab_chain_realized_epoch_settlements_v1(epoch_id DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_chain_settlement_v1_mutation
    ON public.research_lab_chain_realized_epoch_settlements_v1;
CREATE TRIGGER prevent_research_lab_chain_settlement_v1_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_chain_realized_epoch_settlements_v1
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL ON TABLE public.research_lab_chain_realized_epoch_settlements_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_chain_realized_epoch_settlements_v1
    TO service_role;

ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_chain_realized_epoch_settlements_v1;
CREATE POLICY service_role_read
    ON public.research_lab_chain_realized_epoch_settlements_v1
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_chain_realized_epoch_settlements_v1;
CREATE POLICY service_role_insert
    ON public.research_lab_chain_realized_epoch_settlements_v1
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_chain_realized_epoch_settlements_v1 IS
    'Append-only complete epoch markers for Research Lab chain-realized settlement credit sets.';

CREATE TABLE IF NOT EXISTS public.research_lab_chain_realized_obligation_credits_v1 (
    netuid                         INTEGER     NOT NULL CHECK (netuid > 0),
    epoch_id                       INTEGER     NOT NULL CHECK (epoch_id >= 0),
    settlement_hash                TEXT        NOT NULL
                                                REFERENCES public.research_lab_chain_realized_epoch_settlements_v1(settlement_hash)
                                                ON DELETE RESTRICT,
    schema_version                 TEXT        NOT NULL
                                                CHECK (schema_version = 'leadpoet.research_lab_chain_realized_obligation_credit.v1'),
    obligation_kind                TEXT        NOT NULL
                                                CHECK (obligation_kind IN ('champion', 'queued_champion', 'source_add', 'reimbursement')),
    obligation_source_id           TEXT        NOT NULL CHECK (length(obligation_source_id) BETWEEN 1 AND 512),
    miner_hotkey                   TEXT        NOT NULL CHECK (length(miner_hotkey) BETWEEN 1 AND 128),
    miner_uid                      INTEGER     NOT NULL CHECK (miner_uid >= 0),
    observed_chain_alpha_percent   NUMERIC(18, 6) NOT NULL CHECK (observed_chain_alpha_percent >= 0),
    lab_attributed_alpha_percent   NUMERIC(18, 6) NOT NULL CHECK (lab_attributed_alpha_percent >= 0),
    scheduled_alpha_percent        NUMERIC(18, 6) NOT NULL CHECK (scheduled_alpha_percent >= 0),
    credited_alpha_percent         NUMERIC(18, 6) NOT NULL CHECK (credited_alpha_percent >= 0),
    credit_hash                    TEXT        NOT NULL UNIQUE
                                                CHECK (credit_hash ~ '^sha256:[0-9a-f]{64}$'),
    credit_receipt_hash            TEXT        NOT NULL UNIQUE
                                                REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                                ON DELETE RESTRICT,
    credit_doc                     JSONB       NOT NULL CHECK (
                                                jsonb_typeof(credit_doc) = 'object'
                                                AND credit_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                                ),
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (netuid, epoch_id, obligation_kind, obligation_source_id),
    FOREIGN KEY (netuid, epoch_id)
        REFERENCES public.research_lab_chain_realized_epoch_settlements_v1(netuid, epoch_id)
        ON DELETE RESTRICT,
    CHECK (credit_doc->>'schema_version' = schema_version),
    CHECK ((credit_doc->>'netuid')::INTEGER = netuid),
    CHECK ((credit_doc->>'epoch_id')::INTEGER = epoch_id),
    CHECK (credit_doc->>'obligation_kind' = obligation_kind),
    CHECK (credit_doc->>'obligation_source_id' = obligation_source_id),
    CHECK (credit_doc->>'miner_hotkey' = miner_hotkey),
    CHECK ((credit_doc->>'miner_uid')::INTEGER = miner_uid),
    CHECK ((credit_doc->>'observed_chain_alpha_percent')::NUMERIC = observed_chain_alpha_percent),
    CHECK ((credit_doc->>'lab_attributed_alpha_percent')::NUMERIC = lab_attributed_alpha_percent),
    CHECK ((credit_doc->>'scheduled_alpha_percent')::NUMERIC = scheduled_alpha_percent),
    CHECK ((credit_doc->>'credited_alpha_percent')::NUMERIC = credited_alpha_percent),
    CHECK (credited_alpha_percent <= lab_attributed_alpha_percent),
    CHECK (lab_attributed_alpha_percent <= observed_chain_alpha_percent),
    CHECK (scheduled_alpha_percent = 0 OR credited_alpha_percent <= scheduled_alpha_percent),
    CHECK (jsonb_typeof(credit_doc->'attribution_doc') = 'object'),
    CHECK (jsonb_typeof(credit_doc->'observation_doc') = 'object')
);

CREATE INDEX IF NOT EXISTS idx_research_lab_chain_credit_receipt_v1
    ON public.research_lab_chain_realized_obligation_credits_v1(credit_receipt_hash);
CREATE INDEX IF NOT EXISTS idx_research_lab_chain_credit_epoch_v1
    ON public.research_lab_chain_realized_obligation_credits_v1(epoch_id DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_chain_credit_settlement_v1
    ON public.research_lab_chain_realized_obligation_credits_v1(settlement_hash);

DROP TRIGGER IF EXISTS prevent_research_lab_chain_credit_v1_mutation
    ON public.research_lab_chain_realized_obligation_credits_v1;
CREATE TRIGGER prevent_research_lab_chain_credit_v1_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_chain_realized_obligation_credits_v1
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL ON TABLE public.research_lab_chain_realized_obligation_credits_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_chain_realized_obligation_credits_v1
    TO service_role;

ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_chain_realized_obligation_credits_v1;
CREATE POLICY service_role_read
    ON public.research_lab_chain_realized_obligation_credits_v1
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_chain_realized_obligation_credits_v1;
CREATE POLICY service_role_insert
    ON public.research_lab_chain_realized_obligation_credits_v1
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_chain_realized_obligation_credits_v1 IS
    'Append-only per-obligation Lab credits derived from finalized chain emission observations and bounded by Lab attribution.';

COMMIT;
