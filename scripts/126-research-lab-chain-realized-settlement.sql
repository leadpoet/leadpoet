-- Chain-realized Research Lab settlement credits.
--
-- Additive only. These append-only ledgers settle Lab obligations from the
-- primary validator's finalized on-chain weight vector. One coordinator
-- receipt binds the complete epoch settlement and its ordered credit hashes.
-- The SECURITY DEFINER RPC writes the marker and every credit atomically.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Migration 104 deliberately restricted replayable coordinator results to the
-- two operations that existed then. Extend that exact operation/purpose
-- contract before the new measured results can be persisted.
DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_execution_results_v2'::REGCLASS
          AND contype = 'c'
          AND (
              pg_get_constraintdef(oid) ~ '\moperation\M'
              OR pg_get_constraintdef(oid) ~ '\mpurpose\M'
          )
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_attested_execution_results_v2 '
            'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_attested_execution_results_v2
    ADD CONSTRAINT
        research_lab_attested_execution_results_v2_operation_check
    CHECK (
        operation IN (
            'research_lab_allocation',
            'attest_weight_input',
            'observe_chain_realized_weights_v1',
            'attest_chain_realized_settlement_v1'
        )
    ) NOT VALID,
    ADD CONSTRAINT
        research_lab_attested_execution_results_v2_purpose_check
    CHECK (
        purpose IN (
            'research_lab.allocation.v2',
            'research_lab.champion_input.v2',
            'research_lab.reimbursement_input.v2',
            'research_lab.source_add_reward_input.v2',
            'research_lab.sourcing_input.v2',
            'research_lab.fulfillment_input.v2',
            'research_lab.leaderboard_input.v2',
            'research_lab.ban_input.v2',
            'research_lab.anomaly_adjustment_input.v2',
            'research_lab.chain_weight_observation.v1',
            'research_lab.chain_realized_epoch_settlement.v1'
        )
    ) NOT VALID,
    ADD CONSTRAINT
        research_lab_attested_execution_results_v2_operation_purpose_check
    CHECK (
        (
            operation = 'research_lab_allocation'
            AND purpose = 'research_lab.allocation.v2'
        )
        OR (
            operation = 'attest_weight_input'
            AND purpose IN (
                'research_lab.champion_input.v2',
                'research_lab.reimbursement_input.v2',
                'research_lab.source_add_reward_input.v2',
                'research_lab.sourcing_input.v2',
                'research_lab.fulfillment_input.v2',
                'research_lab.leaderboard_input.v2',
                'research_lab.ban_input.v2',
                'research_lab.anomaly_adjustment_input.v2'
            )
        )
        OR (
            operation = 'observe_chain_realized_weights_v1'
            AND purpose = 'research_lab.chain_weight_observation.v1'
        )
        OR (
            operation = 'attest_chain_realized_settlement_v1'
            AND purpose =
                'research_lab.chain_realized_epoch_settlement.v1'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_attested_execution_results_v2
    VALIDATE CONSTRAINT
        research_lab_attested_execution_results_v2_operation_check;
ALTER TABLE public.research_lab_attested_execution_results_v2
    VALIDATE CONSTRAINT
        research_lab_attested_execution_results_v2_purpose_check;
ALTER TABLE public.research_lab_attested_execution_results_v2
    VALIDATE CONSTRAINT
        research_lab_attested_execution_results_v2_operation_purpose_check;

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_execution_receipts_v2'::REGCLASS
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
    VALIDATE CONSTRAINT
        research_lab_attested_execution_receipts_v2_role_purpose_check;

CREATE OR REPLACE VIEW
public.research_lab_finalized_weight_vector_candidates_v1
WITH (security_invoker = true) AS
SELECT
    authority.*,
    authority.bundle_doc #> '{weight_result,sparse_uids}' AS uids,
    authority.bundle_doc #> '{weight_result,sparse_weights_u16}' AS weights_u16
FROM public.research_lab_finalized_allocation_epochs_v2 authority;

REVOKE ALL ON TABLE
    public.research_lab_finalized_weight_vector_candidates_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE
    public.research_lab_finalized_weight_vector_candidates_v1
    TO service_role;

COMMENT ON VIEW
    public.research_lab_finalized_weight_vector_candidates_v1 IS
    'Service-role-only exact finalized V2 bundle lookup by canonical sparse chain vector.';

CREATE TABLE IF NOT EXISTS
public.research_lab_chain_realized_settlement_activation_v1 (
    netuid                  INTEGER     PRIMARY KEY CHECK (netuid > 0),
    schema_version          TEXT        NOT NULL CHECK (
        schema_version =
        'leadpoet.research_lab_chain_realized_settlement_activation.v1'
    ),
    first_epoch_id          INTEGER     NOT NULL CHECK (first_epoch_id >= 0),
    source_bundle_hash      TEXT        NOT NULL CHECK (
        source_bundle_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    source_bundle_epoch_id  INTEGER     NOT NULL CHECK (
        source_bundle_epoch_id = first_epoch_id
    ),
    source_finalized_block  BIGINT      NOT NULL CHECK (
        source_finalized_block >= 0
    ),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO
public.research_lab_chain_realized_settlement_activation_v1 (
    netuid,
    schema_version,
    first_epoch_id,
    source_bundle_hash,
    source_bundle_epoch_id,
    source_finalized_block
)
SELECT DISTINCT ON (authority.netuid)
    authority.netuid,
    'leadpoet.research_lab_chain_realized_settlement_activation.v1',
    authority.epoch_id,
    authority.bundle_hash,
    authority.epoch_id,
    authority.finalized_block
FROM public.research_lab_finalized_allocation_epochs_v2 authority
ORDER BY
    authority.netuid,
    authority.epoch_id DESC,
    authority.finalized_block DESC,
    authority.bundle_hash ASC
ON CONFLICT (netuid) DO NOTHING;

DROP TRIGGER IF EXISTS
    prevent_research_lab_chain_settlement_activation_v1_mutation
    ON public.research_lab_chain_realized_settlement_activation_v1;
CREATE TRIGGER
    prevent_research_lab_chain_settlement_activation_v1_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_chain_realized_settlement_activation_v1
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL
    ON TABLE
    public.research_lab_chain_realized_settlement_activation_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT
    ON TABLE
    public.research_lab_chain_realized_settlement_activation_v1
    TO service_role;

ALTER TABLE
    public.research_lab_chain_realized_settlement_activation_v1
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_chain_realized_settlement_activation_v1;
CREATE POLICY service_role_read
    ON public.research_lab_chain_realized_settlement_activation_v1
    FOR SELECT TO service_role USING (true);

COMMENT ON TABLE
public.research_lab_chain_realized_settlement_activation_v1 IS
    'Immutable migration-time first epoch for contiguous chain-realized Research Lab settlement.';

CREATE TABLE IF NOT EXISTS
public.research_lab_chain_realized_epoch_settlements_v1 (
    netuid                   INTEGER     NOT NULL CHECK (netuid > 0),
    epoch_id                 INTEGER     NOT NULL CHECK (epoch_id >= 0),
    schema_version           TEXT        NOT NULL CHECK (
        schema_version =
        'leadpoet.research_lab_chain_realized_epoch_settlement.v1'
    ),
    settlement_hash          TEXT        NOT NULL UNIQUE CHECK (
        settlement_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    settlement_receipt_hash  TEXT        NOT NULL UNIQUE
        REFERENCES public.research_lab_attested_execution_receipts_v2(
            receipt_hash
        ) ON DELETE RESTRICT,
    settlement_doc           JSONB       NOT NULL CHECK (
        jsonb_typeof(settlement_doc) = 'object'
        AND settlement_doc::TEXT !~*
            '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
    ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (netuid, epoch_id),
    UNIQUE (netuid, epoch_id, settlement_hash),
    CHECK (settlement_doc->>'schema_version' = schema_version),
    CHECK ((settlement_doc->>'netuid')::INTEGER = netuid),
    CHECK ((settlement_doc->>'epoch_id')::INTEGER = epoch_id),
    CHECK (jsonb_typeof(settlement_doc->'credit_hashes') = 'array')
);

CREATE INDEX IF NOT EXISTS
idx_research_lab_chain_settlement_receipt_v1
    ON public.research_lab_chain_realized_epoch_settlements_v1(
        settlement_receipt_hash
    );
CREATE INDEX IF NOT EXISTS idx_research_lab_chain_settlement_epoch_v1
    ON public.research_lab_chain_realized_epoch_settlements_v1(
        epoch_id DESC
    );

DROP TRIGGER IF EXISTS prevent_research_lab_chain_settlement_v1_mutation
    ON public.research_lab_chain_realized_epoch_settlements_v1;
CREATE TRIGGER prevent_research_lab_chain_settlement_v1_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_chain_realized_epoch_settlements_v1
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL
    ON TABLE public.research_lab_chain_realized_epoch_settlements_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT
    ON TABLE public.research_lab_chain_realized_epoch_settlements_v1
    TO service_role;

ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_chain_realized_epoch_settlements_v1;
CREATE POLICY service_role_read
    ON public.research_lab_chain_realized_epoch_settlements_v1
    FOR SELECT TO service_role USING (true);

COMMENT ON TABLE
public.research_lab_chain_realized_epoch_settlements_v1 IS
    'Append-only complete epoch markers for Research Lab chain-realized settlement credit sets.';

CREATE TABLE IF NOT EXISTS
public.research_lab_chain_realized_obligation_credits_v1 (
    netuid                         INTEGER        NOT NULL CHECK (netuid > 0),
    epoch_id                       INTEGER        NOT NULL CHECK (epoch_id >= 0),
    settlement_hash                TEXT           NOT NULL CHECK (
        settlement_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    schema_version                 TEXT           NOT NULL CHECK (
        schema_version =
        'leadpoet.research_lab_chain_realized_obligation_credit.v1'
    ),
    obligation_kind                TEXT           NOT NULL CHECK (
        obligation_kind IN (
            'champion',
            'queued_champion',
            'source_add',
            'reimbursement'
        )
    ),
    obligation_source_id           TEXT           NOT NULL CHECK (
        length(obligation_source_id) BETWEEN 1 AND 512
    ),
    miner_hotkey                   TEXT           NOT NULL CHECK (
        length(miner_hotkey) BETWEEN 1 AND 128
    ),
    miner_uid                      INTEGER        NOT NULL CHECK (
        miner_uid >= 0
    ),
    observed_chain_alpha_percent   NUMERIC(24, 12) NOT NULL CHECK (
        observed_chain_alpha_percent >= 0
    ),
    lab_attributed_alpha_percent   NUMERIC(24, 12) NOT NULL CHECK (
        lab_attributed_alpha_percent >= 0
    ),
    scheduled_alpha_percent        NUMERIC(24, 12) NOT NULL CHECK (
        scheduled_alpha_percent >= 0
    ),
    credited_alpha_percent         NUMERIC(24, 12) NOT NULL CHECK (
        credited_alpha_percent >= 0
    ),
    credit_hash                    TEXT           NOT NULL UNIQUE CHECK (
        credit_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    credit_receipt_hash            TEXT           NOT NULL
        REFERENCES public.research_lab_attested_execution_receipts_v2(
            receipt_hash
        ) ON DELETE RESTRICT,
    credit_doc                     JSONB          NOT NULL CHECK (
        jsonb_typeof(credit_doc) = 'object'
        AND credit_doc::TEXT !~*
            '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
    ),
    created_at                     TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        netuid,
        epoch_id,
        obligation_kind,
        obligation_source_id
    ),
    FOREIGN KEY (netuid, epoch_id, settlement_hash)
        REFERENCES
        public.research_lab_chain_realized_epoch_settlements_v1(
            netuid,
            epoch_id,
            settlement_hash
        ) ON DELETE RESTRICT,
    CHECK (credit_doc->>'schema_version' = schema_version),
    CHECK ((credit_doc->>'netuid')::INTEGER = netuid),
    CHECK ((credit_doc->>'epoch_id')::INTEGER = epoch_id),
    CHECK (credit_doc->>'obligation_kind' = obligation_kind),
    CHECK (credit_doc->>'obligation_source_id' = obligation_source_id),
    CHECK (credit_doc->>'miner_hotkey' = miner_hotkey),
    CHECK ((credit_doc->>'miner_uid')::INTEGER = miner_uid),
    CHECK (
        (credit_doc->>'observed_chain_alpha_percent')::NUMERIC =
        observed_chain_alpha_percent
    ),
    CHECK (
        (credit_doc->>'lab_attributed_alpha_percent')::NUMERIC =
        lab_attributed_alpha_percent
    ),
    CHECK (
        (credit_doc->>'scheduled_alpha_percent')::NUMERIC =
        scheduled_alpha_percent
    ),
    CHECK (
        (credit_doc->>'credited_alpha_percent')::NUMERIC =
        credited_alpha_percent
    ),
    CHECK (credited_alpha_percent <= lab_attributed_alpha_percent),
    CHECK (lab_attributed_alpha_percent <= observed_chain_alpha_percent),
    CHECK (
        scheduled_alpha_percent = 0
        OR credited_alpha_percent <= scheduled_alpha_percent
    ),
    CHECK (jsonb_typeof(credit_doc->'attribution_doc') = 'object'),
    CHECK (jsonb_typeof(credit_doc->'observation_doc') = 'object')
);

CREATE INDEX IF NOT EXISTS idx_research_lab_chain_credit_receipt_v1
    ON public.research_lab_chain_realized_obligation_credits_v1(
        credit_receipt_hash
    );
CREATE INDEX IF NOT EXISTS idx_research_lab_chain_credit_epoch_v1
    ON public.research_lab_chain_realized_obligation_credits_v1(
        epoch_id DESC
    );
CREATE INDEX IF NOT EXISTS idx_research_lab_chain_credit_settlement_v1
    ON public.research_lab_chain_realized_obligation_credits_v1(
        settlement_hash
    );
CREATE UNIQUE INDEX IF NOT EXISTS
    idx_research_lab_chain_credit_obligation_identity_v1
    ON public.research_lab_chain_realized_obligation_credits_v1(
        netuid,
        epoch_id,
        (
            CASE
                WHEN obligation_kind IN ('champion', 'queued_champion')
                    THEN 'champion'
                ELSE obligation_kind
            END
        ),
        obligation_source_id
    );

DROP TRIGGER IF EXISTS prevent_research_lab_chain_credit_v1_mutation
    ON public.research_lab_chain_realized_obligation_credits_v1;
CREATE TRIGGER prevent_research_lab_chain_credit_v1_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_chain_realized_obligation_credits_v1
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL
    ON TABLE public.research_lab_chain_realized_obligation_credits_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT
    ON TABLE public.research_lab_chain_realized_obligation_credits_v1
    TO service_role;

ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_chain_realized_obligation_credits_v1;
CREATE POLICY service_role_read
    ON public.research_lab_chain_realized_obligation_credits_v1
    FOR SELECT TO service_role USING (true);

COMMENT ON TABLE
public.research_lab_chain_realized_obligation_credits_v1 IS
    'Append-only per-obligation Lab credits from finalized active chain weights and canonical V2 allocation evidence.';

CREATE OR REPLACE FUNCTION
public.persist_research_lab_chain_realized_settlement_v1(
    requested_settlement JSONB,
    requested_credits JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    settlement_netuid INTEGER;
    settlement_epoch INTEGER;
    requested_settlement_hash TEXT;
    requested_settlement_receipt_hash TEXT;
    requested_settlement_doc JSONB;
    expected_credit_hashes JSONB;
    requested_credit_count INTEGER;
    requested_unique_credit_count INTEGER;
    stored_credit_hashes JSONB;
    activation_first_epoch INTEGER;
    item JSONB;
    stored_settlement
        public.research_lab_chain_realized_epoch_settlements_v1;
BEGIN
    IF pg_catalog.jsonb_typeof(requested_settlement) <> 'object'
       OR pg_catalog.jsonb_typeof(requested_credits) <> 'array'
       OR requested_settlement <> requested_settlement - 'created_at' THEN
        RAISE EXCEPTION 'chain_realized_settlement_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    BEGIN
        settlement_netuid :=
            (requested_settlement->>'netuid')::INTEGER;
        settlement_epoch :=
            (requested_settlement->>'epoch_id')::INTEGER;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'chain_realized_settlement_scope_invalid'
            USING ERRCODE = '22023';
    END;
    requested_settlement_hash := requested_settlement->>'settlement_hash';
    requested_settlement_receipt_hash :=
        requested_settlement->>'settlement_receipt_hash';
    requested_settlement_doc := requested_settlement->'settlement_doc';
    expected_credit_hashes := requested_settlement_doc->'credit_hashes';
    IF settlement_netuid <= 0
       OR settlement_epoch < 0
       OR requested_settlement->>'schema_version' <>
          'leadpoet.research_lab_chain_realized_epoch_settlement.v1'
       OR requested_settlement_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_settlement_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(requested_settlement_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(expected_credit_hashes) <> 'array' THEN
        RAISE EXCEPTION 'chain_realized_settlement_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    SELECT
        pg_catalog.jsonb_array_length(requested_credits),
        pg_catalog.count(DISTINCT value->>'credit_hash')
    INTO requested_credit_count, requested_unique_credit_count
    FROM pg_catalog.jsonb_array_elements(requested_credits);
    IF requested_credit_count > 1000
       OR requested_credit_count <> requested_unique_credit_count
       OR requested_credit_count <>
          pg_catalog.jsonb_array_length(expected_credit_hashes) THEN
        RAISE EXCEPTION 'chain_realized_credit_set_invalid'
            USING ERRCODE = '22023';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(requested_credits) AS credit(value)
        WHERE pg_catalog.jsonb_typeof(value) <> 'object'
           OR (value->>'netuid')::INTEGER IS DISTINCT FROM settlement_netuid
           OR (value->>'epoch_id')::INTEGER IS DISTINCT FROM settlement_epoch
           OR value->>'settlement_hash'
              IS DISTINCT FROM requested_settlement_hash
           OR value->>'credit_receipt_hash'
              IS DISTINCT FROM requested_settlement_receipt_hash
           OR NOT (expected_credit_hashes ? (value->>'credit_hash'))
    ) THEN
        RAISE EXCEPTION 'chain_realized_credit_scope_invalid'
            USING ERRCODE = '22023';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('chain_realized_settlement_v1'),
        settlement_netuid
    );

    SELECT first_epoch_id
    INTO activation_first_epoch
    FROM public.research_lab_chain_realized_settlement_activation_v1
    WHERE netuid = settlement_netuid;
    IF activation_first_epoch IS NULL
       OR settlement_epoch < activation_first_epoch THEN
        RAISE EXCEPTION 'chain_realized_settlement_activation_invalid'
            USING ERRCODE = '55000';
    END IF;

    IF settlement_epoch > activation_first_epoch
       AND NOT EXISTS (
           SELECT 1
           FROM
               public.research_lab_chain_realized_epoch_settlements_v1
               predecessor
           WHERE predecessor.netuid = settlement_netuid
             AND predecessor.epoch_id = settlement_epoch - 1
       ) THEN
        RAISE EXCEPTION 'chain_realized_settlement_predecessor_missing'
            USING ERRCODE = '55000';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 receipt
        WHERE receipt.receipt_hash = requested_settlement_receipt_hash
          AND receipt.role = 'gateway_coordinator'
          AND receipt.purpose =
              'research_lab.chain_realized_epoch_settlement.v1'
          AND receipt.epoch_id = settlement_epoch
          AND receipt.output_root = requested_settlement_hash
          AND receipt.receipt_status = 'succeeded'
    ) THEN
        RAISE EXCEPTION 'chain_realized_settlement_receipt_invalid'
            USING ERRCODE = '55000';
    END IF;

    INSERT INTO
    public.research_lab_chain_realized_epoch_settlements_v1 (
        netuid,
        epoch_id,
        schema_version,
        settlement_hash,
        settlement_receipt_hash,
        settlement_doc
    ) VALUES (
        settlement_netuid,
        settlement_epoch,
        requested_settlement->>'schema_version',
        requested_settlement_hash,
        requested_settlement_receipt_hash,
        requested_settlement_doc
    )
    ON CONFLICT DO NOTHING;

    SELECT * INTO stored_settlement
    FROM public.research_lab_chain_realized_epoch_settlements_v1
    WHERE netuid = settlement_netuid
      AND epoch_id = settlement_epoch;
    IF stored_settlement.netuid IS NULL
       OR stored_settlement.schema_version IS DISTINCT FROM
          requested_settlement->>'schema_version'
       OR stored_settlement.settlement_hash
          IS DISTINCT FROM requested_settlement_hash
       OR stored_settlement.settlement_receipt_hash IS DISTINCT FROM
          requested_settlement_receipt_hash
       OR stored_settlement.settlement_doc
          IS DISTINCT FROM requested_settlement_doc THEN
        RAISE EXCEPTION 'chain_realized_settlement_conflict'
            USING ERRCODE = '40001';
    END IF;

    FOR item IN
        SELECT value
        FROM pg_catalog.jsonb_array_elements(requested_credits)
        ORDER BY value->>'credit_hash'
    LOOP
        INSERT INTO
        public.research_lab_chain_realized_obligation_credits_v1 (
            netuid,
            epoch_id,
            settlement_hash,
            schema_version,
            obligation_kind,
            obligation_source_id,
            miner_hotkey,
            miner_uid,
            observed_chain_alpha_percent,
            lab_attributed_alpha_percent,
            scheduled_alpha_percent,
            credited_alpha_percent,
            credit_hash,
            credit_receipt_hash,
            credit_doc
        ) VALUES (
            settlement_netuid,
            settlement_epoch,
            requested_settlement_hash,
            item->>'schema_version',
            item->>'obligation_kind',
            item->>'obligation_source_id',
            item->>'miner_hotkey',
            (item->>'miner_uid')::INTEGER,
            (item->>'observed_chain_alpha_percent')::NUMERIC,
            (item->>'lab_attributed_alpha_percent')::NUMERIC,
            (item->>'scheduled_alpha_percent')::NUMERIC,
            (item->>'credited_alpha_percent')::NUMERIC,
            item->>'credit_hash',
            item->>'credit_receipt_hash',
            item->'credit_doc'
        )
        ON CONFLICT DO NOTHING;
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(requested_credits) AS credit(value)
        LEFT JOIN
            public.research_lab_chain_realized_obligation_credits_v1 stored
          ON stored.netuid = settlement_netuid
         AND stored.epoch_id = settlement_epoch
         AND stored.obligation_kind = value->>'obligation_kind'
         AND stored.obligation_source_id =
             value->>'obligation_source_id'
        WHERE stored.netuid IS NULL
           OR stored.settlement_hash
              IS DISTINCT FROM requested_settlement_hash
           OR stored.schema_version IS DISTINCT FROM value->>'schema_version'
           OR stored.miner_hotkey IS DISTINCT FROM value->>'miner_hotkey'
           OR stored.miner_uid IS DISTINCT FROM
              (value->>'miner_uid')::INTEGER
           OR stored.observed_chain_alpha_percent IS DISTINCT FROM
              (value->>'observed_chain_alpha_percent')::NUMERIC
           OR stored.lab_attributed_alpha_percent IS DISTINCT FROM
              (value->>'lab_attributed_alpha_percent')::NUMERIC
           OR stored.scheduled_alpha_percent IS DISTINCT FROM
              (value->>'scheduled_alpha_percent')::NUMERIC
           OR stored.credited_alpha_percent IS DISTINCT FROM
              (value->>'credited_alpha_percent')::NUMERIC
           OR stored.credit_hash IS DISTINCT FROM value->>'credit_hash'
           OR stored.credit_receipt_hash IS DISTINCT FROM
              value->>'credit_receipt_hash'
           OR stored.credit_doc IS DISTINCT FROM value->'credit_doc'
    ) THEN
        RAISE EXCEPTION 'chain_realized_credit_conflict'
            USING ERRCODE = '40001';
    END IF;

    SELECT COALESCE(
        pg_catalog.jsonb_agg(credit_hash ORDER BY credit_hash),
        '[]'::JSONB
    )
    INTO stored_credit_hashes
    FROM public.research_lab_chain_realized_obligation_credits_v1
    WHERE netuid = settlement_netuid
      AND epoch_id = settlement_epoch
      AND settlement_hash = requested_settlement_hash;
    IF stored_credit_hashes IS DISTINCT FROM (
        SELECT COALESCE(
            pg_catalog.jsonb_agg(value ORDER BY value),
            '[]'::JSONB
        )
        FROM pg_catalog.jsonb_array_elements_text(expected_credit_hashes)
    ) THEN
        RAISE EXCEPTION 'chain_realized_credit_set_conflict'
            USING ERRCODE = '40001';
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab_chain_realized_settlement_persistence.v1',
        'netuid',
        settlement_netuid,
        'epoch_id',
        settlement_epoch,
        'settlement_hash',
        requested_settlement_hash,
        'settlement_receipt_hash',
        requested_settlement_receipt_hash,
        'credit_count',
        requested_credit_count,
        'credit_hashes',
        stored_credit_hashes
    );
END;
$$;

REVOKE ALL
    ON FUNCTION
    public.persist_research_lab_chain_realized_settlement_v1(JSONB, JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE
    ON FUNCTION
    public.persist_research_lab_chain_realized_settlement_v1(JSONB, JSONB)
    TO service_role;

COMMIT;
