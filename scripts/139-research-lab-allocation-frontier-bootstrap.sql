-- Measured, bounded activation of the allocation settlement frontier.
--
-- The coordinator converts the latest signed allocation source state into the
-- first recursive settlement frontier.  The conversion consumes only
-- checkpoint-certified ancestry and immutable reward-decision receipts; it
-- never replays the unbounded historical allocation graph on the host.
-- Existing regular-allocation frontier persistence remains unchanged.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_execution_results_v2'::REGCLASS
          AND contype = 'c'
          AND (
              pg_catalog.pg_get_constraintdef(oid) ~ '\moperation\M'
              OR pg_catalog.pg_get_constraintdef(oid) ~ '\mpurpose\M'
          )
    LOOP
        EXECUTE pg_catalog.format(
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
            'allocation_settlement_frontier_bootstrap_v2',
            'attest_weight_input',
            'attest_active_private_model',
            'observe_chain_realized_weights_v1',
            'attest_chain_realized_settlement_v1'
        )
    ) NOT VALID,
    ADD CONSTRAINT
        research_lab_attested_execution_results_v2_purpose_check
    CHECK (
        purpose IN (
            'research_lab.allocation.v2',
            'research_lab.allocation_settlement_frontier_bootstrap.v2',
            'research_lab.champion_input.v2',
            'research_lab.reimbursement_input.v2',
            'research_lab.source_add_reward_input.v2',
            'research_lab.sourcing_input.v2',
            'research_lab.fulfillment_input.v2',
            'research_lab.leaderboard_input.v2',
            'research_lab.ban_input.v2',
            'research_lab.anomaly_adjustment_input.v2',
            'research_lab.active_private_model.v2',
            'research_lab.chain_weight_observation.v1',
            'research_lab.chain_realized_epoch_settlement.v1'
        )
    ) NOT VALID,
    ADD CONSTRAINT
        research_lab_attested_exec_results_v2_op_purpose_check
    CHECK (
        (
            operation = 'research_lab_allocation'
            AND purpose = 'research_lab.allocation.v2'
        )
        OR (
            operation = 'allocation_settlement_frontier_bootstrap_v2'
            AND purpose =
                'research_lab.allocation_settlement_frontier_bootstrap.v2'
        )
        OR (
            operation = 'attest_weight_input'
            AND purpose IN (
                'research_lab.allocation.v2',
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
            operation = 'attest_active_private_model'
            AND purpose = 'research_lab.active_private_model.v2'
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
        research_lab_attested_exec_results_v2_op_purpose_check;

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_execution_receipts_v2'::REGCLASS
          AND contype = 'c'
          AND pg_catalog.pg_get_constraintdef(oid)
              LIKE '%gateway_coordinator%'
          AND pg_catalog.pg_get_constraintdef(oid) LIKE '%purpose%'
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_attested_execution_receipts_v2 '
            'DROP CONSTRAINT %I',
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
            'research_lab.ancestry_checkpoint_bootstrap.v2',
            'research_lab.allocation_settlement_frontier_bootstrap.v2',
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

CREATE OR REPLACE FUNCTION
public.persist_research_lab_allocation_frontier_bootstrap_v2(
    requested_frontier JSONB,
    requested_source_receipt_hash TEXT,
    requested_source_state_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    requested_netuid INTEGER;
    requested_epoch BIGINT;
    requested_settled_through BIGINT;
    requested_checkpoint_count INTEGER;
    observed_checkpoint_count INTEGER;
    bootstrap_execution public.research_lab_attested_execution_results_v2;
    bootstrap_receipt public.research_lab_attested_execution_receipts_v2;
    allocation_execution public.research_lab_attested_execution_results_v2;
    allocation_receipt public.research_lab_attested_execution_receipts_v2;
    existing_row public.research_lab_allocation_settlement_frontiers_v2;
    activation_row
        public.research_lab_allocation_settlement_frontier_activation_v2;
    bootstrap_doc JSONB;
    allocation_receipt_hash TEXT;
BEGIN
    IF requested_frontier IS NULL
       OR pg_catalog.jsonb_typeof(requested_frontier) IS DISTINCT FROM 'object'
       OR (
           SELECT pg_catalog.array_agg(key ORDER BY key)
             FROM pg_catalog.jsonb_object_keys(requested_frontier) AS key
       ) <> ARRAY[
           'allocation_epoch',
           'frontier_hash',
           'mode',
           'netuid',
           'predecessor_frontier_hash',
           'reward_checkpoint_count',
           'reward_checkpoint_hashes_root',
           'reward_checkpoints',
           'schema_version',
           'settled_through_epoch'
       ]::TEXT[] THEN
        RAISE EXCEPTION 'allocation_frontier_bootstrap_request_invalid'
            USING ERRCODE = '22023';
    END IF;
    BEGIN
        requested_netuid := (requested_frontier->>'netuid')::INTEGER;
        requested_epoch :=
            (requested_frontier->>'allocation_epoch')::BIGINT;
        requested_settled_through :=
            (requested_frontier->>'settled_through_epoch')::BIGINT;
        requested_checkpoint_count :=
            (requested_frontier->>'reward_checkpoint_count')::INTEGER;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'allocation_frontier_bootstrap_scope_invalid'
            USING ERRCODE = '22023';
    END;
    IF requested_netuid IS NULL
       OR requested_netuid <= 0
       OR requested_epoch IS NULL
       OR requested_epoch < 1
       OR requested_settled_through <> requested_epoch - 1
       OR requested_frontier->>'schema_version' IS DISTINCT FROM
          'leadpoet.research_lab_allocation_settlement_frontier.v2'
       OR requested_frontier->>'mode' IS DISTINCT FROM
          'legacy_full_history_bootstrap'
       OR requested_frontier->'predecessor_frontier_hash'
          IS DISTINCT FROM 'null'::JSONB
       OR requested_frontier->>'frontier_hash' IS NULL
       OR requested_frontier->>'frontier_hash'
          !~ '^sha256:[0-9a-f]{64}$'
       OR requested_source_receipt_hash IS NULL
       OR requested_source_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_source_state_hash IS NULL
       OR requested_source_state_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_checkpoint_count IS NULL
       OR requested_checkpoint_count < 0
       OR requested_checkpoint_count > 512
       OR pg_catalog.jsonb_typeof(
          requested_frontier->'reward_checkpoints'
       ) IS DISTINCT FROM 'array'
       OR requested_frontier->>'reward_checkpoint_hashes_root' IS NULL
       OR requested_frontier->>'reward_checkpoint_hashes_root'
          !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'allocation_frontier_bootstrap_request_invalid'
            USING ERRCODE = '22023';
    END IF;
    SELECT pg_catalog.count(*)::INTEGER
      INTO observed_checkpoint_count
      FROM pg_catalog.jsonb_array_elements(
          requested_frontier->'reward_checkpoints'
      ) AS checkpoint;
    IF observed_checkpoint_count <> requested_checkpoint_count
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements(
                 requested_frontier->'reward_checkpoints'
             ) AS checkpoint
            WHERE pg_catalog.jsonb_typeof(checkpoint) IS DISTINCT FROM 'object'
               OR checkpoint->>'schema_version' IS DISTINCT FROM
                  'leadpoet.research_lab_reward_settlement_checkpoint.v2'
               OR checkpoint->>'checkpoint_hash' IS NULL
               OR checkpoint->>'checkpoint_hash'
                  !~ '^sha256:[0-9a-f]{64}$'
       )
       OR (
           SELECT pg_catalog.count(DISTINCT checkpoint->>'checkpoint_hash')
             FROM pg_catalog.jsonb_array_elements(
                 requested_frontier->'reward_checkpoints'
             ) AS checkpoint
       ) <> requested_checkpoint_count THEN
        RAISE EXCEPTION 'allocation_frontier_bootstrap_checkpoint_invalid'
            USING ERRCODE = '22023';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(139, requested_netuid);

    SELECT * INTO bootstrap_execution
      FROM public.research_lab_attested_execution_results_v2
     WHERE receipt_hash = requested_source_receipt_hash;
    SELECT * INTO bootstrap_receipt
      FROM public.research_lab_attested_execution_receipts_v2
     WHERE receipt_hash = requested_source_receipt_hash;
    bootstrap_doc := bootstrap_execution.result_doc;
    IF bootstrap_execution.receipt_hash IS NULL
       OR bootstrap_receipt.receipt_hash IS NULL
       OR bootstrap_execution.role <> 'gateway_coordinator'
       OR bootstrap_execution.operation <>
          'allocation_settlement_frontier_bootstrap_v2'
       OR bootstrap_execution.purpose <>
          'research_lab.allocation_settlement_frontier_bootstrap.v2'
       OR bootstrap_receipt.role <> bootstrap_execution.role
       OR bootstrap_receipt.purpose <> bootstrap_execution.purpose
       OR bootstrap_receipt.job_id <> bootstrap_execution.job_id
       OR bootstrap_receipt.epoch_id <> bootstrap_execution.epoch_id
       OR bootstrap_receipt.sequence <> bootstrap_execution.sequence
       OR bootstrap_receipt.input_root <> bootstrap_execution.input_root
       OR bootstrap_receipt.output_root <> bootstrap_execution.output_root
       OR bootstrap_receipt.artifact_root <> bootstrap_execution.artifact_root
       OR bootstrap_receipt.receipt_status <> 'succeeded'
       OR pg_catalog.jsonb_typeof(bootstrap_doc) IS DISTINCT FROM 'object'
       OR (
           SELECT pg_catalog.array_agg(key ORDER BY key)
             FROM pg_catalog.jsonb_object_keys(bootstrap_doc) AS key
       ) <> ARRAY[
           'allocation_epoch',
           'allocation_source_receipt_hash',
           'bootstrap_epoch',
           'bootstrap_hash',
           'frontier',
           'netuid',
           'schema_version',
           'source_state_hash'
       ]::TEXT[]
       OR bootstrap_doc->>'schema_version' IS DISTINCT FROM
          'leadpoet.research_lab_allocation_settlement_frontier_bootstrap.v2'
       OR (bootstrap_doc->>'netuid')::INTEGER <> requested_netuid
       OR (bootstrap_doc->>'allocation_epoch')::BIGINT <> requested_epoch
       OR (bootstrap_doc->>'bootstrap_epoch')::BIGINT < requested_epoch
       OR (bootstrap_doc->>'bootstrap_epoch')::BIGINT <>
          bootstrap_execution.epoch_id
       OR bootstrap_doc->>'source_state_hash' <>
          requested_source_state_hash
       OR bootstrap_doc->'frontier' <> requested_frontier
       OR bootstrap_doc->>'bootstrap_hash' IS NULL
       OR bootstrap_doc->>'bootstrap_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR NOT (
          bootstrap_execution.artifact_hashes ?
          (bootstrap_doc->>'bootstrap_hash')
       )
       OR NOT (
          bootstrap_execution.artifact_hashes ? requested_source_state_hash
       )
       OR NOT (
          bootstrap_execution.artifact_hashes ?
          (requested_frontier->>'frontier_hash')
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements(
                 requested_frontier->'reward_checkpoints'
             ) AS checkpoint
            WHERE NOT (
                bootstrap_execution.artifact_hashes ?
                (checkpoint->>'checkpoint_hash')
            )
       ) THEN
        RAISE EXCEPTION 'allocation_frontier_bootstrap_authority_invalid'
            USING ERRCODE = '23514';
    END IF;

    allocation_receipt_hash :=
        bootstrap_doc->>'allocation_source_receipt_hash';
    SELECT * INTO allocation_execution
      FROM public.research_lab_attested_execution_results_v2
     WHERE receipt_hash = allocation_receipt_hash;
    SELECT * INTO allocation_receipt
      FROM public.research_lab_attested_execution_receipts_v2
     WHERE receipt_hash = allocation_receipt_hash;
    IF allocation_receipt_hash IS NULL
       OR allocation_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR allocation_execution.receipt_hash IS NULL
       OR allocation_receipt.receipt_hash IS NULL
       OR allocation_execution.role <> 'gateway_coordinator'
       OR allocation_execution.operation <> 'research_lab_allocation'
       OR allocation_execution.purpose <> 'research_lab.allocation.v2'
       OR allocation_execution.epoch_id <> requested_epoch
       OR allocation_execution.result_doc->>'source_state_hash' <>
          requested_source_state_hash
       OR (allocation_execution.result_doc #>> '{source_state,netuid}')::INTEGER
          <> requested_netuid
       OR (allocation_execution.result_doc #>> '{source_state,epoch}')::BIGINT
          <> requested_epoch
       OR allocation_execution.result_doc->'source_state'->
          'settlement_frontier' IS DISTINCT FROM 'null'::JSONB
       OR NOT (
          allocation_execution.artifact_hashes ? requested_source_state_hash
       )
       OR allocation_receipt.role <> allocation_execution.role
       OR allocation_receipt.purpose <> allocation_execution.purpose
       OR allocation_receipt.job_id <> allocation_execution.job_id
       OR allocation_receipt.epoch_id <> allocation_execution.epoch_id
       OR allocation_receipt.sequence <> allocation_execution.sequence
       OR allocation_receipt.input_root <> allocation_execution.input_root
       OR allocation_receipt.output_root <> allocation_execution.output_root
       OR allocation_receipt.artifact_root <> allocation_execution.artifact_root
       OR allocation_receipt.receipt_status <> 'succeeded'
       OR pg_catalog.jsonb_typeof(
          bootstrap_receipt.receipt_doc->'parent_receipt_hashes'
       ) IS DISTINCT FROM 'array'
       OR NOT (
          bootstrap_receipt.receipt_doc->'parent_receipt_hashes' ?
          allocation_receipt_hash
       ) THEN
        RAISE EXCEPTION 'allocation_frontier_bootstrap_source_invalid'
            USING ERRCODE = '23514';
    END IF;

    SELECT * INTO activation_row
      FROM public.research_lab_allocation_settlement_frontier_activation_v2
     WHERE netuid = requested_netuid;
    SELECT * INTO existing_row
      FROM public.research_lab_allocation_settlement_frontiers_v2
     WHERE netuid = requested_netuid
       AND allocation_epoch = requested_epoch;
    IF existing_row.netuid IS NOT NULL THEN
        IF activation_row.netuid IS NULL
           OR activation_row.first_allocation_epoch <> requested_epoch
           OR activation_row.first_frontier_hash <>
              requested_frontier->>'frontier_hash'
           OR activation_row.source_receipt_hash <>
              requested_source_receipt_hash
           OR existing_row.frontier_doc <> requested_frontier
           OR existing_row.source_receipt_hash <>
              requested_source_receipt_hash
           OR existing_row.source_state_hash <> requested_source_state_hash THEN
            RAISE EXCEPTION 'allocation_frontier_bootstrap_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'status', 'already_persisted',
            'netuid', existing_row.netuid,
            'allocation_epoch', existing_row.allocation_epoch,
            'frontier_hash', existing_row.frontier_hash,
            'source_receipt_hash', existing_row.source_receipt_hash,
            'source_state_hash', existing_row.source_state_hash
        );
    END IF;
    IF activation_row.netuid IS NOT NULL
       OR EXISTS (
           SELECT 1
             FROM public.research_lab_allocation_settlement_frontiers_v2
            WHERE netuid = requested_netuid
       ) THEN
        RAISE EXCEPTION 'allocation_frontier_bootstrap_already_initialized'
            USING ERRCODE = '23514';
    END IF;

    INSERT INTO public.research_lab_allocation_settlement_frontiers_v2 (
        netuid,
        allocation_epoch,
        settled_through_epoch,
        schema_version,
        frontier_hash,
        predecessor_frontier_hash,
        source_receipt_hash,
        source_state_hash,
        frontier_doc
    ) VALUES (
        requested_netuid,
        requested_epoch,
        requested_settled_through,
        requested_frontier->>'schema_version',
        requested_frontier->>'frontier_hash',
        NULL,
        requested_source_receipt_hash,
        requested_source_state_hash,
        requested_frontier
    );
    INSERT INTO
        public.research_lab_allocation_settlement_frontier_activation_v2 (
            netuid,
            schema_version,
            first_allocation_epoch,
            first_frontier_hash,
            source_receipt_hash
        ) VALUES (
            requested_netuid,
            'leadpoet.research_lab_allocation_settlement_frontier_activation.v2',
            requested_epoch,
            requested_frontier->>'frontier_hash',
            requested_source_receipt_hash
        );
    RETURN pg_catalog.jsonb_build_object(
        'status', 'persisted',
        'netuid', requested_netuid,
        'allocation_epoch', requested_epoch,
        'frontier_hash', requested_frontier->>'frontier_hash',
        'source_receipt_hash', requested_source_receipt_hash,
        'source_state_hash', requested_source_state_hash
    );
END;
$$;

CREATE OR REPLACE FUNCTION
public.research_lab_allocation_frontier_bootstrap_contract_v2()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.allocation_frontier_bootstrap_contract.v2',
        'operation', 'allocation_settlement_frontier_bootstrap_v2',
        'purpose',
        'research_lab.allocation_settlement_frontier_bootstrap.v2',
        'persistence_rpc',
        'persist_research_lab_allocation_frontier_bootstrap_v2',
        'constraints', (
            SELECT pg_catalog.jsonb_object_agg(
                constraint_record.conname,
                pg_catalog.jsonb_build_object(
                    'constraint_definition',
                    pg_catalog.pg_get_constraintdef(constraint_record.oid),
                    'constraint_valid', constraint_record.convalidated
                )
                ORDER BY constraint_record.conname
            )
            FROM pg_catalog.pg_constraint AS constraint_record
            WHERE constraint_record.conname IN (
                'research_lab_attested_execution_results_v2_operation_check',
                'research_lab_attested_execution_results_v2_purpose_check',
                'research_lab_attested_exec_results_v2_op_purpose_check',
                'research_lab_attested_execution_receipts_v2_role_purpose_check'
            )
              AND constraint_record.conrelid IN (
                  'public.research_lab_attested_execution_results_v2'::REGCLASS,
                  'public.research_lab_attested_execution_receipts_v2'::REGCLASS
              )
        )
    );
$$;

REVOKE ALL ON FUNCTION
    public.persist_research_lab_allocation_frontier_bootstrap_v2(
        JSONB,
        TEXT,
        TEXT
    )
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.persist_research_lab_allocation_frontier_bootstrap_v2(
        JSONB,
        TEXT,
        TEXT
    )
    TO service_role;
REVOKE ALL ON FUNCTION
    public.research_lab_allocation_frontier_bootstrap_contract_v2()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_allocation_frontier_bootstrap_contract_v2()
    TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Verify after applying:
-- SELECT public.research_lab_allocation_frontier_bootstrap_contract_v2();
