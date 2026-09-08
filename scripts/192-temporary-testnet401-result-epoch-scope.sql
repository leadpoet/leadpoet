-- Temporary, exact testnet401 epoch scope for replayable coordinator results.
--
-- The production stateful epoch fence is intentionally unchanged.  Only the
-- result and transparency tables are split so the approved testnet401
-- allocation, chain-observation, and signed weight-submission rows are checked
-- against their keyed fresh-network cutover.
-- Remove this function and trigger split after the temporary testnet proof.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION
public.enforce_temporary_testnet401_execution_result_epoch_scope_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    row_doc JSONB := pg_catalog.to_jsonb(NEW);
    result_value JSONB := row_doc->'result_doc';
    source_state JSONB;
    cutover_row RECORD;
    receipt_row RECORD;
    mapping_value JSONB;
BEGIN
    IF pg_catalog.jsonb_typeof(result_value) IS DISTINCT FROM 'object' THEN
        RAISE EXCEPTION 'temporary testnet401 result document is invalid';
    END IF;

    SELECT
        schema_version,
        previous_epoch_scheme,
        network_genesis_hash,
        netuid,
        mapping_hash,
        cutover_authority_hash,
        cutover_receipt_hash,
        first_settlement_epoch_id
    INTO cutover_row
    FROM public.research_lab_stateful_subnet_epoch_cutovers_v1
    WHERE network_genesis_hash =
          '0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105'
      AND netuid = 401;
    IF NOT FOUND
       OR cutover_row.schema_version IS DISTINCT FROM
          'leadpoet.subnet_epoch_cutover_authority.v3'
       OR cutover_row.previous_epoch_scheme IS DISTINCT FROM
          'fresh_network_v1'
       OR cutover_row.mapping_hash IS DISTINCT FROM
          'sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328'
       OR cutover_row.cutover_authority_hash IS DISTINCT FROM
          'sha256:2eb438e142f4c27d5ce28b2e2c5148cc030724a38ab0bf479ce4fc0fade51bc7'
       OR cutover_row.cutover_receipt_hash IS DISTINCT FROM
          'sha256:4db3b2649bbd2182488b3f17cbff87abf04f96c9511055cb51efe743710b6aaa'
       OR cutover_row.first_settlement_epoch_id IS DISTINCT FROM 22042
    THEN
        RAISE EXCEPTION 'temporary testnet401 keyed cutover is unavailable';
    END IF;

    SELECT
        receipt_hash,
        role,
        purpose,
        job_id,
        epoch_id,
        sequence,
        input_root,
        output_root,
        artifact_root,
        receipt_status
    INTO receipt_row
    FROM public.research_lab_attested_execution_receipts_v2
    WHERE receipt_hash = row_doc->>'receipt_hash';
    IF NOT FOUND
       OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
       OR receipt_row.role IS DISTINCT FROM row_doc->>'role'
       OR receipt_row.purpose IS DISTINCT FROM row_doc->>'purpose'
       OR receipt_row.job_id IS DISTINCT FROM row_doc->>'job_id'
       OR receipt_row.epoch_id::BIGINT IS DISTINCT FROM
          (row_doc->>'epoch_id')::BIGINT
       OR receipt_row.sequence IS DISTINCT FROM
          (row_doc->>'sequence')::INTEGER
       OR receipt_row.input_root IS DISTINCT FROM row_doc->>'input_root'
       OR receipt_row.output_root IS DISTINCT FROM row_doc->>'output_root'
       OR receipt_row.artifact_root IS DISTINCT FROM row_doc->>'artifact_root'
    THEN
        RAISE EXCEPTION 'temporary testnet401 result receipt differs';
    END IF;

    IF row_doc->>'schema_version' IS DISTINCT FROM
       'leadpoet.attested_execution_result.v2'
       OR row_doc->>'role' IS DISTINCT FROM 'gateway_coordinator'
       OR COALESCE(row_doc->>'epoch_id', '') !~ '^[0-9]+$'
       OR (row_doc->>'epoch_id')::BIGINT <
          cutover_row.first_settlement_epoch_id::BIGINT
    THEN
        RAISE EXCEPTION 'temporary testnet401 result scope is invalid';
    END IF;

    IF row_doc->>'operation' = 'research_lab_allocation'
       AND row_doc->>'purpose' = 'research_lab.allocation.v2' THEN
        IF result_value - ARRAY[
               'allocation', 'allocation_inputs', 'source_state',
               'source_state_hash'
           ] <> '{}'::JSONB
           OR NOT result_value ?& ARRAY[
               'allocation', 'allocation_inputs', 'source_state',
               'source_state_hash'
           ]
           OR pg_catalog.jsonb_typeof(result_value->'source_state')
              IS DISTINCT FROM 'object'
           OR (result_value #>> '{source_state,netuid}')::INTEGER
              IS DISTINCT FROM 401
           OR (result_value #>> '{source_state,epoch}')::BIGINT
              IS DISTINCT FROM (row_doc->>'epoch_id')::BIGINT
           OR (result_value #>> '{allocation,epoch}')::BIGINT
              IS DISTINCT FROM (row_doc->>'epoch_id')::BIGINT
           OR (result_value #>> '{allocation_inputs,epoch}')::BIGINT
              IS DISTINCT FROM (row_doc->>'epoch_id')::BIGINT
           OR (result_value #>> '{source_state,settlement_frontier,netuid}')::INTEGER
              IS DISTINCT FROM 401
           OR (result_value #>>
               '{source_state,settlement_frontier,allocation_epoch}')::BIGINT
              IS DISTINCT FROM (row_doc->>'epoch_id')::BIGINT
        THEN
            RAISE EXCEPTION 'temporary testnet401 allocation result is invalid';
        END IF;
        source_state := result_value->'source_state';
    ELSIF row_doc->>'operation' = 'observe_chain_realized_weights_v1'
       AND row_doc->>'purpose' =
           'research_lab.chain_weight_observation.v1' THEN
        IF result_value->>'schema_version' IS DISTINCT FROM
           'leadpoet.chain_realized_weight_observation.v2'
           OR (result_value->>'netuid')::INTEGER IS DISTINCT FROM 401
           OR (result_value->>'epoch_id')::BIGINT IS DISTINCT FROM
              (row_doc->>'epoch_id')::BIGINT
           OR result_value->>'validator_hotkey' IS DISTINCT FROM
              '5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz'
           OR (result_value->>'validator_uid')::INTEGER IS DISTINCT FROM 9
           OR result_value->'chain_signing_profile'->>'genesis_hash'
              IS DISTINCT FROM
              '0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105'
        THEN
            RAISE EXCEPTION 'temporary testnet401 chain observation is invalid';
        END IF;
        source_state := result_value;
    ELSE
        RAISE EXCEPTION 'temporary testnet401 result operation is invalid';
    END IF;

    FOR mapping_value IN
        SELECT value
        FROM pg_catalog.jsonb_path_query(
            row_doc,
            'strict $.**.cutover_mapping_hash'
        ) AS mapping_item(value)
    LOOP
        IF pg_catalog.jsonb_typeof(mapping_value) <> 'string'
           OR mapping_value #>> '{}' IS DISTINCT FROM cutover_row.mapping_hash
        THEN
            RAISE EXCEPTION 'temporary testnet401 result has mixed epoch authority';
        END IF;
    END LOOP;

    IF row_doc->>'operation' = 'observe_chain_realized_weights_v1'
       AND source_state->>'cutover_mapping_hash' IS DISTINCT FROM
           cutover_row.mapping_hash THEN
        RAISE EXCEPTION 'temporary testnet401 chain observation lacks epoch authority';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.enforce_temporary_testnet401_execution_result_epoch_scope_v1()
    FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION
public.enforce_temporary_testnet401_weight_submission_epoch_scope_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    row_doc JSONB := pg_catalog.to_jsonb(NEW);
    payload JSONB := row_doc->'payload';
    epoch_authority JSONB;
    cutover_row RECORD;
    mapping_value JSONB;
    mapping_count INTEGER := 0;
BEGIN
    SELECT
        schema_version,
        previous_epoch_scheme,
        mapping_hash,
        cutover_authority_hash,
        cutover_receipt_hash,
        first_settlement_epoch_id
    INTO cutover_row
    FROM public.research_lab_stateful_subnet_epoch_cutovers_v1
    WHERE network_genesis_hash =
          '0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105'
      AND netuid = 401;
    IF NOT FOUND
       OR cutover_row.schema_version IS DISTINCT FROM
          'leadpoet.subnet_epoch_cutover_authority.v3'
       OR cutover_row.previous_epoch_scheme IS DISTINCT FROM
          'fresh_network_v1'
       OR cutover_row.mapping_hash IS DISTINCT FROM
          'sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328'
       OR cutover_row.cutover_authority_hash IS DISTINCT FROM
          'sha256:2eb438e142f4c27d5ce28b2e2c5148cc030724a38ab0bf479ce4fc0fade51bc7'
       OR cutover_row.cutover_receipt_hash IS DISTINCT FROM
          'sha256:4db3b2649bbd2182488b3f17cbff87abf04f96c9511055cb51efe743710b6aaa'
       OR cutover_row.first_settlement_epoch_id IS DISTINCT FROM 22042
    THEN
        RAISE EXCEPTION 'temporary testnet401 keyed cutover is unavailable';
    END IF;

    epoch_authority := payload->'epoch_authority';
    IF row_doc->>'event_type' IS DISTINCT FROM 'WEIGHT_SUBMISSION_V2'
       OR pg_catalog.jsonb_typeof(payload) IS DISTINCT FROM 'object'
       OR payload - ARRAY[
           'actor_hotkey', 'netuid', 'epoch_id', 'block', 'weights_hash',
           'bundle_hash', 'root_receipt_hash', 'epoch_authority'
       ] <> '{}'::JSONB
       OR NOT payload ?& ARRAY[
           'actor_hotkey', 'netuid', 'epoch_id', 'block', 'weights_hash',
           'bundle_hash', 'root_receipt_hash', 'epoch_authority'
       ]
       OR payload->>'actor_hotkey' IS DISTINCT FROM
          '5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz'
       OR row_doc->>'actor_hotkey' IS DISTINCT FROM
          payload->>'actor_hotkey'
       OR (payload->>'netuid')::INTEGER IS DISTINCT FROM 401
       OR COALESCE(payload->>'epoch_id', '') !~ '^[0-9]+$'
       OR (payload->>'epoch_id')::BIGINT <
          cutover_row.first_settlement_epoch_id::BIGINT
       OR pg_catalog.jsonb_typeof(epoch_authority) IS DISTINCT FROM 'object'
       OR epoch_authority->>'mode' IS DISTINCT FROM 'stateful_v1'
       OR (epoch_authority->>'workflow_epoch_id')::BIGINT IS DISTINCT FROM
          (payload->>'epoch_id')::BIGINT
       OR epoch_authority->>'cutover_mapping_hash' IS DISTINCT FROM
          cutover_row.mapping_hash
       OR COALESCE(payload->>'block', '') !~ '^[0-9]+$'
       OR COALESCE(payload->>'weights_hash', '') !~ '^[0-9a-f]{64}$'
       OR COALESCE(payload->>'bundle_hash', '') !~
          '^sha256:[0-9a-f]{64}$'
       OR COALESCE(payload->>'root_receipt_hash', '') !~
          '^sha256:[0-9a-f]{64}$'
       OR COALESCE(row_doc->>'payload_hash', '') !~ '^[0-9a-f]{64}$'
       OR COALESCE(row_doc->>'event_hash', '') !~ '^[0-9a-f]{64}$'
       OR COALESCE(row_doc->>'enclave_pubkey', '') !~ '^[0-9a-f]{64}$'
       OR COALESCE(row_doc->>'signature', '') !~ '^[0-9a-f]{128}$'
       OR COALESCE(row_doc->>'boot_id', '') = ''
       OR COALESCE(row_doc->>'monotonic_seq', '') !~ '^[0-9]+$'
       OR pg_catalog.jsonb_typeof(row_doc->'signed_log_entry')
          IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(
              row_doc->'signed_log_entry'->'signed_event'
          ) IS DISTINCT FROM 'object'
       OR row_doc->'epoch_id' IS DISTINCT FROM 'null'::JSONB
       OR row_doc #>> '{signed_log_entry,signed_event,event_type}'
          IS DISTINCT FROM row_doc->>'event_type'
       OR row_doc #> '{signed_log_entry,signed_event,payload}'
          IS DISTINCT FROM payload
       OR row_doc #>> '{signed_log_entry,enclave_signature}'
          IS DISTINCT FROM row_doc->>'signature'
       OR row_doc #>> '{signed_log_entry,event_hash}'
          IS DISTINCT FROM row_doc->>'event_hash'
       OR row_doc #>> '{signed_log_entry,enclave_pubkey}'
          IS DISTINCT FROM row_doc->>'enclave_pubkey'
       OR row_doc #>> '{signed_log_entry,signed_event,boot_id}'
          IS DISTINCT FROM row_doc->>'boot_id'
       OR row_doc #>> '{signed_log_entry,signed_event,monotonic_seq}'
          IS DISTINCT FROM row_doc->>'monotonic_seq'
       OR row_doc #>> '{signed_log_entry,signed_event,prev_event_hash}'
          IS DISTINCT FROM row_doc->>'prev_event_hash'
    THEN
        RAISE EXCEPTION 'temporary testnet401 weight submission event is invalid';
    END IF;

    FOR mapping_value IN
        SELECT value
        FROM pg_catalog.jsonb_path_query(
            row_doc,
            'strict $.**.cutover_mapping_hash'
        ) AS mapping_item(value)
    LOOP
        mapping_count := mapping_count + 1;
        IF pg_catalog.jsonb_typeof(mapping_value) <> 'string'
           OR mapping_value #>> '{}' IS DISTINCT FROM cutover_row.mapping_hash
        THEN
            RAISE EXCEPTION 'temporary testnet401 result has mixed epoch authority';
        END IF;
    END LOOP;

    IF mapping_count < 1 THEN
        RAISE EXCEPTION 'temporary testnet401 weight submission lacks epoch authority';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.enforce_temporary_testnet401_weight_submission_epoch_scope_v1()
    FROM PUBLIC, anon, authenticated;

DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.research_lab_attested_execution_results_v2;
DROP TRIGGER IF EXISTS enforce_temporary_testnet401_execution_result_epoch_scope_v1
    ON public.research_lab_attested_execution_results_v2;

CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.research_lab_attested_execution_results_v2
    FOR EACH ROW
    WHEN (
        NOT (
            COALESCE((
                NEW.operation = 'research_lab_allocation'
                AND NEW.result_doc #>> '{source_state,netuid}' = '401'
            ), FALSE)
            OR COALESCE((
                NEW.operation = 'observe_chain_realized_weights_v1'
                AND NEW.result_doc->>'netuid' = '401'
            ), FALSE)
            OR pg_catalog.jsonb_path_exists(
                NEW.result_doc,
                'strict $.**.cutover_mapping_hash ? (@ == "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328")'
            )
        )
    )
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

CREATE TRIGGER enforce_temporary_testnet401_execution_result_epoch_scope_v1
    BEFORE INSERT OR UPDATE
    ON public.research_lab_attested_execution_results_v2
    FOR EACH ROW
    WHEN (
        COALESCE((
            NEW.operation = 'research_lab_allocation'
            AND NEW.result_doc #>> '{source_state,netuid}' = '401'
        ), FALSE)
        OR COALESCE((
            NEW.operation = 'observe_chain_realized_weights_v1'
            AND NEW.result_doc->>'netuid' = '401'
        ), FALSE)
        OR pg_catalog.jsonb_path_exists(
            NEW.result_doc,
            'strict $.**.cutover_mapping_hash ? (@ == "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328")'
        )
    )
    EXECUTE FUNCTION
        public.enforce_temporary_testnet401_execution_result_epoch_scope_v1();

DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.transparency_log;
DROP TRIGGER IF EXISTS enforce_temporary_testnet401_weight_submission_epoch_scope_v1
    ON public.transparency_log;

CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.transparency_log
    FOR EACH ROW
    WHEN (
        NOT (
            COALESCE((
                NEW.event_type = 'WEIGHT_SUBMISSION_V2'
                AND NEW.payload->>'netuid' = '401'
            ), FALSE)
            OR pg_catalog.jsonb_path_exists(
                NEW.payload,
                'strict $.**.cutover_mapping_hash ? (@ == "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328")'
            )
        )
    )
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

CREATE TRIGGER enforce_temporary_testnet401_weight_submission_epoch_scope_v1
    BEFORE INSERT OR UPDATE
    ON public.transparency_log
    FOR EACH ROW
    WHEN (
        COALESCE((
            NEW.event_type = 'WEIGHT_SUBMISSION_V2'
            AND NEW.payload->>'netuid' = '401'
        ), FALSE)
        OR pg_catalog.jsonb_path_exists(
            NEW.payload,
            'strict $.**.cutover_mapping_hash ? (@ == "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328")'
        )
    )
    EXECUTE FUNCTION
        public.enforce_temporary_testnet401_weight_submission_epoch_scope_v1();

COMMENT ON FUNCTION
    public.enforce_temporary_testnet401_execution_result_epoch_scope_v1() IS
    'Temporary exact testnet401 result fence. Remove after automatic native weight proof and restore the canonical result-table fence trigger.';

COMMENT ON FUNCTION
    public.enforce_temporary_testnet401_weight_submission_epoch_scope_v1() IS
    'Temporary exact testnet401 signed weight-submission fence. Remove after automatic native weight proof and restore the canonical transparency fence trigger.';

NOTIFY pgrst, 'reload schema';

COMMIT;
