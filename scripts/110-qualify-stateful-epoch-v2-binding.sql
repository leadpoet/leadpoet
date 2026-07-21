-- Repair migration 105's V2 cutover binding RPC. Its RETURNS TABLE output
-- variable named mapping_hash made an unqualified candidate lookup ambiguous.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION
public.research_lab_stateful_subnet_epoch_cutover_bind_v2(
    p_mapping_hash TEXT,
    p_cutover_authority_hash TEXT,
    p_last_legacy_finalization_receipt_hash TEXT,
    p_cutover_receipt_hash TEXT DEFAULT NULL
)
RETURNS TABLE (
    lifecycle_state TEXT,
    mapping_hash TEXT,
    legacy_high_water BIGINT,
    last_legacy_epoch_id BIGINT,
    first_settlement_epoch_id BIGINT,
    candidate_snapshot_hash TEXT,
    candidate_receipt_hash TEXT,
    cutover_authority_hash TEXT,
    cutover_receipt_hash TEXT,
    initialization_nonce UUID,
    initialization_payload_hash TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    candidate_row RECORD;
    predecessor_row RECORD;
    coordinator_receipt RECORD;
BEGIN
    IF p_mapping_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_cutover_authority_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_last_legacy_finalization_receipt_hash
          !~ '^sha256:[0-9a-f]{64}$'
       OR (
           p_cutover_receipt_hash IS NOT NULL
           AND p_cutover_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       ) THEN
        RAISE EXCEPTION 'stateful epoch V2 binding hash is invalid';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1 AS state
    WHERE state.singleton
    FOR UPDATE;
    IF NOT FOUND OR state_row.lifecycle_state <> 'cutover_fenced' THEN
        RAISE EXCEPTION
            'stateful epoch V2 binding requires the pre-boundary fence';
    END IF;

    SELECT * INTO candidate_row
    FROM public.research_lab_stateful_subnet_epoch_candidates_v1 AS candidate
    WHERE candidate.mapping_hash = p_mapping_hash;
    IF NOT FOUND
       OR candidate_row.network_genesis_hash IS DISTINCT FROM
          state_row.network_genesis_hash
       OR candidate_row.netuid IS DISTINCT FROM state_row.netuid
       OR candidate_row.proposed_settlement_epoch_id IS DISTINCT FROM
          state_row.first_settlement_epoch_id THEN
        RAISE EXCEPTION
            'stateful epoch V2 binding candidate differs from fence';
    END IF;

    SELECT migration.netuid, migration.epoch_id, receipt.role,
           receipt.purpose, receipt.receipt_status
    INTO predecessor_row
    FROM public.research_lab_legacy_finalized_allocation_migrations_v2
         AS migration
    JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
      ON receipt.receipt_hash = migration.settlement_receipt_hash
    WHERE migration.settlement_receipt_hash =
          p_last_legacy_finalization_receipt_hash;
    IF NOT FOUND
       OR predecessor_row.netuid IS DISTINCT FROM state_row.netuid
       OR predecessor_row.epoch_id > state_row.last_legacy_epoch_id
       OR predecessor_row.role IS DISTINCT FROM 'gateway_coordinator'
       OR predecessor_row.purpose IS DISTINCT FROM
          'research_lab.legacy_finalized_allocation.v2'
       OR predecessor_row.receipt_status IS DISTINCT FROM 'succeeded' THEN
        RAISE EXCEPTION
            'stateful epoch V2 historical predecessor is invalid';
    END IF;

    IF p_cutover_receipt_hash IS NOT NULL THEN
        SELECT receipt.role, receipt.purpose, receipt.epoch_id,
               receipt.receipt_status, receipt.output_root,
               receipt.receipt_doc
        INTO coordinator_receipt
        FROM public.research_lab_attested_execution_receipts_v2 AS receipt
        WHERE receipt.receipt_hash = p_cutover_receipt_hash;
        IF NOT FOUND
           OR coordinator_receipt.role IS DISTINCT FROM 'gateway_coordinator'
           OR coordinator_receipt.purpose IS DISTINCT FROM
              'research_lab.subnet_epoch_cutover.v2'
           OR coordinator_receipt.epoch_id IS DISTINCT FROM
              state_row.first_settlement_epoch_id
           OR coordinator_receipt.receipt_status IS DISTINCT FROM 'succeeded'
           OR coordinator_receipt.output_root IS DISTINCT FROM
              p_cutover_authority_hash
           OR pg_catalog.jsonb_array_length(
                  coordinator_receipt.receipt_doc->'parent_receipt_hashes'
              ) IS DISTINCT FROM 2
           OR NOT (
               coordinator_receipt.receipt_doc->'parent_receipt_hashes'
               @> pg_catalog.jsonb_build_array(
                   candidate_row.chain_state_receipt_hash,
                   p_last_legacy_finalization_receipt_hash
               )
           ) THEN
            RAISE EXCEPTION
                'stateful epoch V2 coordinator receipt is invalid';
        END IF;
    END IF;

    IF state_row.mapping_hash IS NULL THEN
        UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
            AS state
        SET mapping_hash = candidate_row.mapping_hash,
            candidate_snapshot_hash = candidate_row.snapshot_hash,
            candidate_receipt_hash = candidate_row.chain_state_receipt_hash,
            last_legacy_finalization_receipt_hash =
                p_last_legacy_finalization_receipt_hash,
            cutover_authority_hash = p_cutover_authority_hash,
            updated_at = pg_catalog.clock_timestamp()
        WHERE state.singleton
        RETURNING state.* INTO state_row;
    ELSIF state_row.mapping_hash IS DISTINCT FROM p_mapping_hash
       OR state_row.candidate_snapshot_hash IS DISTINCT FROM
          candidate_row.snapshot_hash
       OR state_row.candidate_receipt_hash IS DISTINCT FROM
          candidate_row.chain_state_receipt_hash
       OR state_row.last_legacy_finalization_receipt_hash IS DISTINCT FROM
          p_last_legacy_finalization_receipt_hash
       OR state_row.cutover_authority_hash IS DISTINCT FROM
          p_cutover_authority_hash THEN
        RAISE EXCEPTION
            'stateful epoch V2 binding conflicts with durable plan';
    END IF;

    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.mapping_hash,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.first_settlement_epoch_id::BIGINT,
        state_row.candidate_snapshot_hash,
        state_row.candidate_receipt_hash,
        state_row.cutover_authority_hash,
        state_row.cutover_receipt_hash,
        state_row.initialization_nonce,
        state_row.initialization_payload_hash;
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_stateful_subnet_epoch_cutover_bind_v2(
        TEXT, TEXT, TEXT, TEXT
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_stateful_subnet_epoch_cutover_bind_v2(
        TEXT, TEXT, TEXT, TEXT
    ) TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;
