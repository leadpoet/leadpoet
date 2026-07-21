-- Canonicalize the staged cutover's first_observed_at as its exact "Z" string.
--
-- The atomic staging RPCs (101 stage_v1, 105 stage_v2) verify the staged
-- cutover row with an exact JSONB round-trip:
--
--     to_jsonb(jsonb_populate_record(NULL::cutovers, p_cutover_row))
--         - 'created_at' IS DISTINCT FROM p_cutover_row
--
-- and read the durable row back with the same equality. The canonical
-- snapshot timestamp everywhere else in the system is the exact string
-- YYYY-MM-DDTHH24:MI:SS"Z" (read_subnet_epoch_snapshot, receipt documents,
-- snapshot hashes), but a TIMESTAMPTZ column re-serializes that value as
-- "+00:00" under to_jsonb, so every staged row fails closed with
-- 'stateful epoch V2 cutover row shape is invalid' before the fence can
-- advance to stateful_staged. Store the canonical string itself; validators
-- that need the instant cast explicitly. The semantic candidate/boundary
-- columns stay TIMESTAMPTZ and are compared through explicit casts.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
    row_count BIGINT;
    check_names TEXT[];
BEGIN
    SELECT COUNT(*) INTO row_count
    FROM public.research_lab_stateful_subnet_epoch_cutovers_v1;
    IF row_count <> 0 THEN
        RAISE EXCEPTION
            'cutover observed_at canonicalization requires an empty cutover table';
    END IF;

    -- Exactly one CHECK compares through TIMESTAMPTZ semantics today
    -- (101: (first_snapshot_doc->>'observed_at')::TIMESTAMPTZ =
    -- first_observed_at). It must be dropped before the type change; the
    -- replacement below is strictly stronger (same string, same format).
    SELECT ARRAY_AGG(conname ORDER BY conname) INTO check_names
    FROM pg_catalog.pg_constraint
    WHERE conrelid =
          'public.research_lab_stateful_subnet_epoch_cutovers_v1'::regclass
      AND contype = 'c'
      AND pg_catalog.pg_get_constraintdef(oid) LIKE '%first_observed_at%';
    IF check_names IS NULL OR array_length(check_names, 1) <> 1 THEN
        RAISE EXCEPTION
            'expected exactly one first_observed_at CHECK, found %',
            COALESCE(array_length(check_names, 1), 0);
    END IF;
    EXECUTE pg_catalog.format(
        'ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 '
        'DROP CONSTRAINT %I',
        check_names[1]
    );
END;
$$;

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ALTER COLUMN first_observed_at TYPE TEXT
    USING to_char(
        first_observed_at AT TIME ZONE 'UTC',
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'
    );

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_observed_at_format_v1
    CHECK (
        first_observed_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
        AND (first_observed_at::TIMESTAMPTZ) IS NOT NULL
    );

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_observed_at_doc_v1
    CHECK ((first_snapshot_doc->>'observed_at') = first_observed_at);

-- Re-pin the cutover INSERT trigger from migration 105 with the only change
-- being an explicit cast: the candidate ledger keeps TIMESTAMPTZ semantics,
-- the cutover row keeps the canonical string, and the trigger compares the
-- instant they both denote.
CREATE OR REPLACE FUNCTION
public.validate_research_lab_stateful_epoch_cutover_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    receipt_row RECORD;
    candidate_row RECORD;
    predecessor_row RECORD;
    finalization_row RECORD;
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    predecessor_receipt TEXT;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    PERFORM pg_catalog.pg_advisory_xact_lock(
        7101,
        pg_catalog.hashtext(NEW.first_epoch_ref)
    );

    SELECT role, purpose, epoch_id, receipt_status, output_root
    INTO receipt_row
    FROM public.research_lab_attested_execution_receipts_v2
    WHERE receipt_hash = NEW.first_snapshot_receipt_hash;
    IF NOT FOUND
       OR receipt_row.role IS DISTINCT FROM 'validator_weights'
       OR receipt_row.purpose IS DISTINCT FROM
          'validator.subnet_epoch_snapshot.v2'
       OR receipt_row.epoch_id IS DISTINCT FROM
          NEW.first_settlement_epoch_id
       OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
       OR receipt_row.output_root IS DISTINCT FROM NEW.first_snapshot_hash THEN
        RAISE EXCEPTION
            'stateful epoch cutover first snapshot receipt is invalid';
    END IF;

    predecessor_receipt := COALESCE(
        NEW.predecessor_receipt_hash,
        NEW.last_legacy_finalization_receipt_hash
    );
    SELECT role, purpose, epoch_id, receipt_status, output_root, receipt_doc
    INTO receipt_row
    FROM public.research_lab_attested_execution_receipts_v2
    WHERE receipt_hash = NEW.cutover_receipt_hash;
    IF NOT FOUND
       OR receipt_row.role IS DISTINCT FROM 'gateway_coordinator'
       OR receipt_row.purpose IS DISTINCT FROM
          'research_lab.subnet_epoch_cutover.v2'
       OR receipt_row.epoch_id IS DISTINCT FROM
          NEW.first_settlement_epoch_id
       OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
       OR receipt_row.output_root IS DISTINCT FROM NEW.cutover_authority_hash
       OR pg_catalog.jsonb_typeof(
              receipt_row.receipt_doc->'parent_receipt_hashes'
          ) IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_array_length(
              receipt_row.receipt_doc->'parent_receipt_hashes'
          ) IS DISTINCT FROM 2
       OR NOT (
           receipt_row.receipt_doc->'parent_receipt_hashes'
           @> pg_catalog.jsonb_build_array(
               NEW.first_snapshot_receipt_hash,
               predecessor_receipt
           )
       ) THEN
        RAISE EXCEPTION
            'stateful epoch cutover coordinator receipt is invalid';
    END IF;

    IF NEW.schema_version =
       'leadpoet.subnet_epoch_cutover_authority.v2' THEN
        SELECT
            migration.netuid,
            migration.epoch_id,
            migration.allocation_hash,
            migration.settlement_receipt_hash,
            migration.settlement_doc,
            receipt.output_root,
            receipt.role,
            receipt.purpose,
            receipt.receipt_status
        INTO predecessor_row
        FROM public.research_lab_legacy_finalized_allocation_migrations_v2
             AS migration
        JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
          ON receipt.receipt_hash = migration.settlement_receipt_hash
        WHERE migration.settlement_receipt_hash =
              NEW.predecessor_receipt_hash;
        IF NOT FOUND
           OR predecessor_row.netuid IS DISTINCT FROM NEW.netuid
           OR predecessor_row.epoch_id IS DISTINCT FROM
              NEW.predecessor_epoch_id
           OR predecessor_row.epoch_id > NEW.last_legacy_epoch_id
           OR predecessor_row.allocation_hash IS DISTINCT FROM
              NEW.predecessor_allocation_hash
           OR predecessor_row.output_root IS DISTINCT FROM
              NEW.predecessor_authority_hash
           OR predecessor_row.role IS DISTINCT FROM 'gateway_coordinator'
           OR predecessor_row.purpose IS DISTINCT FROM
              'research_lab.legacy_finalized_allocation.v2'
           OR predecessor_row.receipt_status IS DISTINCT FROM 'succeeded'
           OR (
               predecessor_row.settlement_doc->>'chain_target_block'
           )::BIGINT >= NEW.cutover_block THEN
            RAISE EXCEPTION
                'stateful epoch historical predecessor proof is invalid';
        END IF;
    ELSE
        SELECT
            bundle.netuid,
            bundle.epoch_id,
            finalization.bundle_hash,
            finalization.finalization_receipt_hash,
            finalization.finalized_block
        INTO finalization_row
        FROM public.research_lab_attested_weight_finalizations_v2 finalization
        JOIN public.research_lab_attested_weight_bundles_v2 bundle
          ON bundle.bundle_hash = finalization.bundle_hash
        WHERE finalization.weight_finalization_event_hash =
              NEW.last_legacy_weight_finalization_event_hash;
        IF NOT FOUND
           OR finalization_row.netuid IS DISTINCT FROM NEW.netuid
           OR finalization_row.epoch_id IS DISTINCT FROM
              NEW.last_legacy_epoch_id
           OR finalization_row.bundle_hash IS DISTINCT FROM
              NEW.last_legacy_bundle_hash
           OR finalization_row.finalization_receipt_hash IS DISTINCT FROM
              NEW.last_legacy_finalization_receipt_hash
           OR finalization_row.finalized_block > NEW.cutover_block THEN
            RAISE EXCEPTION
                'stateful epoch cutover finalization proof is invalid';
        END IF;
    END IF;

    SELECT * INTO candidate_row
    FROM public.research_lab_stateful_subnet_epoch_candidates_v1
    WHERE snapshot_hash = NEW.first_snapshot_hash;
    IF NOT FOUND
       OR candidate_row.mapping_hash IS DISTINCT FROM NEW.mapping_hash
       OR candidate_row.epoch_scheme IS DISTINCT FROM NEW.epoch_scheme
       OR candidate_row.network_genesis_hash IS DISTINCT FROM
          NEW.network_genesis_hash
       OR candidate_row.netuid IS DISTINCT FROM NEW.netuid
       OR candidate_row.current_block IS DISTINCT FROM NEW.cutover_block
       OR candidate_row.last_epoch_block IS DISTINCT FROM NEW.cutover_block
       OR candidate_row.block_hash IS DISTINCT FROM NEW.cutover_block_hash
       OR candidate_row.subnet_epoch_index IS DISTINCT FROM
          NEW.first_subnet_epoch_index
       OR candidate_row.epoch_ref IS DISTINCT FROM NEW.first_epoch_ref
       OR candidate_row.proposed_settlement_epoch_id IS DISTINCT FROM
          NEW.first_settlement_epoch_id
       OR candidate_row.tempo IS DISTINCT FROM NEW.first_tempo
       OR candidate_row.pending_epoch_at IS DISTINCT FROM
          NEW.first_pending_epoch_at
       OR candidate_row.blocks_since_last_step IS DISTINCT FROM
          NEW.first_blocks_since_last_step
       OR candidate_row.next_epoch_block IS DISTINCT FROM
          NEW.first_next_epoch_block
       OR candidate_row.observed_at IS DISTINCT FROM
          NEW.first_observed_at::TIMESTAMPTZ
       OR candidate_row.chain_state_receipt_hash IS DISTINCT FROM
          NEW.first_snapshot_receipt_hash
       OR candidate_row.snapshot_doc IS DISTINCT FROM
          NEW.first_snapshot_doc THEN
        RAISE EXCEPTION
            'stateful epoch cutover first candidate snapshot differs';
    END IF;

    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR SHARE;
    IF NOT FOUND
       OR state_row.lifecycle_state NOT IN (
           'cutover_fenced', 'stateful_staged', 'stateful_active'
       )
       OR state_row.mapping_hash IS DISTINCT FROM NEW.mapping_hash
       OR state_row.network_genesis_hash IS DISTINCT FROM
          NEW.network_genesis_hash
       OR state_row.netuid IS DISTINCT FROM NEW.netuid
       OR state_row.last_legacy_epoch_id IS DISTINCT FROM
          NEW.last_legacy_epoch_id
       OR state_row.first_settlement_epoch_id IS DISTINCT FROM
          NEW.first_settlement_epoch_id
       OR state_row.candidate_snapshot_hash IS DISTINCT FROM
          NEW.first_snapshot_hash
       OR state_row.candidate_receipt_hash IS DISTINCT FROM
          NEW.first_snapshot_receipt_hash
       OR state_row.last_legacy_finalization_receipt_hash IS DISTINCT FROM
          predecessor_receipt
       OR state_row.cutover_authority_hash IS DISTINCT FROM
          NEW.cutover_authority_hash
       OR state_row.cutover_receipt_hash IS DISTINCT FROM
          NEW.cutover_receipt_hash THEN
        RAISE EXCEPTION 'stateful epoch cutover differs from durable fence';
    END IF;
    RETURN NEW;
END;
$$;

NOTIFY pgrst, 'reload schema';

COMMIT;
