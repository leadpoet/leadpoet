-- V2-native stateful epoch bootstrap from an independently attested historical
-- finalization. Apply after migration 104.
--
-- The settlement ordinal still starts strictly after the measured legacy
-- namespace high-water. The predecessor epoch is separate: it is the newest
-- historical allocation whose signed validator vector, finalized chain state,
-- signed audit event, and immutable checkpoint were verified by the coordinator
-- enclave and persisted under its execution receipt.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ADD COLUMN IF NOT EXISTS predecessor_kind TEXT,
    ADD COLUMN IF NOT EXISTS predecessor_epoch_id INTEGER,
    ADD COLUMN IF NOT EXISTS predecessor_allocation_hash TEXT,
    ADD COLUMN IF NOT EXISTS predecessor_authority_hash TEXT,
    ADD COLUMN IF NOT EXISTS predecessor_receipt_hash TEXT
        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
        ON DELETE RESTRICT;

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ALTER COLUMN last_legacy_bundle_hash DROP NOT NULL,
    ALTER COLUMN last_legacy_weight_finalization_event_hash DROP NOT NULL,
    ALTER COLUMN last_legacy_finalization_receipt_hash DROP NOT NULL;

DO $constraints$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_stateful_subnet_epoch_cutovers_v1'::regclass
          AND contype = 'c'
          AND (
              pg_catalog.pg_get_constraintdef(oid) LIKE '%authority_doc%'
              OR (
                  pg_catalog.pg_get_constraintdef(oid) LIKE '%schema_version%'
                  AND pg_catalog.pg_get_constraintdef(oid) LIKE
                      '%leadpoet.subnet_epoch_cutover_authority.v1%'
              )
          )
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 '
            'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$constraints$;

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_schema_v2_check
        CHECK (
            schema_version IN (
                'leadpoet.subnet_epoch_cutover_authority.v1',
                'leadpoet.subnet_epoch_cutover_authority.v2'
            )
        ),
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_predecessor_v2_check
        CHECK (
            (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v1'
                AND last_legacy_bundle_hash IS NOT NULL
                AND last_legacy_weight_finalization_event_hash IS NOT NULL
                AND last_legacy_finalization_receipt_hash IS NOT NULL
                AND predecessor_kind IS NULL
                AND predecessor_epoch_id IS NULL
                AND predecessor_allocation_hash IS NULL
                AND predecessor_authority_hash IS NULL
                AND predecessor_receipt_hash IS NULL
            )
            OR
            (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v2'
                AND last_legacy_bundle_hash IS NULL
                AND last_legacy_weight_finalization_event_hash IS NULL
                AND last_legacy_finalization_receipt_hash IS NULL
                AND predecessor_kind =
                    'legacy_finalized_chain_migration_v2'
                AND predecessor_epoch_id BETWEEN 0 AND last_legacy_epoch_id
                AND predecessor_allocation_hash ~ '^sha256:[0-9a-f]{64}$'
                AND predecessor_authority_hash ~ '^sha256:[0-9a-f]{64}$'
                AND predecessor_receipt_hash ~ '^sha256:[0-9a-f]{64}$'
            )
        ),
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_authority_v2_check
        CHECK (
            (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v1'
                AND authority_doc ?& ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'last_legacy_bundle_hash',
                    'last_legacy_weight_finalization_event_hash',
                    'last_legacy_finalization_receipt_hash', 'manifest'
                ]
                AND authority_doc - ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'last_legacy_bundle_hash',
                    'last_legacy_weight_finalization_event_hash',
                    'last_legacy_finalization_receipt_hash', 'manifest'
                ] = '{}'::JSONB
                AND authority_doc->>'last_legacy_bundle_hash' =
                    last_legacy_bundle_hash
                AND authority_doc->>'last_legacy_weight_finalization_event_hash'
                    = last_legacy_weight_finalization_event_hash
                AND authority_doc->>'last_legacy_finalization_receipt_hash' =
                    last_legacy_finalization_receipt_hash
            )
            OR
            (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v2'
                AND authority_doc ?& ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'predecessor_kind', 'predecessor_epoch_id',
                    'predecessor_allocation_hash',
                    'predecessor_authority_hash',
                    'predecessor_receipt_hash', 'manifest'
                ]
                AND authority_doc - ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'predecessor_kind', 'predecessor_epoch_id',
                    'predecessor_allocation_hash',
                    'predecessor_authority_hash',
                    'predecessor_receipt_hash', 'manifest'
                ] = '{}'::JSONB
                AND authority_doc->>'predecessor_kind' = predecessor_kind
                AND (authority_doc->>'predecessor_epoch_id')::INTEGER =
                    predecessor_epoch_id
                AND authority_doc->>'predecessor_allocation_hash' =
                    predecessor_allocation_hash
                AND authority_doc->>'predecessor_authority_hash' =
                    predecessor_authority_hash
                AND authority_doc->>'predecessor_receipt_hash' =
                    predecessor_receipt_hash
            )
        ),
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_authority_common_v2_check
        CHECK (
            pg_catalog.jsonb_typeof(authority_doc) = 'object'
            AND authority_doc::TEXT !~*
                '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
            AND authority_doc->>'schema_version' = schema_version
            AND authority_doc->>'mapping_hash' = mapping_hash
            AND authority_doc->>'first_epoch_ref' = first_epoch_ref
            AND authority_doc->>'first_snapshot_hash' = first_snapshot_hash
            AND authority_doc->>'first_snapshot_receipt_hash' =
                first_snapshot_receipt_hash
            AND authority_doc->'manifest' = manifest_doc
        );

CREATE UNIQUE INDEX IF NOT EXISTS
    uq_research_lab_stateful_epoch_cutover_predecessor_v2
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1(
        predecessor_receipt_hash
    )
    WHERE predecessor_receipt_hash IS NOT NULL;

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
       OR candidate_row.observed_at IS DISTINCT FROM NEW.first_observed_at
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

DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_cutover_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE TRIGGER validate_research_lab_stateful_epoch_cutover_v1
    BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW EXECUTE FUNCTION
    public.validate_research_lab_stateful_epoch_cutover_v2();

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
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR UPDATE;
    IF NOT FOUND OR state_row.lifecycle_state <> 'cutover_fenced' THEN
        RAISE EXCEPTION
            'stateful epoch V2 binding requires the pre-boundary fence';
    END IF;

    SELECT * INTO candidate_row
    FROM public.research_lab_stateful_subnet_epoch_candidates_v1
    WHERE mapping_hash = p_mapping_hash;
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
        SELECT role, purpose, epoch_id, receipt_status, output_root, receipt_doc
        INTO coordinator_receipt
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = p_cutover_receipt_hash;
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
        SET mapping_hash = candidate_row.mapping_hash,
            candidate_snapshot_hash = candidate_row.snapshot_hash,
            candidate_receipt_hash = candidate_row.chain_state_receipt_hash,
            last_legacy_finalization_receipt_hash =
                p_last_legacy_finalization_receipt_hash,
            cutover_authority_hash = p_cutover_authority_hash,
            updated_at = pg_catalog.clock_timestamp()
        WHERE singleton
        RETURNING * INTO state_row;
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

CREATE OR REPLACE FUNCTION
public.research_lab_stateful_subnet_epoch_stage_v2(
    p_cutover_row JSONB,
    p_initialization_event JSONB
)
RETURNS TABLE (
    lifecycle_state TEXT,
    mapping_hash TEXT,
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
    cutover_input public.research_lab_stateful_subnet_epoch_cutovers_v1%ROWTYPE;
    stored_cutover public.research_lab_stateful_subnet_epoch_cutovers_v1%ROWTYPE;
    initialization_row RECORD;
    initialization_payload JSONB;
    initialization_nonce_value UUID;
    initialization_ts TIMESTAMPTZ;
BEGIN
    SELECT * INTO cutover_input
    FROM pg_catalog.jsonb_populate_record(
        NULL::public.research_lab_stateful_subnet_epoch_cutovers_v1,
        p_cutover_row
    );
    IF cutover_input.schema_version IS DISTINCT FROM
       'leadpoet.subnet_epoch_cutover_authority.v2'
       OR pg_catalog.to_jsonb(cutover_input) - 'created_at'
          IS DISTINCT FROM p_cutover_row THEN
        RAISE EXCEPTION 'stateful epoch V2 cutover row shape is invalid';
    END IF;
    IF pg_catalog.jsonb_typeof(p_initialization_event) IS DISTINCT FROM 'object'
       OR NOT (p_initialization_event ?& ARRAY[
           'event_type', 'actor_hotkey', 'nonce', 'ts', 'payload_hash',
           'build_id', 'signature', 'payload'
       ])
       OR p_initialization_event - ARRAY[
           'event_type', 'actor_hotkey', 'nonce', 'ts', 'payload_hash',
           'build_id', 'signature', 'payload'
       ] <> '{}'::JSONB THEN
        RAISE EXCEPTION
            'stateful epoch V2 initialization event shape is invalid';
    END IF;
    initialization_payload := p_initialization_event->'payload';
    initialization_nonce_value := (p_initialization_event->>'nonce')::UUID;
    initialization_ts := (p_initialization_event->>'ts')::TIMESTAMPTZ;

    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR UPDATE;
    IF NOT FOUND OR state_row.lifecycle_state <> 'cutover_fenced'
       OR cutover_input.mapping_hash IS DISTINCT FROM state_row.mapping_hash
       OR cutover_input.cutover_authority_hash IS DISTINCT FROM
          state_row.cutover_authority_hash
       OR cutover_input.first_settlement_epoch_id IS DISTINCT FROM
          state_row.first_settlement_epoch_id
       OR cutover_input.last_legacy_epoch_id IS DISTINCT FROM
          state_row.last_legacy_epoch_id
       OR cutover_input.first_snapshot_hash IS DISTINCT FROM
          state_row.candidate_snapshot_hash
       OR cutover_input.first_snapshot_receipt_hash IS DISTINCT FROM
          state_row.candidate_receipt_hash
       OR cutover_input.predecessor_receipt_hash IS DISTINCT FROM
          state_row.last_legacy_finalization_receipt_hash THEN
        RAISE EXCEPTION
            'stateful epoch V2 staged cutover differs from fenced plan';
    END IF;
    IF p_initialization_event->>'event_type' <> 'EPOCH_INITIALIZATION'
       OR p_initialization_event->>'actor_hotkey' <> 'system'
       OR p_initialization_event->>'signature' <> 'system'
       OR p_initialization_event->>'payload_hash' !~ '^[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(initialization_payload) IS DISTINCT FROM
          'object'
       OR (initialization_payload->>'epoch_id')::BIGINT IS DISTINCT FROM
          state_row.first_settlement_epoch_id::BIGINT
       OR initialization_payload->>'epoch_key_semantics' <>
          'settlement_ordinal'
       OR initialization_payload->'epoch_authority' IS DISTINCT FROM
          cutover_input.first_snapshot_doc
       OR (initialization_payload->'epoch_boundaries'->>'start_block')::BIGINT
          IS DISTINCT FROM cutover_input.cutover_block
       OR (initialization_payload->'epoch_boundaries'->>'end_block')::BIGINT
          IS DISTINCT FROM cutover_input.first_next_epoch_block
       OR (
           initialization_payload->'epoch_boundaries'
           ->>'expected_end_block'
       )::BIGINT IS DISTINCT FROM cutover_input.first_next_epoch_block
       OR (
           initialization_payload->'epoch_boundaries'
           ->>'pending_epoch_at'
       )::BIGINT IS DISTINCT FROM cutover_input.first_pending_epoch_at
       OR (initialization_payload->'epoch_boundaries'->>'tempo')::INTEGER
          IS DISTINCT FROM cutover_input.first_tempo THEN
        RAISE EXCEPTION
            'stateful epoch V2 initialization differs from authority';
    END IF;

    UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    SET cutover_receipt_hash = cutover_input.cutover_receipt_hash,
        initialization_nonce = initialization_nonce_value,
        initialization_payload_hash = p_initialization_event->>'payload_hash',
        updated_at = pg_catalog.clock_timestamp()
    WHERE singleton
    RETURNING * INTO state_row;

    cutover_input.created_at := pg_catalog.clock_timestamp();
    INSERT INTO public.research_lab_stateful_subnet_epoch_cutovers_v1
    SELECT cutover_input.*
    ON CONFLICT DO NOTHING;

    SELECT cutover.* INTO stored_cutover
    FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 AS cutover
    WHERE cutover.mapping_hash = state_row.mapping_hash;
    IF NOT FOUND
       OR pg_catalog.to_jsonb(stored_cutover) - 'created_at'
          IS DISTINCT FROM p_cutover_row THEN
        RAISE EXCEPTION
            'stateful epoch V2 staged cutover exact readback failed';
    END IF;

    INSERT INTO public.transparency_log (
        event_type, actor_hotkey, nonce, ts, payload_hash, build_id,
        signature, payload
    ) VALUES (
        p_initialization_event->>'event_type',
        p_initialization_event->>'actor_hotkey',
        initialization_nonce_value,
        initialization_ts,
        p_initialization_event->>'payload_hash',
        p_initialization_event->>'build_id',
        p_initialization_event->>'signature',
        initialization_payload
    )
    ON CONFLICT DO NOTHING;

    SELECT event_type, actor_hotkey, nonce, ts, payload_hash, build_id,
           signature, payload
    INTO initialization_row
    FROM public.transparency_log
    WHERE nonce = initialization_nonce_value;
    IF NOT FOUND
       OR initialization_row.event_type IS DISTINCT FROM
          p_initialization_event->>'event_type'
       OR initialization_row.actor_hotkey IS DISTINCT FROM
          p_initialization_event->>'actor_hotkey'
       OR initialization_row.ts IS DISTINCT FROM initialization_ts
       OR initialization_row.payload_hash IS DISTINCT FROM
          p_initialization_event->>'payload_hash'
       OR initialization_row.build_id IS DISTINCT FROM
          p_initialization_event->>'build_id'
       OR initialization_row.signature IS DISTINCT FROM
          p_initialization_event->>'signature'
       OR initialization_row.payload IS DISTINCT FROM initialization_payload THEN
        RAISE EXCEPTION
            'stateful epoch V2 initialization exact readback failed';
    END IF;

    UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    SET lifecycle_state = 'stateful_staged',
        staged_at = COALESCE(staged_at, pg_catalog.clock_timestamp()),
        updated_at = pg_catalog.clock_timestamp()
    WHERE singleton
    RETURNING * INTO state_row;

    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.mapping_hash,
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
REVOKE ALL ON FUNCTION
    public.research_lab_stateful_subnet_epoch_stage_v2(JSONB, JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_stateful_subnet_epoch_stage_v2(JSONB, JSONB)
    TO service_role;

REVOKE ALL ON FUNCTION
    public.validate_research_lab_stateful_epoch_cutover_v2()
    FROM PUBLIC, anon, authenticated;

NOTIFY pgrst, 'reload schema';

COMMIT;
