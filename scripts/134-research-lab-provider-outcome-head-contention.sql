-- Return the authenticated durable head with stale provider-outcome appends.
--
-- Migration 133 made expected lock and lineage contention non-exceptional.
-- Preserve those response fields so rolling callers remain compatible, and
-- add the encrypted durable head to conflict responses so a measured caller
-- can rebase without issuing a separate checkpoint read.

BEGIN;

CREATE OR REPLACE FUNCTION public.append_research_lab_provider_outcome_checkpoint_v2(
    checkpoint_row JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    required_fields CONSTANT TEXT[] := ARRAY[
        'schema_version',
        'artifact_master_key_ref_hash',
        'utc_day',
        'sequence',
        'checkpoint_hash',
        'previous_checkpoint_hash',
        'state_document_hash',
        'checkpoint_artifact_id',
        'encrypted_checkpoint_doc'
    ];
    key_ref_hash TEXT;
    checkpoint_day DATE;
    checkpoint_sequence BIGINT;
    incoming_checkpoint_hash TEXT;
    incoming_previous_hash TEXT;
    current_row JSONB;
    current_sequence BIGINT;
    current_checkpoint_hash TEXT;
    existing_row JSONB;
    inserted_row JSONB;
BEGIN
    IF pg_catalog.jsonb_typeof(checkpoint_row) IS DISTINCT FROM 'object'
       OR NOT checkpoint_row ?& required_fields
       OR (
           SELECT pg_catalog.count(*)
           FROM pg_catalog.jsonb_object_keys(checkpoint_row)
       ) <> pg_catalog.cardinality(required_fields)
    THEN
        RAISE EXCEPTION 'provider outcome checkpoint fields are invalid'
            USING ERRCODE = '22023';
    END IF;

    key_ref_hash := checkpoint_row->>'artifact_master_key_ref_hash';
    checkpoint_day := (checkpoint_row->>'utc_day')::DATE;
    checkpoint_sequence := (checkpoint_row->>'sequence')::BIGINT;
    incoming_checkpoint_hash := checkpoint_row->>'checkpoint_hash';
    incoming_previous_hash := COALESCE(
        checkpoint_row->>'previous_checkpoint_hash',
        ''
    );

    IF key_ref_hash !~ '^sha256:[0-9a-f]{64}$'
       OR incoming_checkpoint_hash !~ '^sha256:[0-9a-f]{64}$'
       OR checkpoint_sequence <= 0
    THEN
        RAISE EXCEPTION 'provider outcome checkpoint identity is invalid'
            USING ERRCODE = '22023';
    END IF;

    IF NOT pg_catalog.pg_try_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_provider_outcome_checkpoint_v2'),
        pg_catalog.hashtext(key_ref_hash || ':' || checkpoint_day::TEXT)
    ) THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'busy',
            'checkpoint_hash', incoming_checkpoint_hash
        );
    END IF;

    SELECT pg_catalog.to_jsonb(c) - 'created_at'
      INTO existing_row
      FROM public.research_lab_provider_outcome_checkpoints_v2 AS c
     WHERE c.checkpoint_hash = incoming_checkpoint_hash;

    IF existing_row IS NOT NULL THEN
        IF existing_row IS DISTINCT FROM checkpoint_row THEN
            RAISE EXCEPTION 'provider outcome checkpoint hash already identifies another row'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'status', 'existing',
            'checkpoint_hash', incoming_checkpoint_hash
        );
    END IF;

    SELECT
        pg_catalog.to_jsonb(c) - 'created_at',
        c.sequence,
        c.checkpoint_hash
      INTO current_row, current_sequence, current_checkpoint_hash
      FROM public.research_lab_provider_outcome_checkpoints_v2 AS c
     WHERE c.artifact_master_key_ref_hash = key_ref_hash
       AND c.utc_day = checkpoint_day
     ORDER BY c.sequence DESC
     LIMIT 1;

    IF current_sequence IS NULL THEN
        IF checkpoint_sequence <> 1 OR incoming_previous_hash <> '' THEN
            RETURN pg_catalog.jsonb_build_object(
                'status', 'conflict',
                'checkpoint_hash', incoming_checkpoint_hash,
                'head_checkpoint_row', NULL
            );
        END IF;
    ELSIF checkpoint_sequence <> current_sequence + 1
       OR incoming_previous_hash <> current_checkpoint_hash
    THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'conflict',
            'checkpoint_hash', incoming_checkpoint_hash,
            'head_checkpoint_row', current_row
        );
    END IF;

    INSERT INTO public.research_lab_provider_outcome_checkpoints_v2 (
        schema_version,
        artifact_master_key_ref_hash,
        utc_day,
        sequence,
        checkpoint_hash,
        previous_checkpoint_hash,
        state_document_hash,
        checkpoint_artifact_id,
        encrypted_checkpoint_doc
    )
    VALUES (
        checkpoint_row->>'schema_version',
        key_ref_hash,
        checkpoint_day,
        checkpoint_sequence,
        incoming_checkpoint_hash,
        incoming_previous_hash,
        checkpoint_row->>'state_document_hash',
        checkpoint_row->>'checkpoint_artifact_id',
        checkpoint_row->'encrypted_checkpoint_doc'
    );

    SELECT pg_catalog.to_jsonb(c) - 'created_at'
      INTO inserted_row
      FROM public.research_lab_provider_outcome_checkpoints_v2 AS c
     WHERE c.checkpoint_hash = incoming_checkpoint_hash;

    IF inserted_row IS DISTINCT FROM checkpoint_row THEN
        RAISE EXCEPTION 'provider outcome checkpoint durable insert differs'
            USING ERRCODE = '23514';
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'status', 'inserted',
        'checkpoint_hash', incoming_checkpoint_hash
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_provider_outcome_contention_contract_v3()
RETURNS JSONB
LANGUAGE sql
STABLE
PARALLEL SAFE
SET search_path = ''
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.provider_outcome_contention_contract.v3',
        'lock_contention_status', 'busy',
        'stale_lineage_status', 'conflict',
        'candidate_checkpoint_hash', TRUE,
        'conflict_head_checkpoint_row', 'encrypted_or_null'
    );
$$;

REVOKE ALL ON FUNCTION public.append_research_lab_provider_outcome_checkpoint_v2(JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.append_research_lab_provider_outcome_checkpoint_v2(JSONB)
    TO service_role;

REVOKE ALL ON FUNCTION public.research_lab_provider_outcome_contention_contract_v3()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_provider_outcome_contention_contract_v3()
    TO service_role;

COMMENT ON FUNCTION public.append_research_lab_provider_outcome_checkpoint_v2(JSONB) IS
    'Atomically extends one encrypted provider-outcome lineage and returns measured busy/conflict state with candidate identity and the authenticated durable head.';
COMMENT ON FUNCTION public.research_lab_provider_outcome_contention_contract_v3() IS
    'Declares the candidate-bound busy/conflict response and encrypted durable-head contract introduced by migration 134.';

NOTIFY pgrst, 'reload schema';

COMMIT;
