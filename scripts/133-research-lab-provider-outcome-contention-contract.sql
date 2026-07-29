-- Make expected provider-outcome contention a non-exceptional RPC contract.
--
-- Migration 131 made the lineage lock non-blocking, but represented both a
-- held lock and a stale lineage head as SQLSTATE 40001. Under rolling restart
-- overlap, every rejected append therefore rolled back a transaction and the
-- gateway performed a separate read to discover the durable head. Return
-- authenticated JSON outcomes instead: busy callers back off without reading,
-- while stale callers receive the encrypted durable head needed to rebase.

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
    IF jsonb_typeof(checkpoint_row) IS DISTINCT FROM 'object'
       OR NOT checkpoint_row ?& required_fields
       OR (
           SELECT pg_catalog.count(*)
           FROM pg_catalog.jsonb_object_keys(checkpoint_row)
       ) <> cardinality(required_fields)
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
        RETURN pg_catalog.jsonb_build_object('status', 'busy');
    END IF;

    SELECT pg_catalog.to_jsonb(c) - 'created_at'
      INTO existing_row
      FROM public.research_lab_provider_outcome_checkpoints_v2 c
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
      FROM public.research_lab_provider_outcome_checkpoints_v2 c
     WHERE c.artifact_master_key_ref_hash = key_ref_hash
       AND c.utc_day = checkpoint_day
     ORDER BY c.sequence DESC
     LIMIT 1;

    IF current_sequence IS NULL THEN
        IF checkpoint_sequence <> 1 OR incoming_previous_hash <> '' THEN
            RETURN pg_catalog.jsonb_build_object(
                'status', 'conflict',
                'head_checkpoint_row', NULL
            );
        END IF;
    ELSIF checkpoint_sequence <> current_sequence + 1
       OR incoming_previous_hash <> current_checkpoint_hash
    THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'conflict',
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
      FROM public.research_lab_provider_outcome_checkpoints_v2 c
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

REVOKE ALL ON FUNCTION public.append_research_lab_provider_outcome_checkpoint_v2(JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.append_research_lab_provider_outcome_checkpoint_v2(JSONB)
    TO service_role;

COMMENT ON FUNCTION public.append_research_lab_provider_outcome_checkpoint_v2(JSONB) IS
    'Atomically extends one encrypted provider-outcome lineage and returns non-exceptional busy/conflict backpressure with the durable head.';

NOTIFY pgrst, 'reload schema';

COMMIT;
