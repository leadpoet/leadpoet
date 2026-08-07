-- Bound provider persistence round trips without weakening exact readback.
--
-- Cache puts return the exact inserted/existing encrypted row from the same
-- transaction. Provider-outcome batches insert up to 32 already encrypted,
-- hash-chained checkpoint rows atomically and preserve every sequence.

BEGIN;

CREATE OR REPLACE FUNCTION public.put_research_lab_provider_evidence_cache_v2(
    cache_row JSONB
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
        'request_fingerprint',
        'cache_entry_hash',
        'cache_artifact_id',
        'source_record_hash',
        'source_boot_identity_hash',
        'response_body_hash',
        'encrypted_cache_doc'
    ];
    key_ref_hash TEXT;
    cache_day DATE;
    fingerprint TEXT;
    entry_hash TEXT;
    durable_row JSONB;
BEGIN
    IF pg_catalog.jsonb_typeof(cache_row) IS DISTINCT FROM 'object'
       OR NOT cache_row ?& required_fields
       OR (
           SELECT pg_catalog.count(*)
           FROM pg_catalog.jsonb_object_keys(cache_row)
       ) <> pg_catalog.cardinality(required_fields)
    THEN
        RAISE EXCEPTION 'provider evidence cache fields are invalid'
            USING ERRCODE = '22023';
    END IF;

    key_ref_hash := cache_row->>'artifact_master_key_ref_hash';
    cache_day := (cache_row->>'utc_day')::DATE;
    fingerprint := cache_row->>'request_fingerprint';
    entry_hash := cache_row->>'cache_entry_hash';
    IF key_ref_hash !~ '^sha256:[0-9a-f]{64}$'
       OR fingerprint !~ '^[0-9a-f]{64}$'
       OR entry_hash !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'provider evidence cache identity is invalid'
            USING ERRCODE = '22023';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_provider_evidence_cache_v2'),
        pg_catalog.hashtext(
            key_ref_hash || ':' || cache_day::TEXT || ':' || fingerprint
        )
    );

    SELECT pg_catalog.to_jsonb(c) - 'created_at'
      INTO durable_row
      FROM public.research_lab_provider_evidence_cache_v2 AS c
     WHERE c.artifact_master_key_ref_hash = key_ref_hash
       AND c.utc_day = cache_day
       AND c.request_fingerprint = fingerprint;

    IF durable_row IS NOT NULL THEN
        IF durable_row IS DISTINCT FROM cache_row THEN
            RAISE EXCEPTION 'provider evidence cache identity identifies another row'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'status', 'existing',
            'cache_entry_hash', entry_hash,
            'cache_row', durable_row
        );
    END IF;

    INSERT INTO public.research_lab_provider_evidence_cache_v2 (
        schema_version,
        artifact_master_key_ref_hash,
        utc_day,
        request_fingerprint,
        cache_entry_hash,
        cache_artifact_id,
        source_record_hash,
        source_boot_identity_hash,
        response_body_hash,
        encrypted_cache_doc
    )
    VALUES (
        cache_row->>'schema_version',
        key_ref_hash,
        cache_day,
        fingerprint,
        entry_hash,
        cache_row->>'cache_artifact_id',
        cache_row->>'source_record_hash',
        cache_row->>'source_boot_identity_hash',
        cache_row->>'response_body_hash',
        cache_row->'encrypted_cache_doc'
    );

    SELECT pg_catalog.to_jsonb(c) - 'created_at'
      INTO durable_row
      FROM public.research_lab_provider_evidence_cache_v2 AS c
     WHERE c.artifact_master_key_ref_hash = key_ref_hash
       AND c.utc_day = cache_day
       AND c.request_fingerprint = fingerprint;
    IF durable_row IS DISTINCT FROM cache_row THEN
        RAISE EXCEPTION 'provider evidence cache durable insert differs'
            USING ERRCODE = '23514';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'status', 'inserted',
        'cache_entry_hash', entry_hash,
        'cache_row', durable_row
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.append_research_lab_provider_outcome_checkpoints_v2(
    checkpoint_rows JSONB
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
    row_count INTEGER;
    row_index INTEGER := 0;
    item JSONB;
    first_row JSONB;
    final_row JSONB;
    key_ref_hash TEXT;
    checkpoint_day DATE;
    first_sequence BIGINT;
    first_previous_hash TEXT;
    final_checkpoint_hash TEXT;
    prior_sequence BIGINT;
    prior_checkpoint_hash TEXT;
    current_row JSONB;
    current_sequence BIGINT;
    current_checkpoint_hash TEXT;
    existing_row JSONB;
    existing_count INTEGER := 0;
    durable_row JSONB;
BEGIN
    IF pg_catalog.jsonb_typeof(checkpoint_rows) IS DISTINCT FROM 'array' THEN
        RAISE EXCEPTION 'provider outcome checkpoint batch is not an array'
            USING ERRCODE = '22023';
    END IF;
    row_count := pg_catalog.jsonb_array_length(checkpoint_rows);
    IF row_count < 1 OR row_count > 32 THEN
        RAISE EXCEPTION 'provider outcome checkpoint batch size is invalid'
            USING ERRCODE = '22023';
    END IF;

    FOR item IN
        SELECT value FROM pg_catalog.jsonb_array_elements(checkpoint_rows)
    LOOP
        row_index := row_index + 1;
        IF pg_catalog.jsonb_typeof(item) IS DISTINCT FROM 'object'
           OR NOT item ?& required_fields
           OR (
               SELECT pg_catalog.count(*)
               FROM pg_catalog.jsonb_object_keys(item)
           ) <> pg_catalog.cardinality(required_fields)
        THEN
            RAISE EXCEPTION 'provider outcome checkpoint batch row fields are invalid'
                USING ERRCODE = '22023';
        END IF;
        IF row_index = 1 THEN
            first_row := item;
            key_ref_hash := item->>'artifact_master_key_ref_hash';
            checkpoint_day := (item->>'utc_day')::DATE;
            first_sequence := (item->>'sequence')::BIGINT;
            first_previous_hash := COALESCE(
                item->>'previous_checkpoint_hash',
                ''
            );
        ELSIF item->>'artifact_master_key_ref_hash' <> key_ref_hash
           OR (item->>'utc_day')::DATE <> checkpoint_day
           OR (item->>'sequence')::BIGINT <> prior_sequence + 1
           OR COALESCE(item->>'previous_checkpoint_hash', '')
              <> prior_checkpoint_hash
        THEN
            RAISE EXCEPTION 'provider outcome checkpoint batch lineage is invalid'
                USING ERRCODE = '22023';
        END IF;
        IF item->>'artifact_master_key_ref_hash' !~ '^sha256:[0-9a-f]{64}$'
           OR item->>'checkpoint_hash' !~ '^sha256:[0-9a-f]{64}$'
           OR (item->>'sequence')::BIGINT <= 0
        THEN
            RAISE EXCEPTION 'provider outcome checkpoint batch identity is invalid'
                USING ERRCODE = '22023';
        END IF;
        prior_sequence := (item->>'sequence')::BIGINT;
        prior_checkpoint_hash := item->>'checkpoint_hash';
        final_row := item;
    END LOOP;
    final_checkpoint_hash := final_row->>'checkpoint_hash';

    IF NOT pg_catalog.pg_try_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_provider_outcome_checkpoint_v2'),
        pg_catalog.hashtext(key_ref_hash || ':' || checkpoint_day::TEXT)
    ) THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'busy',
            'checkpoint_hash', final_checkpoint_hash,
            'checkpoint_count', row_count
        );
    END IF;

    FOR item IN
        SELECT value FROM pg_catalog.jsonb_array_elements(checkpoint_rows)
    LOOP
        SELECT pg_catalog.to_jsonb(c) - 'created_at'
          INTO existing_row
          FROM public.research_lab_provider_outcome_checkpoints_v2 AS c
         WHERE c.checkpoint_hash = item->>'checkpoint_hash';
        IF existing_row IS NOT NULL THEN
            IF existing_row IS DISTINCT FROM item THEN
                RAISE EXCEPTION 'provider outcome checkpoint hash identifies another row'
                    USING ERRCODE = '23505';
            END IF;
            existing_count := existing_count + 1;
        END IF;
    END LOOP;
    IF existing_count = row_count THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'existing',
            'checkpoint_hash', final_checkpoint_hash,
            'checkpoint_count', row_count
        );
    ELSIF existing_count > 0 THEN
        RAISE EXCEPTION 'provider outcome checkpoint batch is partially durable'
            USING ERRCODE = '23514';
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
        IF first_sequence <> 1 OR first_previous_hash <> '' THEN
            RETURN pg_catalog.jsonb_build_object(
                'status', 'conflict',
                'checkpoint_hash', final_checkpoint_hash,
                'checkpoint_count', row_count,
                'head_checkpoint_row', NULL
            );
        END IF;
    ELSIF first_sequence <> current_sequence + 1
       OR first_previous_hash <> current_checkpoint_hash
    THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'conflict',
            'checkpoint_hash', final_checkpoint_hash,
            'checkpoint_count', row_count,
            'head_checkpoint_row', current_row
        );
    END IF;

    FOR item IN
        SELECT value FROM pg_catalog.jsonb_array_elements(checkpoint_rows)
    LOOP
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
            item->>'schema_version',
            item->>'artifact_master_key_ref_hash',
            (item->>'utc_day')::DATE,
            (item->>'sequence')::BIGINT,
            item->>'checkpoint_hash',
            COALESCE(item->>'previous_checkpoint_hash', ''),
            item->>'state_document_hash',
            item->>'checkpoint_artifact_id',
            item->'encrypted_checkpoint_doc'
        );
    END LOOP;

    FOR item IN
        SELECT value FROM pg_catalog.jsonb_array_elements(checkpoint_rows)
    LOOP
        SELECT pg_catalog.to_jsonb(c) - 'created_at'
          INTO durable_row
          FROM public.research_lab_provider_outcome_checkpoints_v2 AS c
         WHERE c.checkpoint_hash = item->>'checkpoint_hash';
        IF durable_row IS DISTINCT FROM item THEN
            RAISE EXCEPTION 'provider outcome checkpoint batch durable insert differs'
                USING ERRCODE = '23514';
        END IF;
    END LOOP;
    RETURN pg_catalog.jsonb_build_object(
        'status', 'inserted',
        'checkpoint_hash', final_checkpoint_hash,
        'checkpoint_count', row_count
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_provider_persistence_batch_contract_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
PARALLEL SAFE
SET search_path = ''
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.provider_persistence_batch_contract.v1',
        'cache_put', 'atomic_exact_row',
        'outcome_append', 'atomic_contiguous_batch',
        'outcome_batch_max', 32,
        'conflict_head_checkpoint_row', 'encrypted_or_null'
    );
$$;

REVOKE ALL ON FUNCTION public.put_research_lab_provider_evidence_cache_v2(JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.put_research_lab_provider_evidence_cache_v2(JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.append_research_lab_provider_outcome_checkpoints_v2(JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.append_research_lab_provider_outcome_checkpoints_v2(JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_provider_persistence_batch_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_provider_persistence_batch_contract_v1()
    TO service_role;

COMMENT ON FUNCTION public.put_research_lab_provider_evidence_cache_v2(JSONB) IS
    'Atomically inserts or returns one exact encrypted provider-evidence cache row.';
COMMENT ON FUNCTION public.append_research_lab_provider_outcome_checkpoints_v2(JSONB) IS
    'Atomically extends one encrypted provider-outcome lineage with up to 32 exact contiguous rows.';
COMMENT ON FUNCTION public.research_lab_provider_persistence_batch_contract_v1() IS
    'Declares exact atomic cache put and bounded outcome checkpoint batch semantics.';

NOTIFY pgrst, 'reload schema';

COMMIT;
