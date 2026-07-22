-- Research Lab egress reduction: conflict-safe batched provider-usage ledger insert.
--
-- Today every projected provider-usage row costs one existence SELECT (in the
-- project path) or two (in the backfill path) plus one INSERT. The rows are
-- deterministic (PK usage_row_id is a uuid5 over run/event/provider/endpoint),
-- so the existence checks add no correctness — only ~13.28M weekly PostgREST
-- requests. This function replaces the per-row read+insert with a single
-- set-based INSERT ... ON CONFLICT (usage_row_id) DO NOTHING.
--
-- Behavior preserved: the table is append-only (BEFORE UPDATE OR DELETE
-- trigger). ON CONFLICT DO NOTHING performs only INSERTs, so the append-only
-- guard is not triggered, and previously-written rows are left byte-identical.
-- The projector's in-memory dedup and deterministic PKs make DO NOTHING exactly
-- equivalent to the old skip-if-exists behavior.

BEGIN;

CREATE OR REPLACE FUNCTION public.insert_research_lab_provider_usage_ledger_rows(
    rows JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    inserted_ids UUID[];
    requested_count INTEGER;
BEGIN
    -- Fail closed on a malformed batch rather than silently dropping rows.
    IF rows IS NULL OR pg_catalog.jsonb_typeof(rows) <> 'array' THEN
        RAISE EXCEPTION 'provider usage ledger batch must be a JSON array'
            USING ERRCODE = '22023';
    END IF;

    requested_count := pg_catalog.jsonb_array_length(rows);
    IF requested_count = 0 THEN
        RETURN pg_catalog.jsonb_build_object(
            'requested', 0, 'inserted', 0, 'inserted_ids', '[]'::JSONB
        );
    END IF;
    -- Bounded batch: the projector writes at most a few hundred rows per run.
    IF requested_count > 5000 THEN
        RAISE EXCEPTION 'provider usage ledger batch exceeds 5000 rows'
            USING ERRCODE = '22023';
    END IF;

    WITH incoming AS (
        SELECT * FROM pg_catalog.jsonb_to_recordset(rows) AS r(
            usage_row_id        UUID,
            schema_version      TEXT,
            utc_day             TEXT,
            recorded_at         TIMESTAMPTZ,
            provider_id         TEXT,
            endpoint_class      TEXT,
            request_fingerprint TEXT,
            evidence            TEXT,
            status              INTEGER,
            est_cost_microusd   BIGINT,
            caller_doc          JSONB
        )
    ),
    ins AS (
        INSERT INTO public.research_lab_provider_usage_ledger AS l (
            usage_row_id, schema_version, utc_day, recorded_at, provider_id,
            endpoint_class, request_fingerprint, evidence, status,
            est_cost_microusd, caller_doc
        )
        SELECT
            i.usage_row_id,
            COALESCE(i.schema_version, '1.0'),
            i.utc_day,
            COALESCE(i.recorded_at, pg_catalog.now()),
            i.provider_id,
            COALESCE(i.endpoint_class, ''),
            COALESCE(i.request_fingerprint, ''),
            i.evidence,
            COALESCE(i.status, 0),
            COALESCE(i.est_cost_microusd, 0),
            COALESCE(i.caller_doc, '{}'::JSONB)
        FROM incoming i
        WHERE i.usage_row_id IS NOT NULL
        ON CONFLICT (usage_row_id) DO NOTHING
        RETURNING l.usage_row_id
    )
    SELECT pg_catalog.array_agg(usage_row_id) INTO inserted_ids FROM ins;

    RETURN pg_catalog.jsonb_build_object(
        'requested', requested_count,
        'inserted', COALESCE(pg_catalog.array_length(inserted_ids, 1), 0),
        'inserted_ids', pg_catalog.to_jsonb(COALESCE(inserted_ids, ARRAY[]::UUID[]))
    );
END;
$$;

REVOKE ALL ON FUNCTION public.insert_research_lab_provider_usage_ledger_rows(JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.insert_research_lab_provider_usage_ledger_rows(JSONB)
    TO service_role;

COMMIT;
