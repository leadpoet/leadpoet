-- Research Lab: global private-model lineage generation for exact activation.
--
-- Rollout order is intentionally migration first, then an exact all-component
-- runtime restart.  During that bounded cutover, every legacy version-event
-- insert fails closed because it does not carry the v1 generation contract.
-- This prevents an old supersede write from removing the active model before
-- its corresponding old activation write is rejected by the new guard.

BEGIN;

CREATE OR REPLACE FUNCTION public.research_lab_private_model_lineage_generation()
RETURNS TABLE(generation BIGINT)
LANGUAGE sql
STABLE
SECURITY INVOKER
AS $$
    SELECT pg_catalog.count(*)::BIGINT AS generation
      FROM public.research_lab_private_model_version_events;
$$;

REVOKE ALL ON FUNCTION public.research_lab_private_model_lineage_generation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_private_model_lineage_generation()
    TO service_role;

CREATE OR REPLACE FUNCTION public.guard_research_lab_one_active_private_model_version()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    conflicting_version TEXT;
    activation_protocol TEXT;
    expected_generation_text TEXT;
    expected_generation BIGINT;
    actual_generation BIGINT;
BEGIN
    -- Every append participates in the global generation.  Taking the lock
    -- before the status branch makes COUNT(*) an atomic generation compare.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_private_model_version'),
        pg_catalog.hashtext('one_active_version')
    );

    activation_protocol := NEW.event_doc->>'activation_protocol_version';
    IF activation_protocol IS DISTINCT FROM
       'leadpoet.private-model-activation.v1' THEN
        RAISE EXCEPTION
            'research_lab_private_model_activation_protocol_required'
            USING ERRCODE = '23514';
    END IF;
    IF pg_catalog.jsonb_typeof(
        NEW.event_doc->'expected_global_lineage_generation'
    ) IS DISTINCT FROM 'number' THEN
        RAISE EXCEPTION
            'research_lab_private_model_lineage_generation_required'
            USING ERRCODE = '23514';
    END IF;
    expected_generation_text :=
        NEW.event_doc->>'expected_global_lineage_generation';
    IF expected_generation_text !~ '^(0|[1-9][0-9]*)$' THEN
        RAISE EXCEPTION
            'research_lab_private_model_lineage_generation_invalid'
            USING ERRCODE = '23514';
    END IF;
    expected_generation := expected_generation_text::BIGINT;

    SELECT pg_catalog.count(*)::BIGINT
      INTO actual_generation
      FROM public.research_lab_private_model_version_events;
    IF expected_generation <> actual_generation THEN
        RAISE EXCEPTION
            'research_lab_private_model_lineage_generation_conflict: expected %, actual %',
            expected_generation,
            actual_generation
            USING ERRCODE = '40001';
    END IF;

    IF NEW.version_status <> 'active' THEN
        RETURN NEW;
    END IF;

    SELECT latest.private_model_version_id
      INTO conflicting_version
      FROM (
        SELECT DISTINCT ON (event.private_model_version_id)
               event.private_model_version_id,
               event.version_status
          FROM public.research_lab_private_model_version_events event
         WHERE event.private_model_version_id <> NEW.private_model_version_id
         ORDER BY event.private_model_version_id,
                  event.seq DESC,
                  event.created_at DESC
      ) latest
     WHERE latest.version_status = 'active'
     LIMIT 1;

    IF conflicting_version IS NOT NULL THEN
        RAISE EXCEPTION
            'research_lab_one_active_version_conflict: version % cannot become active while version % is active; supersede it first',
            NEW.private_model_version_id,
            conflicting_version
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.guard_research_lab_one_active_private_model_version()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_one_active_private_model_version()
    TO service_role;

COMMENT ON FUNCTION public.research_lab_private_model_lineage_generation() IS
    'Read-only append count consumed as the global private-model activation generation.';
COMMENT ON FUNCTION public.guard_research_lab_one_active_private_model_version() IS
    'Serializes every private-model lineage append, requires its exact v1 global generation, and admits active events only when no other version is active.';

COMMIT;
