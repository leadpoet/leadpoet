-- Research Lab: DB-side one-active private model version guard.
--
-- Deployment policy:
--   * Apply after scripts 37 and 42.
--   * Additive only: a new guard function + BEFORE INSERT trigger on the
--     append-only version-events table. No schema shape change, no history
--     rewrite, no live-write enablement.
--   * Serializes 'active' version-event inserts (two workers, or a worker and
--     the replay CLI, can otherwise both merge and leave two "active"
--     versions with a last-write-wins champion — bug 4).
--   * Design notes (must stay consistent with gateway/research_lab/promotion.py):
--       - The guard is STRICT: at most one version may have a latest event of
--         'active'. The application therefore writes supersede -> create
--         (the previous champion's latest event becomes 'superseded' before
--         the replacement's 'active' insert), and repairs the crash window
--         between the two writes with reconcile_active_private_model_lineage
--         ("zero active but lineage non-empty -> re-activate newest
--         superseded"), which this guard permits (no other active exists).
--       - Bootstrap registration on a genuinely empty lineage and the
--         operator reregister-active-manifest flow (supersede -> create) are
--         likewise permitted.
--       - A partial unique index is impossible here: current_version_status
--         exists only on the non-materialized view
--         research_lab_private_model_version_current, and the events table is
--         append-only (the prior 'active' event is never deleted). An
--         app-side advisory lock is impractical: the gateway writes via
--         Supabase PostgREST with no persistent SQL session. Hence this
--         trigger, modeled on guard_research_lab_run_claim (scripts/42).
--
--   * BEFORE APPLYING in an environment with live history, run the read-only
--     duplicate-active check at the bottom of this file. If it reports more
--     than one active version (the 2026-07 probe suggested 2 of 17 active
--     events lack a superseding event), supersede the stray version(s) first;
--     the guard will otherwise (correctly) block every new 'active' insert
--     until the duplicate state is resolved.

BEGIN;

CREATE OR REPLACE FUNCTION public.guard_research_lab_one_active_private_model_version()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    conflicting_version TEXT;
BEGIN
    IF NEW.version_status <> 'active' THEN
        RETURN NEW;
    END IF;

    -- Constant key: all 'active' inserts across the whole lineage serialize
    -- on the same advisory lock (there is exactly one champion slot).
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_private_model_version'),
        pg_catalog.hashtext('one_active_version')
    );

    SELECT latest.private_model_version_id
      INTO conflicting_version
      FROM (
        SELECT DISTINCT ON (e.private_model_version_id)
               e.private_model_version_id,
               e.version_status
          FROM public.research_lab_private_model_version_events e
         WHERE e.private_model_version_id <> NEW.private_model_version_id
         ORDER BY e.private_model_version_id, e.seq DESC, e.created_at DESC
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

DROP TRIGGER IF EXISTS guard_research_lab_one_active_version_insert
    ON public.research_lab_private_model_version_events;
CREATE TRIGGER guard_research_lab_one_active_version_insert
    BEFORE INSERT ON public.research_lab_private_model_version_events
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_one_active_private_model_version();

COMMENT ON FUNCTION public.guard_research_lab_one_active_private_model_version() IS
    'Serializes private model version activation and permits an active event only when no other version''s latest event is active.';

COMMIT;

-- Smoke checks after applying:
--
--   SELECT tgname
--   FROM pg_trigger
--   WHERE tgname = 'guard_research_lab_one_active_version_insert';
--
-- Read-only duplicate-active check (run BEFORE applying; also exposed as the
-- `check-duplicate-active` admin command):
--
--   SELECT latest.private_model_version_id,
--          latest.version_status,
--          latest.created_at
--     FROM (
--       SELECT DISTINCT ON (e.private_model_version_id)
--              e.private_model_version_id,
--              e.version_status,
--              e.created_at
--         FROM public.research_lab_private_model_version_events e
--        ORDER BY e.private_model_version_id, e.seq DESC, e.created_at DESC
--     ) latest
--    WHERE latest.version_status = 'active'
--    ORDER BY latest.created_at DESC;
