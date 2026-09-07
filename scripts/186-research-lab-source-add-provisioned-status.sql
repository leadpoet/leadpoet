-- 186-research-lab-source-add-provisioned-status.sql
-- Rename the active SOURCE_ADD eligible state after retiring the Leg 2 judge.
-- Historical migration files remain immutable; this updates the live table and
-- the deployed Leg 1 functions that still compare the old literal.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $source_add_status_table$
BEGIN
    IF to_regclass('public.research_lab_source_add_provisioning_events') IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD provisioning events table is missing';
    END IF;
END;
$source_add_status_table$;

-- Expand the live check for the transition.  The old value is accepted only
-- during this transaction so an already-running old process cannot make the
-- table invalid while the rows are moved.
DO $source_add_status_drop_check$
DECLARE
    constraint_row RECORD;
BEGIN
    FOR constraint_row IN
        SELECT c.conname
          FROM pg_catalog.pg_constraint c
         WHERE c.conrelid =
               'public.research_lab_source_add_provisioning_events'::REGCLASS
           AND c.contype = 'c'
           AND (
               c.conname = 'research_lab_source_provision_status_check'
               OR pg_catalog.pg_get_constraintdef(c.oid)
                  ILIKE '%provisioned_autoresearch_eligible%'
           )
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_source_add_provisioning_events DROP CONSTRAINT %I',
            constraint_row.conname
        );
    END LOOP;
END;
$source_add_status_drop_check$;
ALTER TABLE public.research_lab_source_add_provisioning_events
    ADD CONSTRAINT research_lab_source_provision_status_check
    CHECK (provision_status IN (
        'approved_pending_provision', 'provisioned',
        'provisioned_autoresearch_eligible', 'disabled'
    )) NOT VALID;

-- Provisioning events are append-only, but changing the state literal keeps
-- the current projection and historical state queries consistent.  The
-- existing eligible trigger ignores the old literal and therefore does not
-- re-run Leg 1 admission checks for this state-only rewrite.
UPDATE public.research_lab_source_add_provisioning_events
   SET provision_status = 'provisioned'
 WHERE provision_status = 'provisioned_autoresearch_eligible';

-- Update deployed function bodies without rewriting the immutable migration
-- history.  pg_get_functiondef returns CREATE OR REPLACE FUNCTION text, so
-- this is idempotent and covers both the post-accept and provenance-era
-- Leg 1 predicates currently installed in public.
DO $source_add_status_functions$
DECLARE
    function_row RECORD;
    function_definition TEXT;
BEGIN
    FOR function_row IN
        SELECT p.oid
          FROM pg_catalog.pg_proc p
          JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
         WHERE n.nspname = 'public'
           AND p.prokind = 'f'
           AND pg_catalog.pg_get_functiondef(p.oid)
               LIKE '%provisioned_autoresearch_eligible%'
    LOOP
        function_definition := pg_catalog.replace(
            pg_catalog.pg_get_functiondef(function_row.oid),
            '''provisioned_autoresearch_eligible''',
            '''provisioned'''
        );
        EXECUTE function_definition;
    END LOOP;
END;
$source_add_status_functions$;

ALTER TABLE public.research_lab_source_add_provisioning_events
    DROP CONSTRAINT IF EXISTS research_lab_source_provision_status_check;
ALTER TABLE public.research_lab_source_add_provisioning_events
    ADD CONSTRAINT research_lab_source_provision_status_check
    CHECK (provision_status IN (
        'approved_pending_provision', 'provisioned', 'disabled'
    )) NOT VALID;
ALTER TABLE public.research_lab_source_add_provisioning_events
    VALIDATE CONSTRAINT
        research_lab_source_provision_status_check;

COMMIT;
