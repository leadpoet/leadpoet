-- Migration 226: retire the legacy Fulfillment request, scoring, and
-- translation-cache schema.
-- Arena, qualification, provider accounting, and weight state remain intact.
--
-- The retired relations contain Fulfillment client and miner history. Back up
-- that data before this migration. The gateway and validators must stop every
-- Fulfillment writer before apply. This migration is the forward-only teardown;
-- it does not reinterpret or migrate Fulfillment rows into Arena.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Fail before deletion if this is not a current Arena database. These are the
-- shared relations whose continued presence represents the active competition,
-- scoring, provider-accounting, and weight paths. miner_test_leads remains the
-- qualification corpus used by refresh_miner_test_leads().
DO $require_shared_arena_state$
BEGIN
    IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
       OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
       OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
       OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL
       OR pg_catalog.to_regclass(
           'public.lab_arena_accepted_weight_states'
       ) IS NULL
       OR pg_catalog.to_regclass('public.miner_test_leads') IS NULL THEN
        RAISE EXCEPTION
            'Arena, provider, weight, and qualification state must exist before Fulfillment retirement';
    END IF;
END;
$require_shared_arena_state$;

-- Trigger functions cannot be dropped while a trigger still owns them. Every
-- non-internal trigger attached to a relation deleted below is itself exclusive
-- to that relation. Remove those triggers explicitly, without CASCADE.
DO $drop_fulfillment_relation_triggers$
DECLARE
    v_trigger RECORD;
BEGIN
    FOR v_trigger IN
        SELECT n.nspname, c.relname, t.tgname
        FROM pg_catalog.pg_trigger t
        JOIN pg_catalog.pg_class c ON c.oid = t.tgrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname IN (
              'fulfillment_requests',
              'fulfillment_submissions',
              'fulfillment_scores',
              'fulfillment_score_consensus',
              'role_translations'
          )
          AND NOT t.tgisinternal
        ORDER BY t.oid DESC
    LOOP
        EXECUTE pg_catalog.format(
            'DROP TRIGGER IF EXISTS %I ON %I.%I',
            v_trigger.tgname,
            v_trigger.nspname,
            v_trigger.relname
        );
    END LOOP;
END;
$drop_fulfillment_relation_triggers$;

-- Use an exact routine allowlist. Do not use a prefix sweep: the database also
-- contains shared qualification and Research Lab routines whose definitions use
-- words such as "fulfillment" but do not belong to the retired product.
DROP FUNCTION IF EXISTS public.fulfillment_accept_commit(UUID, TEXT, JSONB);
DROP FUNCTION IF EXISTS public.fulfillment_close_window(UUID, TEXT);
DROP FUNCTION IF EXISTS public.fulfillment_release_lifecycle_lock(BIGINT);
DROP FUNCTION IF EXISTS public.fulfillment_try_lifecycle_lock(BIGINT);
DROP FUNCTION IF EXISTS public.fulfillment_upsert_consensus(JSONB);
DROP FUNCTION IF EXISTS public.fulfillment_upsert_scores(JSONB, TEXT);
DROP FUNCTION IF EXISTS public.fulfillment_claim_finalization(
    UUID,
    BIGINT,
    TIMESTAMPTZ
);
DROP FUNCTION IF EXISTS public.get_chain_held_count(UUID);
DROP FUNCTION IF EXISTS public.get_chain_root_num_leads(UUID);
DROP FUNCTION IF EXISTS public.get_chain_summaries(UUID[]);
DROP FUNCTION IF EXISTS public.get_chain_winners(UUID);
DROP FUNCTION IF EXISTS public.get_fulfillment_graph_summary(UUID[]);
DROP FUNCTION IF EXISTS public.get_fulfillment_rejection_stats();
DROP FUNCTION IF EXISTS public.get_rejection_reason_histogram(UUID[]);

-- Children precede parents. No CASCADE is permitted: an unknown external view,
-- foreign key, or routine must stop and roll back the migration.
DROP TABLE IF EXISTS public.fulfillment_score_consensus;
DROP TABLE IF EXISTS public.fulfillment_scores;
DROP TABLE IF EXISTS public.fulfillment_submissions;
DROP TABLE IF EXISTS public.fulfillment_requests;
DROP TABLE IF EXISTS public.role_translations;

DO $assert_fulfillment_retired_and_shared_state_preserved$
BEGIN
    IF pg_catalog.to_regclass('public.fulfillment_requests') IS NOT NULL
       OR pg_catalog.to_regclass('public.fulfillment_submissions') IS NOT NULL
       OR pg_catalog.to_regclass('public.fulfillment_scores') IS NOT NULL
       OR pg_catalog.to_regclass(
           'public.fulfillment_score_consensus'
       ) IS NOT NULL
       OR pg_catalog.to_regclass('public.role_translations') IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.fulfillment_accept_commit(uuid,text,jsonb)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.fulfillment_close_window(uuid,text)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.fulfillment_release_lifecycle_lock(bigint)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.fulfillment_try_lifecycle_lock(bigint)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.fulfillment_upsert_consensus(jsonb)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.fulfillment_upsert_scores(jsonb,text)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.fulfillment_claim_finalization(uuid,bigint,timestamp with time zone)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.get_chain_held_count(uuid)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.get_chain_root_num_leads(uuid)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.get_chain_summaries(uuid[])'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.get_chain_winners(uuid)'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.get_fulfillment_graph_summary(uuid[])'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.get_fulfillment_rejection_stats()'
       ) IS NOT NULL
       OR pg_catalog.to_regprocedure(
           'public.get_rejection_reason_histogram(uuid[])'
       ) IS NOT NULL THEN
        RAISE EXCEPTION 'legacy Fulfillment schema retirement is incomplete';
    END IF;

    IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
       OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
       OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
       OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL
       OR pg_catalog.to_regclass(
           'public.lab_arena_accepted_weight_states'
       ) IS NULL
       OR pg_catalog.to_regclass('public.miner_test_leads') IS NULL
       OR pg_catalog.to_regprocedure(
           'public.refresh_miner_test_leads()'
       ) IS NULL THEN
        RAISE EXCEPTION
            'Fulfillment retirement removed shared Arena, provider, weight, or qualification state';
    END IF;
END;
$assert_fulfillment_retired_and_shared_state_preserved$;

NOTIFY pgrst, 'reload schema';

COMMIT;
