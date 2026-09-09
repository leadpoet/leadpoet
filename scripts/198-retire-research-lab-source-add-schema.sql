-- Retire the SOURCE_ADD intake and execution schema.
--
-- Historical migration files remain unchanged for normal upgrades. Historical
-- allocation snapshots also keep their SOURCE_ADD accounting fields so old
-- epoch replay and weight verification do not change.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Keep old signed rows readable, but reject every new SOURCE_ADD receipt,
-- result, or transport attempt. NOT VALID is deliberate: PostgreSQL enforces
-- these checks for new writes without rescanning immutable history.
DO $retire_source_add_attested_writes$
BEGIN
    IF to_regclass(
        'public.research_lab_attested_execution_receipts_v2'
    ) IS NOT NULL THEN
        ALTER TABLE public.research_lab_attested_execution_receipts_v2
            DROP CONSTRAINT IF EXISTS
                research_lab_attested_receipts_source_add_retired_check;
        ALTER TABLE public.research_lab_attested_execution_receipts_v2
            ADD CONSTRAINT
                research_lab_attested_receipts_source_add_retired_check
            CHECK (purpose NOT LIKE 'research_lab.source_add_%') NOT VALID;
    END IF;

    IF to_regclass(
        'public.research_lab_attested_execution_results_v2'
    ) IS NOT NULL THEN
        ALTER TABLE public.research_lab_attested_execution_results_v2
            DROP CONSTRAINT IF EXISTS
                research_lab_attested_results_source_add_retired_check;
        ALTER TABLE public.research_lab_attested_execution_results_v2
            ADD CONSTRAINT
                research_lab_attested_results_source_add_retired_check
            CHECK (
                operation <> 'source_add_catalog_snapshot_v2'
                AND purpose NOT LIKE 'research_lab.source_add_%'
            ) NOT VALID;
    END IF;

    IF to_regclass(
        'public.research_lab_attested_transport_attempts_v2'
    ) IS NOT NULL THEN
        ALTER TABLE public.research_lab_attested_transport_attempts_v2
            DROP CONSTRAINT IF EXISTS
                research_lab_attested_transport_source_add_retired_check;
        ALTER TABLE public.research_lab_attested_transport_attempts_v2
            ADD CONSTRAINT
                research_lab_attested_transport_source_add_retired_check
            CHECK (purpose NOT LIKE 'research_lab.source_add_%') NOT VALID;
    END IF;
END;
$retire_source_add_attested_writes$;

-- The page RPC returns the miner-status view row type, so remove it before the
-- view. Other functions use JSON or scalar return types and can be removed
-- after their trigger-owning tables.
DROP FUNCTION IF EXISTS
    public.research_lab_source_add_miner_status_page_v1(TEXT, TEXT, INTEGER);

-- Views depend on several SOURCE_ADD tables and must go first.
DROP VIEW IF EXISTS public.research_lab_source_add_miner_status_v1;
DROP VIEW IF EXISTS public.research_lab_source_add_reward_current;
DROP VIEW IF EXISTS public.research_lab_source_add_provenance_leg1_authority_v1;
DROP VIEW IF EXISTS public.research_lab_source_add_provisioning_smoke_current;
DROP VIEW IF EXISTS public.research_lab_source_add_functional_probe_current;
DROP VIEW IF EXISTS public.research_lab_source_add_probe_config_current;
DROP VIEW IF EXISTS public.research_lab_source_add_identity_current;
DROP VIEW IF EXISTS public.research_lab_source_add_provider_origin_current;
DROP VIEW IF EXISTS public.research_lab_source_add_provisioning_current;
DROP VIEW IF EXISTS public.research_lab_source_add_submission_current;

-- Operational state only. Do not use CASCADE: an unexpected external
-- dependency must stop the migration instead of being deleted silently.
DROP TABLE IF EXISTS public.research_lab_source_add_reward_slots;
DROP TABLE IF EXISTS public.research_lab_source_add_reward_intents;
DROP TABLE IF EXISTS public.research_lab_source_add_work_items;
DROP TABLE IF EXISTS public.research_lab_source_add_functional_probe_attempts;
DROP TABLE IF EXISTS public.research_lab_source_add_probe_config_events;
DROP TABLE IF EXISTS public.research_lab_source_add_identity_events;
DROP TABLE IF EXISTS public.research_lab_source_add_provider_origin_events;
DROP TABLE IF EXISTS public.research_lab_source_add_provisioning_events;
DROP TABLE IF EXISTS public.research_lab_source_add_control;
DROP TABLE IF EXISTS public.research_lab_source_add_submissions;
DROP TABLE IF EXISTS public.research_lab_source_add_reward_events;
DROP TABLE IF EXISTS public.research_lab_source_add_reward_obligations;
DROP TABLE IF EXISTS public.research_lab_source_catalog;

-- Drop every overload of the retired API. pg_get_function_identity_arguments
-- makes reruns idempotent.
DO $drop_source_add_functions$
DECLARE
    v_function RECORD;
BEGIN
    FOR v_function IN
        SELECT
            n.nspname,
            p.proname,
            pg_catalog.pg_get_function_identity_arguments(p.oid) AS arguments
        FROM pg_catalog.pg_proc p
        JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public'
          AND (
              p.proname LIKE 'research_lab_source_add_%'
              OR p.proname LIKE 'enforce_research_lab_source_add_%'
              OR p.proname LIKE 'enforce_source_add_%'
              OR p.proname LIKE 'prevent_research_lab_source_add_%'
              OR p.proname LIKE 'release_research_lab_source_add_%'
              OR p.proname LIKE 'assert_research_lab_source_add_%'
              OR p.proname = 'research_lab_source_catalog_replay_contract_v2'
              OR p.proname = 'enforce_research_lab_source_catalog_provider_origin'
              OR p.proname = 'prevent_research_lab_source_catalog_mutation'
          )
        ORDER BY p.oid DESC
    LOOP
        EXECUTE pg_catalog.format(
            'DROP FUNCTION IF EXISTS %I.%I(%s)',
            v_function.nspname,
            v_function.proname,
            v_function.arguments
        );
    END LOOP;
END;
$drop_source_add_functions$;

-- Fail closed if a future edit broadens cleanup into Arena or removes the
-- immutable allocation field used when replaying old epochs.
DO $assert_historical_weight_and_arena_state_retained$
BEGIN
    IF to_regclass('public.lab_arena_submissions') IS NULL
       OR (
           to_regclass(
               'public.research_lab_emission_allocation_snapshots'
           ) IS NOT NULL
           AND NOT EXISTS (
           SELECT 1
           FROM pg_catalog.pg_attribute a
           WHERE a.attrelid = to_regclass(
               'public.research_lab_emission_allocation_snapshots'
           )
             AND a.attname = 'source_add_alpha_percent'
             AND NOT a.attisdropped
           )
       ) THEN
        RAISE EXCEPTION
            'Arena and historical allocation replay state must remain available';
    END IF;
END;
$assert_historical_weight_and_arena_state_retained$;

NOTIFY pgrst, 'reload schema';

COMMIT;
