-- Retire every legacy incentive, audit-weight, and settlement relation.
-- Arena scoring/model/qualification data and generic provider evidence remain.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $requires_202$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_weight_state_schema_v1()') IS NULL
     OR (public.lab_arena_weight_state_schema_v1() ->> 'version')::INTEGER <> 202 THEN
    RAISE EXCEPTION 'apply 202-arena-accepted-weight-state.sql first';
  END IF;
END;
$requires_202$;

DO $requires_source_add_retirement$
BEGIN
  IF pg_catalog.to_regclass('public.research_lab_source_add_submissions') IS NOT NULL
     OR pg_catalog.to_regclass('public.research_lab_source_add_reward_obligations') IS NOT NULL
     OR pg_catalog.to_regclass('public.research_lab_source_catalog') IS NOT NULL THEN
    RAISE EXCEPTION 'apply 198-retire-research-lab-source-add-schema.sql first';
  END IF;
END;
$requires_source_add_retirement$;

DROP FUNCTION IF EXISTS public.lab_arena_weight_inputs_v1(BIGINT, INTEGER, TEXT, BOOLEAN, BOOLEAN);

-- Migration 192 attached temporary allocation/weight fences to otherwise
-- shared evidence tables. Remove the retired triggers before their functions.
DO $drop_temporary_weight_fences$
BEGIN
  IF pg_catalog.to_regclass('public.research_lab_attested_execution_results_v2') IS NOT NULL THEN
    DROP TRIGGER IF EXISTS enforce_temporary_testnet401_execution_result_epoch_scope_v1
      ON public.research_lab_attested_execution_results_v2;
  END IF;
  IF pg_catalog.to_regclass('public.transparency_log') IS NOT NULL THEN
    DROP TRIGGER IF EXISTS enforce_temporary_testnet401_weight_submission_epoch_scope_v1
      ON public.transparency_log;
  END IF;
END;
$drop_temporary_weight_fences$;

DO $drop_retired_functions$
DECLARE fn RECORD;
BEGIN
  FOR fn IN
    SELECT n.nspname, p.proname,
           pg_catalog.pg_get_function_identity_arguments(p.oid) AS args
    FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public' AND (
      p.proname LIKE 'persist_research_lab_chain_realized_%'
      OR p.proname LIKE 'research_lab_chain_realized_%'
      OR p.proname LIKE 'persist_research_lab_allocation_%'
      OR p.proname LIKE 'research_lab_allocation_frontier_%'
      OR p.proname LIKE 'research_lab_compact_weight_%'
      OR p.proname IN (
        'research_lab_stateful_subnet_epoch_cutover_preflight_v1',
        'research_lab_stateful_subnet_epoch_legacy_high_water_v1',
        'research_lab_stateful_subnet_epoch_cutover_fence_v1',
        'research_lab_stateful_subnet_epoch_cutover_bind_v1',
        'research_lab_stateful_subnet_epoch_cutover_bind_v2',
        'research_lab_stateful_subnet_epoch_stage_v1',
        'research_lab_stateful_subnet_epoch_stage_v2',
        'research_lab_stateful_subnet_epoch_activate_v1',
        'research_lab_stateful_subnet_epoch_refresh_fence_v1',
        'research_lab_fresh_network_epoch_cutover_public_state_v1',
        'research_lab_champion_lifetime_credit_contract_v1',
        'enforce_temporary_testnet401_execution_result_epoch_scope_v1',
        'enforce_temporary_testnet401_weight_submission_epoch_scope_v1'
      )
    )
    ORDER BY p.oid DESC
  LOOP
    EXECUTE pg_catalog.format('DROP FUNCTION IF EXISTS %I.%I(%s)', fn.nspname, fn.proname, fn.args);
  END LOOP;
END;
$drop_retired_functions$;

DROP VIEW IF EXISTS public.research_lab_finalized_weight_vector_candidates_v1;
DROP VIEW IF EXISTS public.research_lab_finalized_allocation_epochs_v2;
DROP VIEW IF EXISTS public.research_lab_emission_allocation_current;
DROP VIEW IF EXISTS public.research_lab_champion_reward_current;
DROP VIEW IF EXISTS public.research_reimbursement_award_current;
DROP VIEW IF EXISTS public.research_loop_shadow_weight_inputs;
DROP VIEW IF EXISTS public.research_lab_arweave_epoch_audit_anchor_current;
DROP VIEW IF EXISTS public.research_lab_signed_audit_bundle_current;
DROP VIEW IF EXISTS public.research_lab_stateful_subnet_epoch_mapping_v1;

-- The gateway still reads the immutable cutover manifest as its shared epoch
-- namespace. Detach only the two retired weight-history references; preserve
-- every cutover row and the public state reader.
DO $detach_retired_weight_history$
DECLARE fk RECORD;
BEGIN
  IF pg_catalog.to_regclass('public.research_lab_stateful_subnet_epoch_cutovers_v1') IS NOT NULL THEN
    FOR fk IN
      SELECT con.conname
      FROM pg_catalog.pg_constraint con
      WHERE con.conrelid = 'public.research_lab_stateful_subnet_epoch_cutovers_v1'::regclass
        AND con.contype = 'f'
        AND con.confrelid IN (
          pg_catalog.to_regclass('public.research_lab_attested_weight_bundles_v2'),
          pg_catalog.to_regclass('public.research_lab_attested_weight_finalizations_v2')
        )
    LOOP
      EXECUTE pg_catalog.format(
        'ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 DROP CONSTRAINT %I',
        fk.conname
      );
    END LOOP;
  END IF;
END;
$detach_retired_weight_history$;

-- Children precede parents. Avoid CASCADE so an unknown live dependency stops
-- retirement rather than deleting scoring or model data.
DROP TABLE IF EXISTS public.research_lab_compact_weight_publication_intents_v2;
DROP TABLE IF EXISTS public.research_lab_compact_weight_authorities_v2;
DROP TABLE IF EXISTS public.research_lab_compact_weight_submissions_v2;
DROP TABLE IF EXISTS public.research_lab_chain_realized_obligation_credits_v1;
DROP TABLE IF EXISTS public.research_lab_chain_realized_epoch_settlements_v1;
DROP TABLE IF EXISTS public.research_lab_chain_realized_settlement_activation_v1;
DROP TABLE IF EXISTS public.research_lab_allocation_settlement_frontier_activation_v2;
DROP TABLE IF EXISTS public.research_lab_allocation_settlement_frontiers_v2;
DROP TABLE IF EXISTS public.research_lab_attested_weight_finalizations_v2;
DROP TABLE IF EXISTS public.research_lab_attested_publication_events_v2;
DROP TABLE IF EXISTS public.research_lab_attested_weight_bundles_v2;
DROP TABLE IF EXISTS public.research_lab_attested_weight_bundles;
DROP TABLE IF EXISTS public.research_lab_legacy_allocation_nonfinalizations_v2;
DROP TABLE IF EXISTS public.research_lab_legacy_finalized_allocation_migrations_v2;
DROP TABLE IF EXISTS public.research_lab_emission_allocation_snapshots;
DROP TABLE IF EXISTS public.research_lab_champion_reward_events;
DROP TABLE IF EXISTS public.research_lab_champion_reward_obligations;
DROP TABLE IF EXISTS public.research_lab_arweave_epoch_audit_anchor_events;
DROP TABLE IF EXISTS public.research_lab_arweave_epoch_audit_anchors;
DROP TABLE IF EXISTS public.research_lab_signed_audit_bundle_events;
DROP TABLE IF EXISTS public.research_lab_signed_audit_bundles;
DROP TABLE IF EXISTS public.research_reimbursement_award_events;
DROP TABLE IF EXISTS public.research_reimbursement_schedules;
DROP TABLE IF EXISTS public.research_reimbursement_awards;
DROP TABLE IF EXISTS public.research_weight_input_snapshots;

DO $replace_state_shape$
DECLARE constraint_name TEXT;
BEGIN
  FOR constraint_name IN
    SELECT con.conname FROM pg_catalog.pg_constraint con
    WHERE con.conrelid = 'public.lab_arena_accepted_weight_states'::regclass
      AND con.contype = 'c'
      AND pg_catalog.pg_get_constraintdef(con.oid) LIKE '%fixed_allocations%'
  LOOP
    EXECUTE pg_catalog.format('ALTER TABLE public.lab_arena_accepted_weight_states DROP CONSTRAINT %I', constraint_name);
  END LOOP;
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_constraint
    WHERE conrelid = 'public.lab_arena_accepted_weight_states'::regclass
      AND conname = 'lab_arena_accepted_weight_state_fields_ck'
  ) THEN
    ALTER TABLE public.lab_arena_accepted_weight_states
      ADD CONSTRAINT lab_arena_accepted_weight_state_fields_ck CHECK (
        state_doc ?& ARRAY['schema_version','network','genesis_hash','netuid','epoch',
          'valid_from_block','valid_until_block','reward_basis','burn_hotkey',
          'issued_at','state_hash','signature']
      );
  END IF;
END;
$replace_state_shape$;

-- Keep fulfillment scoring/history. These two columns are inert historical
-- incentive facts; no active Arena code reads or writes them.
DO $assert_retirement$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_accepted_weight_states') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL THEN
    RAISE EXCEPTION 'Arena state must remain available';
  END IF;
  IF pg_catalog.to_regclass('public.research_lab_emission_allocation_snapshots') IS NOT NULL
     OR pg_catalog.to_regclass('public.research_reimbursement_awards') IS NOT NULL
     OR pg_catalog.to_regclass('public.research_lab_compact_weight_authorities_v2') IS NOT NULL
     OR pg_catalog.to_regclass('public.research_lab_chain_realized_epoch_settlements_v1') IS NOT NULL THEN
    RAISE EXCEPTION 'legacy incentive retirement is incomplete';
  END IF;
END;
$assert_retirement$;

CREATE OR REPLACE FUNCTION public.lab_arena_incentive_retirement_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SET search_path = pg_catalog, public
AS $$ SELECT pg_catalog.jsonb_build_object(
  'schema_version', 'leadpoet.lab_arena.incentive_retirement_schema.v1',
  'version', 203
) $$;
REVOKE ALL ON FUNCTION public.lab_arena_incentive_retirement_schema_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_incentive_retirement_schema_v1() TO service_role;

NOTIFY pgrst, 'reload schema';
COMMIT;
