-- Migration 232: retire the populated legacy lead-validation corpus and the
-- remaining legacy Research Lab execution telemetry selected by the operator.
--
-- This is a forward-only teardown. Supabase reports a completed daily backup
-- from 2026-09-12 09:52 UTC, without PITR; this migration does not claim a
-- fresh full-row export. Current Arena, baseline/rebenchmark, provider
-- accounting, reward/weight state, and the shared SOURCE_ADD attested graph
-- must remain unchanged.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

-- Fail closed on the wrong database or on a partial prerequisite deployment.
DO $require_preserved_state$
DECLARE
    relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'lab_arena_accepted_weight_states',
        'lab_arena_chain_outcomes',
        'lab_arena_company_judgment_reservations',
        'lab_arena_company_judgments',
        'lab_arena_judgment_cache',
        'lab_arena_ledger',
        'lab_arena_restart_claim_control',
        'lab_arena_rounds',
        'lab_arena_runs',
        'lab_arena_submission_credentials',
        'lab_arena_submissions',
        'qualification_baselines',
        'qualification_private_icp_sets',
        'research_lab_official_baseline_action_attempts_v1',
        'research_lab_official_baseline_action_terminals_v1',
        'research_lab_official_baseline_runs_v1',
        'research_lab_official_baseline_unit_closures_v1',
        'research_lab_attested_ancestry_activations_v2',
        'research_lab_attested_ancestry_checkpoints_v2',
        'research_lab_attested_artifact_links_v2',
        'research_lab_attested_boot_identities_v2',
        'research_lab_attested_business_artifact_links_v2',
        'research_lab_attested_execution_receipts_v2',
        'research_lab_attested_execution_results_v2',
        'research_lab_attested_host_operations_v2',
        'research_lab_attested_receipt_edges_v2',
        'research_lab_attested_receipt_transport_v2',
        'research_lab_attested_transport_attempts_v2',
        'validation_evidence_private'
    ]
    LOOP
        IF pg_catalog.to_regclass('public.' || relation_name) IS NULL THEN
            RAISE EXCEPTION
                'required preserved relation public.% is missing',
                relation_name;
        END IF;
    END LOOP;
END;
$require_preserved_state$;

-- Migration 231 already removed the two empty scoring queue tables. Stop if a
-- new scoring table appeared after the reviewed catalog snapshot instead of
-- deleting it through a prefix sweep.
DO $reject_unreviewed_scoring_tables$
DECLARE
    unexpected_tables TEXT[];
BEGIN
    SELECT pg_catalog.array_agg(c.relname ORDER BY c.relname)
      INTO unexpected_tables
      FROM pg_catalog.pg_class c
      JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
     WHERE n.nspname = 'public'
       AND c.relkind IN ('r', 'p')
       AND c.relname LIKE 'research_lab_scoring\_%' ESCAPE '\'
       AND c.relname NOT IN (
           'research_lab_scoring_category_results',
           'research_lab_scoring_dispatch_events',
           'research_lab_scoring_icp_events',
           'research_lab_scoring_icp_executions',
           'research_lab_scoring_run_events',
           'research_lab_scoring_runs'
       );
    IF unexpected_tables IS NOT NULL THEN
        RAISE EXCEPTION
            'unreviewed research_lab_scoring tables exist: %',
            unexpected_tables;
    END IF;
END;
$reject_unreviewed_scoring_tables$;

-- These functions either write a target table or are the reviewed contract
-- and caller closure. Verify every still-present body before removing it. A
-- changed function requires a new dependency review and aborts this migration.
DO $verify_retired_routine_bodies$
DECLARE
    expected RECORD;
    routine_oid OID;
    actual_md5 TEXT;
BEGIN
    FOR expected IN
        SELECT *
        FROM (VALUES
            ('public.append_research_lab_provider_outcome_checkpoint_v2(jsonb)', 'f0a9d6e8d830bd685230c086ed5a12a6'),
            ('public.append_research_lab_provider_outcome_checkpoints_v2(jsonb)', '31a489738d09675ed3166409db37393c'),
            ('public.claim_ops_research_lab_event_monitor_lease(text,text,integer)', '4302395c73fb7bde104369761a804621'),
            ('public.claim_ops_research_lab_improvement_analysis(integer)', '0c09d93438b55caae5e6c9a0c552344b'),
            ('public.extract_lead_blob_fields()', '996ed868937937c7910e729fddb67482'),
            ('public.prevent_research_lab_provider_outcome_checkpoint_mutation()', '171cdf81bf40c5b1a9326b60faa4285e'),
            ('public.refresh_miner_test_leads()', '788cf3ff2d62eb7ec6a85f66968c8e38'),
            ('public.research_lab_corpus_source_watermark(uuid)', 'def57a7d8e4dd2ea4c00f3b0718fc0c8'),
            ('public.research_lab_mark_corpus_complete(uuid,uuid)', '3e255cb1522df79cd529d9c77e196002'),
            ('public.research_lab_mark_corpus_complete(uuid,uuid,text)', '975ac21e83560ecae9076acd8f42289c'),
            ('public.research_lab_provider_outcome_contention_contract_v2()', '9f667cd7f7ea62ff2d47f983d0e58da7'),
            ('public.research_lab_provider_outcome_contention_contract_v3()', '93bfd737acdaefb3e863a7f03fe1e595'),
            ('public.research_lab_provider_persistence_batch_contract_v1()', 'c3a1356e7396632de393681f3bcb5b37'),
            ('public.research_lab_terminal_runs_needing_corpus(integer,boolean)', 'e642cf586ad07359d0ba630aa95dc9e0')
        ) AS expected_routine(signature, body_md5)
    LOOP
        routine_oid := pg_catalog.to_regprocedure(expected.signature);
        IF routine_oid IS NULL THEN
            CONTINUE;
        END IF;
        SELECT pg_catalog.md5(p.prosrc)
          INTO actual_md5
          FROM pg_catalog.pg_proc p
         WHERE p.oid = routine_oid;
        IF actual_md5 IS DISTINCT FROM expected.body_md5 THEN
            RAISE EXCEPTION
                'routine % changed after review: expected %, found %',
                expected.signature,
                expected.body_md5,
                actual_md5;
        END IF;
    END LOOP;
END;
$verify_retired_routine_bodies$;

-- This guard also protects retained Research Loop ticket history. Remove only
-- the deleted auto-loop evidence source and keep every other evidence check.
DO $verify_shared_ticket_guard_body$
DECLARE
    routine_oid OID;
    actual_md5 TEXT;
BEGIN
    routine_oid := pg_catalog.to_regprocedure(
        'public.research_lab_ticket_has_unpaid_lifecycle_evidence(uuid)'
    );
    IF routine_oid IS NULL THEN
        RAISE EXCEPTION
            'required shared ticket evidence guard is missing';
    END IF;
    SELECT pg_catalog.md5(p.prosrc)
      INTO actual_md5
      FROM pg_catalog.pg_proc p
     WHERE p.oid = routine_oid;
    IF actual_md5 NOT IN (
        '9e24b68996825050e169601eb2632a74',
        '12321f60dfacab18ead8342c9656a9e8'
    ) THEN
        RAISE EXCEPTION
            'shared ticket evidence guard changed after review: found %',
            actual_md5;
    END IF;
END;
$verify_shared_ticket_guard_body$;

CREATE OR REPLACE FUNCTION public.research_lab_ticket_has_unpaid_lifecycle_evidence(
    target_ticket_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
VOLATILE
SET search_path = ''
AS $$
DECLARE
    legacy_evidence_exists BOOLEAN := FALSE;
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.research_loop_start_payments p
        WHERE p.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_loop_start_credit_events ce
        WHERE ce.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_loop_run_queue_events q
        WHERE q.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_loop_receipts r
        WHERE r.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_lab_candidate_artifacts c
        WHERE c.ticket_id = target_ticket_id
    )
    THEN
        RETURN TRUE;
    END IF;

    IF pg_catalog.to_regclass('public.research_loop_start_credits') IS NOT NULL THEN
        EXECUTE
            'SELECT EXISTS (SELECT 1 FROM public.research_loop_start_credits c '
            || 'WHERE c.ticket_id = $1)'
            INTO legacy_evidence_exists
            USING target_ticket_id;
        IF legacy_evidence_exists THEN
            RETURN TRUE;
        END IF;
    END IF;

    IF pg_catalog.to_regclass('public.research_loop_balance_ledger') IS NOT NULL THEN
        EXECUTE
            'SELECT EXISTS (SELECT 1 FROM public.research_loop_balance_ledger bl '
            || 'WHERE bl.ticket_id = $1)'
            INTO legacy_evidence_exists
            USING target_ticket_id;
        IF legacy_evidence_exists THEN
            RETURN TRUE;
        END IF;
    END IF;

    RETURN FALSE;
END;
$$;

-- Stop the only reviewed cron job that reaches a target table. The other four
-- jobs do not depend on this deletion closure.
DO $unschedule_legacy_jobs$
DECLARE
    legacy_job RECORD;
BEGIN
    FOR legacy_job IN
        SELECT jobid, jobname
          FROM cron.job
         WHERE jobname = 'refresh-miner-test-leads'
         ORDER BY jobid
    LOOP
        IF NOT cron.unschedule(legacy_job.jobid) THEN
            RAISE EXCEPTION
                'could not unschedule legacy cron job % (%)',
                legacy_job.jobname,
                legacy_job.jobid;
        END IF;
    END LOOP;
END;
$unschedule_legacy_jobs$;

-- Drop callers before callees. These corpus routines are the complete
-- transitive caller closure of research_lab_corpus_source_watermark(uuid).
DROP FUNCTION IF EXISTS public.research_lab_terminal_runs_needing_corpus(
    INTEGER,
    BOOLEAN
);
DROP FUNCTION IF EXISTS public.research_lab_mark_corpus_complete(UUID, UUID);
DROP FUNCTION IF EXISTS public.research_lab_mark_corpus_complete(
    UUID,
    UUID,
    TEXT
);
DROP FUNCTION IF EXISTS public.research_lab_corpus_source_watermark(UUID);

DROP FUNCTION IF EXISTS public.append_research_lab_provider_outcome_checkpoint_v2(
    JSONB
);
DROP FUNCTION IF EXISTS public.append_research_lab_provider_outcome_checkpoints_v2(
    JSONB
);
DROP FUNCTION IF EXISTS public.claim_ops_research_lab_event_monitor_lease(
    TEXT,
    TEXT,
    INTEGER
);
DROP FUNCTION IF EXISTS public.claim_ops_research_lab_improvement_analysis(INTEGER);
DROP FUNCTION IF EXISTS public.refresh_miner_test_leads();
DROP FUNCTION IF EXISTS public.research_lab_provider_outcome_contention_contract_v2();
DROP FUNCTION IF EXISTS public.research_lab_provider_outcome_contention_contract_v3();
DROP FUNCTION IF EXISTS public.research_lab_provider_persistence_batch_contract_v1();

-- Drop the complete reviewed view dependency graph, from wrappers to base
-- views. No CASCADE is used; an unknown dependent object aborts the migration.
DROP VIEW IF EXISTS public.research_lab_private_benchmark_dashboard_telemetry;
DROP VIEW IF EXISTS public.research_lab_scoring_dashboard_telemetry;
DROP VIEW IF EXISTS public.research_lab_scoring_dashboard_telemetry_legacy;
DROP VIEW IF EXISTS public.research_lab_scoring_dashboard_telemetry_v2;
DROP VIEW IF EXISTS public.research_lab_scoring_icp_execution_current;
DROP VIEW IF EXISTS public.research_lab_scoring_run_current;
DROP VIEW IF EXISTS public.research_lab_auto_research_loop_current;

-- Keep validation evidence as historical data while removing its only inbound
-- dependency on the retired lead table.
ALTER TABLE public.validation_evidence_private
    DROP CONSTRAINT IF EXISTS validation_evidence_private_lead_id_fkey;

-- Children precede parents. PostgreSQL removes each target's owned indexes,
-- policies, constraints, and triggers. Unknown external dependencies fail and
-- roll back the entire transaction because CASCADE is intentionally absent.
DROP TABLE IF EXISTS public.research_lab_scoring_category_results;
DROP TABLE IF EXISTS public.research_lab_scoring_dispatch_events;
DROP TABLE IF EXISTS public.research_lab_provider_cost_events;
DROP TABLE IF EXISTS public.research_lab_scoring_icp_events;
DROP TABLE IF EXISTS public.research_lab_scoring_run_events;
DROP TABLE IF EXISTS public.research_lab_scoring_icp_executions;
DROP TABLE IF EXISTS public.research_lab_scoring_runs;

DROP TABLE IF EXISTS public.research_lab_auto_research_loop_events;
DROP TABLE IF EXISTS public.research_lab_provider_outcome_checkpoints_v2;
DROP TABLE IF EXISTS public.ops_research_lab_event_notifications;
DROP TABLE IF EXISTS public.ops_research_lab_event_monitor_state;
DROP TABLE IF EXISTS public.test_leads_for_miners;
DROP TABLE IF EXISTS public.miner_test_leads;
DROP TABLE IF EXISTS public.leads_private_backup;
DROP TABLE IF EXISTS public.leads_private;

-- The two identity sequences are owned and normally disappear with their
-- tables. The queue-position sequence is already unowned in production.
DROP SEQUENCE IF EXISTS public.test_leads_for_miners_id_seq;
DROP SEQUENCE IF EXISTS public.miner_test_leads_id_seq;
DROP SEQUENCE IF EXISTS public.leads_private_queue_position_seq;

-- Trigger functions exclusive to deleted tables can now be removed. Shared
-- append-only and stateful epoch fence functions remain in place.
DROP FUNCTION IF EXISTS public.extract_lead_blob_fields();
DROP FUNCTION IF EXISTS public.prevent_research_lab_provider_outcome_checkpoint_mutation();

-- PostgreSQL does not consistently record dependencies from stored function
-- bodies or cron command text. Scan every non-system schema after the drops so
-- a private or dynamic caller cannot survive as a latent runtime failure.
DO $assert_no_hidden_callers$
DECLARE
    retired_name TEXT;
    name_pattern TEXT;
    function_callers TEXT[];
    cron_callers TEXT[];
BEGIN
    FOREACH retired_name IN ARRAY ARRAY[
        'leads_private',
        'leads_private_backup',
        'miner_test_leads',
        'ops_research_lab_event_monitor_state',
        'ops_research_lab_event_notifications',
        'research_lab_auto_research_loop_events',
        'research_lab_provider_cost_events',
        'research_lab_provider_outcome_checkpoints_v2',
        'research_lab_scoring_category_results',
        'research_lab_scoring_dispatch_events',
        'research_lab_scoring_icp_events',
        'research_lab_scoring_icp_executions',
        'research_lab_scoring_run_events',
        'research_lab_scoring_runs',
        'test_leads_for_miners',
        'append_research_lab_provider_outcome_checkpoint_v2',
        'append_research_lab_provider_outcome_checkpoints_v2',
        'claim_ops_research_lab_event_monitor_lease',
        'claim_ops_research_lab_improvement_analysis',
        'extract_lead_blob_fields',
        'prevent_research_lab_provider_outcome_checkpoint_mutation',
        'refresh_miner_test_leads',
        'research_lab_corpus_source_watermark',
        'research_lab_mark_corpus_complete',
        'research_lab_provider_outcome_contention_contract_v2',
        'research_lab_provider_outcome_contention_contract_v3',
        'research_lab_provider_persistence_batch_contract_v1',
        'research_lab_terminal_runs_needing_corpus'
    ]
    LOOP
        name_pattern := '(^|[^a-zA-Z0-9_])' || retired_name
            || '([^a-zA-Z0-9_]|$)';
        SELECT pg_catalog.array_agg(
                   pg_catalog.format(
                       '%I.%I(%s)',
                       n.nspname,
                       p.proname,
                       pg_catalog.pg_get_function_identity_arguments(p.oid)
                   )
                   ORDER BY n.nspname, p.proname, p.oid
               )
          INTO function_callers
          FROM pg_catalog.pg_proc p
          JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
         WHERE p.prokind IN ('f', 'p')
           AND n.nspname NOT IN ('pg_catalog', 'information_schema')
           AND n.nspname NOT LIKE 'pg\_%' ESCAPE '\'
           AND pg_catalog.pg_get_functiondef(p.oid) ~* name_pattern;
        IF function_callers IS NOT NULL THEN
            RAISE EXCEPTION
                'remaining functions reference retired object %: %',
                retired_name,
                function_callers;
        END IF;

        SELECT pg_catalog.array_agg(
                   pg_catalog.format('%s:%s', jobid, jobname)
                   ORDER BY jobid
               )
          INTO cron_callers
          FROM cron.job
         WHERE command ~* name_pattern;
        IF cron_callers IS NOT NULL THEN
            RAISE EXCEPTION
                'remaining cron jobs reference retired object %: %',
                retired_name,
                cron_callers;
        END IF;
    END LOOP;
END;
$assert_no_hidden_callers$;

DO $assert_retirement_complete$
DECLARE
    relation_name TEXT;
    routine_signature TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'leads_private',
        'leads_private_backup',
        'leads_private_queue_position_seq',
        'miner_test_leads',
        'miner_test_leads_id_seq',
        'ops_research_lab_event_monitor_state',
        'ops_research_lab_event_notifications',
        'research_lab_auto_research_loop_current',
        'research_lab_auto_research_loop_events',
        'research_lab_private_benchmark_dashboard_telemetry',
        'research_lab_provider_cost_events',
        'research_lab_provider_outcome_checkpoints_v2',
        'research_lab_scoring_category_results',
        'research_lab_scoring_dashboard_telemetry',
        'research_lab_scoring_dashboard_telemetry_legacy',
        'research_lab_scoring_dashboard_telemetry_v2',
        'research_lab_scoring_dispatch_events',
        'research_lab_scoring_icp_events',
        'research_lab_scoring_icp_execution_current',
        'research_lab_scoring_icp_executions',
        'research_lab_scoring_run_current',
        'research_lab_scoring_run_events',
        'research_lab_scoring_runs',
        'test_leads_for_miners',
        'test_leads_for_miners_id_seq'
    ]
    LOOP
        IF pg_catalog.to_regclass('public.' || relation_name) IS NOT NULL THEN
            RAISE EXCEPTION 'legacy relation public.% remains', relation_name;
        END IF;
    END LOOP;

    FOREACH routine_signature IN ARRAY ARRAY[
        'public.append_research_lab_provider_outcome_checkpoint_v2(jsonb)',
        'public.append_research_lab_provider_outcome_checkpoints_v2(jsonb)',
        'public.claim_ops_research_lab_event_monitor_lease(text,text,integer)',
        'public.claim_ops_research_lab_improvement_analysis(integer)',
        'public.extract_lead_blob_fields()',
        'public.prevent_research_lab_provider_outcome_checkpoint_mutation()',
        'public.refresh_miner_test_leads()',
        'public.research_lab_corpus_source_watermark(uuid)',
        'public.research_lab_mark_corpus_complete(uuid,uuid)',
        'public.research_lab_mark_corpus_complete(uuid,uuid,text)',
        'public.research_lab_provider_outcome_contention_contract_v2()',
        'public.research_lab_provider_outcome_contention_contract_v3()',
        'public.research_lab_provider_persistence_batch_contract_v1()',
        'public.research_lab_terminal_runs_needing_corpus(integer,boolean)'
    ]
    LOOP
        IF pg_catalog.to_regprocedure(routine_signature) IS NOT NULL THEN
            RAISE EXCEPTION 'legacy routine % remains', routine_signature;
        END IF;
    END LOOP;

    IF EXISTS (
        SELECT 1 FROM cron.job
         WHERE jobname = 'refresh-miner-test-leads'
    ) THEN
        RAISE EXCEPTION 'legacy cron job remains';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_class c
          JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public'
           AND c.relkind IN ('r', 'p')
           AND c.relname LIKE 'research_lab_scoring\_%' ESCAPE '\'
    ) THEN
        RAISE EXCEPTION 'a research_lab_scoring table remains';
    END IF;

    IF pg_catalog.pg_get_functiondef(
        'public.research_lab_ticket_has_unpaid_lifecycle_evidence(uuid)'::REGPROCEDURE
    ) LIKE '%research_lab_auto_research_loop_events%' THEN
        RAISE EXCEPTION 'shared ticket guard still references deleted auto-loop events';
    END IF;
END;
$assert_retirement_complete$;

NOTIFY pgrst, 'reload schema';

COMMIT;
