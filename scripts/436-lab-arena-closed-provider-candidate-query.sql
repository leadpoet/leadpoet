-- Bound closed-round candidate validation without changing eligibility or billing.
-- Both audited Deepline helpers require a failed score run and the same causes.
-- Materialize uncertainty IDs once, apply only those necessary conditions and
-- the original scope/latest-head guards, then retain both full helper checks.
-- OFFSET 0 preserves cursor order before expensive validation and prevents the
-- outer provider selector from evaluating the Deepline selector more than once.
-- No indexes, settlement rules, active V1/V2 listers or publication paths change.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $closed_provider_candidate_query$
DECLARE
  v_signature TEXT;
  v_definition TEXT;
  v_source TEXT;
  v_updated TEXT;
  v_old_hash TEXT;
  v_new_hash TEXT;
BEGIN
  -- Pin the helper bodies that prove the redundant run prefilter. A future
  -- eligibility change must review this optimization rather than silently
  -- dropping newly eligible liabilities, including cancelled-round calls.
  SELECT prosrc INTO v_source FROM pg_catalog.pg_proc
    WHERE oid = 'public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)'::REGPROCEDURE;
  IF pg_catalog.md5(v_source) IS DISTINCT FROM '27a8859101fa0074dbd12670225ec8fd' THEN
    RAISE EXCEPTION 'Closed provider query helper prerequisite changed: lab_arena__closed_score_dynamic_uncertainty_v1';
  END IF;
  SELECT prosrc INTO v_source FROM pg_catalog.pg_proc
    WHERE oid = 'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)'::REGPROCEDURE;
  IF pg_catalog.md5(v_source) IS DISTINCT FROM 'b5534b589c356dfa036edb4d47d1e2c5' THEN
    RAISE EXCEPTION 'Closed provider query helper prerequisite changed: lab_arena__closed_host_score_success_uncertainty_v1';
  END IF;

  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)',
    'public.lab_arena_next_closed_provider_reconciliation_v1(text,text,integer,text,bigint)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid), p.prosrc
      INTO v_definition, v_source
    FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid = p.proowner
    WHERE p.oid = pg_catalog.to_regprocedure(v_signature)
      AND o.rolname = 'lab_arena_owner'
      AND p.proacl::TEXT = '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile = 's'
      AND p.proconfig = ARRAY['search_path=pg_catalog, public'];
    IF v_definition IS NULL THEN
      RAISE EXCEPTION 'Closed provider query security shape changed: %', v_signature;
    END IF;
    IF v_signature LIKE 'public.lab_arena_next_closed_deepline_%' THEN
      v_old_hash := '90be86d436ed75e08c2f9f03bb884fe0';
      v_new_hash := '85a2a079b5464e2bbc433b5dff8fd7c6';
      v_updated := pg_catalog.replace(v_definition, v_source, $candidate$
  -- lab_arena_closed_provider_candidate_query_436
  WITH uncertainties AS MATERIALIZED (
    SELECT entry_id, round_id, run_id, call_identity
    FROM public.lab_arena_ledger
    WHERE entry_kind = 'uncertain' AND provider = 'deepline'
  )
  SELECT COALESCE((
    SELECT pg_catalog.jsonb_build_object(
      'status', 'ok', 'uncertain_entry_id', uncertainty.entry_id,
      'round_id', uncertainty.round_id, 'run_id', uncertainty.run_id
    )
    FROM (
      SELECT uncertainty.entry_id, uncertainty.round_id, uncertainty.run_id,
        uncertainty.entry_id <= COALESCE(p_after_entry_id, 0) AS cursor_wrapped
      FROM uncertainties AS uncertainty
      JOIN public.lab_arena_rounds AS rounds
        ON rounds.round_id = uncertainty.round_id
      JOIN public.lab_arena_runs AS eligible_run
        ON eligible_run.run_id = uncertainty.run_id
       AND eligible_run.kind = 'score' AND eligible_run.status = 'failed'
       AND eligible_run.terminal_cause IN (
         'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
         'stage_closed', 'judge_error', 'judge_timeout'
       )
      WHERE rounds.status IN ('cancelled', 'published')
        AND rounds.configuration_doc ->> 'mode' = p_mode
        AND rounds.configuration_doc ->> 'network_name' = p_network_name
        AND rounds.configuration_doc ->> 'netuid' = p_netuid::TEXT
        AND (COALESCE(p_round_id, '') = '' OR uncertainty.round_id = p_round_id)
        AND (
          rounds.status = 'cancelled'
          OR rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
               IN ('successful_calls_v1', 'successful_calls_per_icp_v1')
        )
        AND NOT EXISTS (
          SELECT 1 FROM public.lab_arena_ledger AS later
          WHERE later.call_identity = uncertainty.call_identity
            AND later.entry_id > uncertainty.entry_id
        )
      ORDER BY cursor_wrapped, uncertainty.entry_id
      OFFSET 0
    ) AS uncertainty
    WHERE (
      public.lab_arena__closed_score_dynamic_uncertainty_v1(uncertainty.entry_id)
      OR public.lab_arena__closed_host_score_success_uncertainty_v1(uncertainty.entry_id)
    )
    ORDER BY uncertainty.cursor_wrapped, uncertainty.entry_id
    LIMIT 1
  ), pg_catalog.jsonb_build_object('status', 'none'));
$candidate$);
    ELSE
      v_old_hash := 'e44109bc96f0be89c600079ab07a1ee7';
      v_new_hash := '5d4c5aa3d0f68bfbc5557c2aa06be37a';
      v_updated := pg_catalog.replace(v_definition,
        ') AS doc) original', ') AS doc OFFSET 0) original');
    END IF;
    IF pg_catalog.md5(v_definition) = v_new_hash THEN
      CONTINUE;
    END IF;
    IF pg_catalog.md5(v_definition) <> v_old_hash
       OR pg_catalog.md5(v_updated) <> v_new_hash THEN
      RAISE EXCEPTION 'Closed provider query preimage changed: %', v_signature;
    END IF;
    -- CREATE OR REPLACE retains owner, ACL and settings.
    EXECUTE v_updated;
    IF pg_catalog.md5(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE)) <> v_new_hash THEN
      RAISE EXCEPTION 'Closed provider query readback changed: %', v_signature;
    END IF;
  END LOOP;
END;
$closed_provider_candidate_query$;

NOTIFY pgrst, 'reload schema';
COMMIT;
