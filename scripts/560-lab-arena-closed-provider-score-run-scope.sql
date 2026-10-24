-- Reject already-ineligible execution/run states before global head lookups.
-- Keep original outer scope, latest heads, cursor order and every full proof.
-- No index, payer, billing, settlement, history or publication change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $closed_provider_score_scope560$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)';
  definition TEXT; updated TEXT; security_before JSONB; security_after JSONB;
  prerequisite RECORD;
BEGIN
  -- Preserve all exact billing dependencies before accepting application/replay.
  -- Validate dependencies even on replay; changed eligibility needs review.
  FOR prerequisite IN SELECT * FROM (VALUES
    ('public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)','e347886710c2461366607941044af1123d092622cfb65ecae1c5b70a1a22f914'),
    ('public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)','67fcd9dc83d29491aa4608e069b6680ac652e85b93b1feac64cc01a72b1f93f8'),
    ('public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)','0b876ab01b0c9e88ca9dc4a1cfcd3eb9e8cc13dc453a2bd418593e5dd3a96f82'),
    ('public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)','d5885be424024b95f7aac98b175872857baa6b0856e407ee9108b2382581ff0d'),
    ('public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)','8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49')
  ) AS required(signature,sha256) LOOP
    IF encode(extensions.digest(pg_get_functiondef(to_regprocedure(prerequisite.signature)),
        'sha256'),'hex') IS DISTINCT FROM prerequisite.sha256 THEN
      RAISE EXCEPTION 'Closed provider score scope helper preimage differs: %',prerequisite.signature;
    END IF;
  END LOOP;
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(o.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO definition,security_before
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF definition IS NULL OR security_before IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'s',ARRAY['search_path=pg_catalog, public'],'sql') THEN
    RAISE EXCEPTION 'Closed provider score scope security preimage differs';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='9762e92771dfa3333811caabeaec56d01b5c02c888869193c258660796d29bdd' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM
      '629daf05a1f6aa4415450d3e39b69186ffa1ad121182189ca295b71939abd289' THEN
    RAISE EXCEPTION 'Closed provider score scope exact preimage differs';
  END IF;
  updated:=replace(definition,$old_run_scope560$      -- Preserve the same global latest-head fence before materialization.
$old_run_scope560$,$new_run_scope560$      -- The unchanged outer eligible_runs join requires this exact run state.
      AND EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS scoped_score_run
        WHERE scoped_score_run.run_id = uncertainty.run_id
          AND scoped_score_run.kind = 'score'
          AND (
            (scoped_score_run.status = 'failed'
             AND scoped_score_run.terminal_cause IN (
               'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
               'stage_closed', 'judge_error', 'judge_timeout'
             ))
            OR (scoped_score_run.status = 'accepted'
                AND scoped_score_run.terminal_cause = 'accepted')
          )
      )
      -- Preserve the same global latest-head fence before materialization.
$new_run_scope560$);
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM
      '9762e92771dfa3333811caabeaec56d01b5c02c888869193c258660796d29bdd' THEN
    RAISE EXCEPTION 'Closed provider score scope target differs';
  END IF;
  EXECUTE updated;
  SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO security_after
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')
      IS DISTINCT FROM '9762e92771dfa3333811caabeaec56d01b5c02c888869193c258660796d29bdd' OR security_after IS DISTINCT FROM security_before THEN
    RAISE EXCEPTION 'Closed provider score scope readback differs';
  END IF;
END;
$closed_provider_score_scope560$;
NOTIFY pgrst,'reload schema';
COMMIT;
