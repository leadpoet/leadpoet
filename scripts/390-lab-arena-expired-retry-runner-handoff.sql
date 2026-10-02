-- Keep a zero-provider expired score retry off the validator that just lost it.
-- Existing expiry already carries previous_runner_hotkey; no expiry rewrite.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- A zero-provider scoring attempt that expired after a host startup failure
-- must not return to that same validator. Keep the ordinary lone-runner
-- fallback for every other failure, and keep the pre-existing execute rule.
DO $zero_call_score_handoff$
DECLARE
  v_definition TEXT;
  v_anchor CONSTANT TEXT := $anchor$    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
         OR NOT EXISTS ($anchor$;
  v_insert CONSTANT TEXT := $insert$    -- lab_arena_zero_call_score_expiry_handoff_v1
    AND (
      runs.previous_runner_hotkey IS NULL
      OR runs.previous_runner_hotkey <> p_runner_hotkey
      OR NOT EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS prior_score
        WHERE prior_score.assignment_id=runs.assignment_id
          AND prior_score.attempt=runs.attempt-1
          AND prior_score.round_id=runs.round_id
          AND prior_score.stage=runs.stage
          AND prior_score.icp_position=runs.icp_position
          AND prior_score.stage_generation=runs.stage_generation
          AND prior_score.submission_id=runs.submission_id
          AND prior_score.miner_hotkey=runs.miner_hotkey
          AND prior_score.scored_run_id IS NOT DISTINCT FROM runs.scored_run_id
          AND prior_score.kind='score' AND runs.kind='score'
          AND prior_score.runner_hotkey=p_runner_hotkey
          AND prior_score.status='failed'
          AND prior_score.terminal_cause='lease_expired'
          AND prior_score.lease_expires_at<=pg_catalog.clock_timestamp()
          AND prior_score.output_ref IS NULL
          AND prior_score.result_doc IS NULL
          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS prior_cost
            WHERE prior_cost.run_id=prior_score.run_id
          )
      )
    )
$insert$;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment'
    AND p.pronargs=9;
  IF v_definition IS NULL
     OR v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config)) THEN
    RAISE EXCEPTION 'Arena score expiry claim security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_definition,'lab_arena_zero_call_score_expiry_handoff_v1')>0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_insert,'')))
         <> pg_catalog.length(v_insert)
       OR pg_catalog.strpos(v_definition,'lab_arena_zero_setup_runner_handoff_v1')=0 THEN
      RAISE EXCEPTION 'Arena score expiry claim replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
       IS DISTINCT FROM '0252fa3efafe895a7cade9169a36493dc8be70a9148a5ca037d0d0dde870a7c0'
     OR pg_catalog.strpos(v_definition,'lab_arena_zero_setup_runner_handoff_v1')=0
     OR (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_anchor,'')))
       <> pg_catalog.length(v_anchor) THEN
    RAISE EXCEPTION 'Arena score expiry claim function preimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_anchor,v_insert||v_anchor);
END;
$zero_call_score_handoff$;

COMMIT;
