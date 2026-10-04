-- Stop a repeatedly failing execute host from taking more work after its
-- existing leases expire. Do not expire leases early or change retry policy.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $execute_host_cooldown$
DECLARE
  v_definition TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_anchor CONSTANT TEXT := $anchor$    -- lab_arena_zero_call_score_runner_cooldown_v1: score candidates only.$anchor$;
  v_insert CONSTANT TEXT := $insert$    -- lab_arena_execute_host_cooldown_v1: three distinct, lease-bound host
    -- failures in this stage and frozen-TTL window stop new execute claims.
    AND (runs.kind <> 'execute' OR v_round.status NOT IN ('stage1', 'stage2')
      OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS expired_execute
       WHERE expired_execute.round_id=p_round_id
         AND expired_execute.stage=v_stage
         AND expired_execute.stage_generation=v_round.stage_generation
         AND expired_execute.runner_hotkey=p_runner_hotkey
         AND expired_execute.kind='execute'
         AND expired_execute.status='failed'
         AND expired_execute.terminal_cause='lease_expired'
         AND expired_execute.result_doc IS NULL
         AND expired_execute.output_ref IS NULL
         AND expired_execute.lease_expires_at<=pg_catalog.clock_timestamp()
         AND expired_execute.lease_expires_at>
           pg_catalog.clock_timestamp()-pg_catalog.make_interval(
             secs => (v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
         AND NOT EXISTS (
           SELECT 1 FROM public.lab_arena_ledger AS prior_cost
           WHERE prior_cost.run_id=expired_execute.run_id
         )
         AND EXISTS (
           SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
           WHERE host_error.run_id=expired_execute.run_id
             AND host_error.runner_hotkey=expired_execute.runner_hotkey
             AND host_error.run_kind='execute'
             AND host_error.event_kind='runtime.error'
             AND host_error.content->>'status'='abandoned'
             AND host_error.content->>'failure_stage'='runtime'
             AND host_error.content->>'error_class'='RuntimeHostError'
         )
       GROUP BY expired_execute.runner_hotkey
       HAVING COUNT(DISTINCT expired_execute.assignment_id)>=3
      ))
$insert$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname, p.proacl,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_namespace AS n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment'
    AND p.pronargs=9;
  IF v_definition IS NULL
     OR v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config)) THEN
    RAISE EXCEPTION 'Arena execute cooldown claim security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_definition,'lab_arena_execute_host_cooldown_v1')>0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_insert,'')))
         <> pg_catalog.length(v_insert)
       OR pg_catalog.strpos(v_definition,'lab_arena_zero_call_score_runner_cooldown_v1')=0 THEN
      RAISE EXCEPTION 'Arena execute cooldown claim replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
       IS DISTINCT FROM 'f81a3cdb122caad413d38e4d24573b5d67fb07f480b79bf7ce3ec052419eb488'
     OR (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_anchor,'')))
          <> pg_catalog.length(v_anchor) THEN
    RAISE EXCEPTION 'Arena execute cooldown claim function preimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_anchor,v_insert||v_anchor);
END;
$execute_host_cooldown$;

COMMIT;
