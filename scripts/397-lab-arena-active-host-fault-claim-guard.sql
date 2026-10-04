-- Stop new claims on a runner with three active, lease-bound host failures.
-- Existing leases expire normally; the expired-host cooldown stays unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $active_host_fault_guard$
DECLARE
  v_definition TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_anchor CONSTANT TEXT := $anchor$    -- lab_arena_execute_host_cooldown_v1: three distinct, lease-bound host$anchor$;
  v_insert CONSTANT TEXT := $insert$    -- lab_arena_active_host_fault_claim_guard_v1: three current, distinct
    -- lease-bound runtime host faults stop further claims on this runner.
    AND NOT EXISTS (
      SELECT 1 FROM public.lab_arena_runs AS active_fault
      WHERE active_fault.round_id=p_round_id
        AND active_fault.stage=v_stage
        AND active_fault.stage_generation=v_round.stage_generation
        AND active_fault.kind=runs.kind
        AND active_fault.runner_hotkey=p_runner_hotkey
        AND active_fault.status='leased'
        AND active_fault.lease_expires_at>pg_catalog.clock_timestamp()
        AND active_fault.result_doc IS NULL
        AND active_fault.output_ref IS NULL
        AND NOT EXISTS (
          SELECT 1 FROM public.lab_arena_ledger AS prior_cost
          WHERE prior_cost.run_id=active_fault.run_id
        )
        AND EXISTS (
          SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
          WHERE host_error.run_id=active_fault.run_id
            AND host_error.runner_hotkey=active_fault.runner_hotkey
            AND host_error.run_kind=active_fault.kind
            AND host_error.event_kind='runtime.error'
            AND host_error.content->>'status'='abandoned'
            AND host_error.content->>'failure_stage'='runtime'
            AND host_error.content->>'error_class'='RuntimeHostError'
        )
      GROUP BY active_fault.runner_hotkey
      HAVING COUNT(DISTINCT active_fault.assignment_id)>=3
    )
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
    RAISE EXCEPTION 'Arena active host fault claim security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_definition,'lab_arena_active_host_fault_claim_guard_v1')>0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_insert,'')))
         <> pg_catalog.length(v_insert)
       OR pg_catalog.strpos(v_definition,'lab_arena_execute_host_cooldown_v1')=0 THEN
      RAISE EXCEPTION 'Arena active host fault claim replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
       IS DISTINCT FROM '755819d63ea1ae5f330552fd90fbf6905cbd85b6611e1b26e092ba80fb85ebcc'
     OR (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_anchor,'')))
          <> pg_catalog.length(v_anchor) THEN
    RAISE EXCEPTION 'Arena active host fault claim function preimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_anchor,v_insert||v_anchor);
END;
$active_host_fault_guard$;

COMMIT;
