-- Do not prefer a live runner that the existing round fault guard disqualifies.
-- Share its exact predicate once; preserve TTL handoff, paid work and all locks.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $quarantined_alternatives516$
DECLARE
  target OID:=to_regprocedure('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)');
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  old_select CONSTANT TEXT := $old_select$  SELECT * INTO v_run FROM public.lab_arena_runs AS runs$old_select$;
  new_select CONSTANT TEXT := $new_select$  -- One statement snapshot and one fault scan for claimant and alternatives.
  -- IS NOT TRUE preserves the original WHERE behavior for an unknown phase.
  WITH round_quarantined_runners AS MATERIALIZED (
    SELECT host_fault.runner_hotkey FROM public.lab_arena_runs AS host_fault
    WHERE (v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage1_judged',
          'stage1_scored', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored')) IS NOT TRUE
      -- An idle claim does not need a fault-history scan. This cheap indexed
      -- demand gate shares the candidate statement's snapshot.
      AND EXISTS (SELECT 1 FROM public.lab_arena_runs AS pending_fault_check
        WHERE pending_fault_check.round_id=p_round_id
          AND pending_fault_check.status='pending')
      AND host_fault.round_id=p_round_id
      AND (
        (host_fault.status='leased'
          -- lab_arena_untransitioned_host_fault_guard_v1: a signed
          -- runtime fault remains evidence before the expiry tick.
          AND host_fault.lease_expires_at IS NOT NULL)
        OR (host_fault.status='failed'
          AND host_fault.terminal_cause='lease_expired'
          AND host_fault.lease_expires_at<=pg_catalog.clock_timestamp())
      )
      AND host_fault.result_doc IS NULL
      AND host_fault.output_ref IS NULL
     AND (NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS prior_cost
       WHERE prior_cost.run_id=host_fault.run_id
     ) OR public.lab_arena__execute_host_interruption_v1(host_fault))
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
       WHERE host_error.run_id=host_fault.run_id
         AND host_error.runner_hotkey=host_fault.runner_hotkey
         AND host_error.run_kind=host_fault.kind
         AND host_error.event_kind='runtime.error'
         AND host_error.content->>'status'='abandoned'
         AND host_error.content->>'failure_stage'='runtime'
         AND host_error.content->>'error_class'='RuntimeHostError'
     )
    GROUP BY host_fault.runner_hotkey
    HAVING COUNT(DISTINCT host_fault.assignment_id)>=3
  )
  SELECT * INTO v_run FROM public.lab_arena_runs AS runs$new_select$;
  old_gate CONSTANT TEXT := $old_gate$    -- lab_arena_active_host_fault_claim_guard_v1: three distinct
    -- authenticated, zero-call host faults stop new claims.
    -- lab_arena_round_host_fault_quarantine_v1: count both kinds and stages
    -- in this round; a new round_id resets the runner's eligibility.
    AND (v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage1_judged',
      'stage1_scored', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored')
      OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS host_fault
       WHERE host_fault.round_id=p_round_id
         AND host_fault.runner_hotkey=p_runner_hotkey
         AND (
           (host_fault.status='leased'
             -- lab_arena_untransitioned_host_fault_guard_v1: a signed
             -- runtime fault remains evidence before the expiry tick.
             AND host_fault.lease_expires_at IS NOT NULL)
           OR (host_fault.status='failed'
             AND host_fault.terminal_cause='lease_expired'
             AND host_fault.lease_expires_at<=pg_catalog.clock_timestamp())
         )
         AND host_fault.result_doc IS NULL
         AND host_fault.output_ref IS NULL
        AND (NOT EXISTS (
          SELECT 1 FROM public.lab_arena_ledger AS prior_cost
          WHERE prior_cost.run_id=host_fault.run_id
        ) OR public.lab_arena__execute_host_interruption_v1(host_fault))
        AND EXISTS (
          SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
          WHERE host_error.run_id=host_fault.run_id
            AND host_error.runner_hotkey=host_fault.runner_hotkey
            AND host_error.run_kind=host_fault.kind
            AND host_error.event_kind='runtime.error'
            AND host_error.content->>'status'='abandoned'
            AND host_error.content->>'failure_stage'='runtime'
            AND host_error.content->>'error_class'='RuntimeHostError'
        )
       GROUP BY host_fault.runner_hotkey
       HAVING COUNT(DISTINCT host_fault.assignment_id)>=3
      ))
$old_gate$;
  new_gate CONSTANT TEXT := $new_gate$    -- lab_arena_active_host_fault_claim_guard_v1: three distinct
    -- authenticated, zero-call host faults stop new claims.
    -- lab_arena_round_host_fault_quarantine_v1: count both kinds and stages
    -- in this round; a new round_id resets the runner's eligibility.
    AND NOT EXISTS (
      SELECT 1 FROM round_quarantined_runners AS quarantined
      WHERE quarantined.runner_hotkey=p_runner_hotkey
    )
$new_gate$;
  old_alternative CONSTANT TEXT := $old_alternative$             AND others.stage_generation = v_round.stage_generation))$old_alternative$;
  new_alternative CONSTANT TEXT := $new_alternative$             AND others.stage_generation = v_round.stage_generation
             AND NOT EXISTS (
               SELECT 1 FROM round_quarantined_runners AS quarantined
               WHERE quarantined.runner_hotkey=others.runner_hotkey
             )))$new_alternative$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Quarantined alternative security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='a1ee8afde0f2f04f6cdd5e9c0cda3d0710c5d9d2b1c72e05e6f604722e7dd345' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'584a2b835f0d74ede019f4076b7000c9a3e8e681342e5e6d356c7fb90c8c7ad1'
    OR (length(definition)-length(replace(definition,old_select,'')))<>length(old_select)
    OR (length(definition)-length(replace(definition,old_gate,'')))<>length(old_gate)
    OR (length(definition)-length(replace(definition,old_alternative,'')))<>length(old_alternative) THEN
    RAISE EXCEPTION 'Quarantined alternative preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(replace(replace(definition,old_select,new_select),old_gate,new_gate),old_alternative,new_alternative);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'a1ee8afde0f2f04f6cdd5e9c0cda3d0710c5d9d2b1c72e05e6f604722e7dd345' THEN
    RAISE EXCEPTION 'Quarantined alternative postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a1ee8afde0f2f04f6cdd5e9c0cda3d0710c5d9d2b1c72e05e6f604722e7dd345' THEN
    RAISE EXCEPTION 'Quarantined alternative readback differs' USING ERRCODE='55000';
  END IF;
END;
$quarantined_alternatives516$;
COMMIT;
