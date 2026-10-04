-- A score lease abandoned before any provider call has not received its
-- second judgment. Give only that proved attempt-2 host failure one final
-- attempt. The ordinary two-attempt rule and all old rows remain unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $score_host_retry$
DECLARE
  v_definition TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_anchor CONSTANT TEXT := $anchor$    IF v_run.attempt < LEAST(5, 2 +
       (SELECT count(*) FROM public.lab_arena_runs AS failed_run
        WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required)) AND v_run.stage_generation = v_round.stage_generation THEN$anchor$;
  v_replacement CONSTANT TEXT := $replacement$    IF (
      v_run.attempt < LEAST(5, 2 +
        (SELECT count(*) FROM public.lab_arena_runs AS failed_run
         WHERE failed_run.assignment_id = v_run.assignment_id
           AND failed_run.champion_restart_required))
      OR (
        -- lab_arena_zero_call_score_host_retry_v1
        v_run.kind = 'score'
        AND v_run.attempt = 2
        AND v_round.status = 'stage' || v_run.stage::TEXT || '_scoring'
        AND pg_catalog.clock_timestamp() <
          (v_round.configuration_doc #>> ARRAY[
            'schedule', CASE WHEN v_run.stage = 1
              THEN 'stage_1_scoring_close' ELSE 'final_scoring_close' END
          ])::TIMESTAMPTZ
        AND v_run.result_doc IS NULL
        AND v_run.output_ref IS NULL
        AND NOT EXISTS (
          SELECT 1 FROM public.lab_arena_ledger AS cost
          WHERE cost.run_id = v_run.run_id
        )
        AND EXISTS (
          SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
          WHERE host_error.run_id = v_run.run_id
            AND host_error.runner_hotkey = v_run.runner_hotkey
            AND host_error.round_id = v_run.round_id
            AND host_error.assignment_id = v_run.assignment_id
            AND host_error.attempt = v_run.attempt
            AND host_error.run_kind = 'score'
            AND host_error.event_kind = 'runtime.error'
            AND host_error.content ->> 'status' = 'abandoned'
            AND host_error.content ->> 'failure_stage' = 'runtime'
            AND host_error.content ->> 'error_class' = 'RuntimeHostError'
        )
        AND NOT EXISTS (
          SELECT 1 FROM public.lab_arena_trajectory_events AS provider_event
          WHERE provider_event.run_id = v_run.run_id
            AND provider_event.event_kind LIKE 'provider.%'
        )
      )
    ) AND v_run.stage_generation = v_round.stage_generation THEN$replacement$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_definition, v_owner, v_acl, v_security, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE n.nspname = 'public' AND p.proname = 'lab_arena_expire_leases'
    AND p.pronargs = 1;
  IF v_definition IS NULL
     OR v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public' = ANY(v_config))
     OR pg_catalog.strpos(v_definition,
       'lab_arena_judgment_group_expiry_handoff_v1') = 0 THEN
    RAISE EXCEPTION 'Arena score host retry expiry security shape differs'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.strpos(v_definition,
       'lab_arena_zero_call_score_host_retry_v1') > 0 THEN
    IF pg_catalog.strpos(v_definition, v_replacement) = 0
       OR pg_catalog.strpos(v_definition, v_anchor) > 0 THEN
      RAISE EXCEPTION 'Arena score host retry replay differs'
        USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
       IS DISTINCT FROM
       '6db3ae0ffd11b21c867930e6df271bb8eef16aa56c3d52feae6a0190a89151b5'
     OR pg_catalog.strpos(v_definition, v_anchor) = 0
     OR pg_catalog.strpos(pg_catalog.substr(v_definition,
       pg_catalog.strpos(v_definition, v_anchor) + pg_catalog.length(v_anchor)),
       v_anchor) > 0 THEN
    RAISE EXCEPTION 'Arena score host retry expiry preimage differs'
      USING ERRCODE = '55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_anchor, v_replacement);
END;
$score_host_retry$;

COMMIT;
