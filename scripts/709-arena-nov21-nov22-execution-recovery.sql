-- Owner-approved operational recovery. Existing workers select, bill and judge
-- all work. The predecessor champion guard remains responsible for the handoff.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '45s';
SET LOCAL TIME ZONE 'UTC';
SET LOCAL ROLE lab_arena_owner;
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion', 0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control', 0));

DO $recovery709$
DECLARE
  days CONSTANT text[] := ARRAY['arena-2026-11-21','arena-2026-11-22'];
  request CONSTANT text := 'nov21-nov22-window709-v1';
  parent_id CONSTANT text := 'arena-2026-11-21:baseline-2026-11-21:1:1:2';
  parent public.lab_arena_runs;
  r public.lab_arena_rounds;
  n public.lab_arena_rounds;
  activation timestamptz := date_trunc('second',clock_timestamp());
  windows jsonb;
  evidence jsonb;
  cost jsonb;
  slot integer;
  existing integer;
BEGIN
  IF current_setting('session_replication_role') <> 'origin' THEN
    RAISE EXCEPTION 'Recovery709 requires ordinary triggers';
  END IF;
  PERFORM 1 FROM public.lab_arena_rounds WHERE round_id=ANY(days) ORDER BY round_id FOR UPDATE;
  SELECT count(*) INTO existing FROM public.lab_arena_scoring_supersessions
   WHERE round_id=ANY(days) AND request_id=request;
  IF existing=2 AND NOT EXISTS (
    SELECT 1 FROM public.lab_arena_scoring_supersessions
     WHERE round_id=ANY(days) AND request_id=request AND
       (recovery_doc->>'operation' IS DISTINCT FROM 'round_operational_update_v1'
        OR recovery_doc->>'reason' IS DISTINCT FROM 'nov21_provider_close_and_deferred_nov22'))
  THEN RETURN; END IF;
  IF existing<>0 OR (SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=ANY(days))<>2
     OR (public.lab_arena_restart_guard_state_v1()->>'paused')::boolean IS DISTINCT FROM false
     OR EXISTS (SELECT 1 FROM public.lab_arena_rounds WHERE arena_netuid=71 AND rewards_enabled
                 AND round_id>days[2] AND configuration_doc->>'mode'='live' AND status NOT IN ('open','cancelled'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=days[2])
     OR (SELECT encode(extensions.digest(jsonb_agg(to_jsonb(a) ORDER BY run_id)::text,'sha256'),'hex')
         FROM public.lab_arena_runs a WHERE round_id=days[1]) IS DISTINCT FROM
        '1afe7ab8ec375dde75b4b7cfdf29e317ae88b9f40f9bad55c42a9b3b6313693b'
  THEN RAISE EXCEPTION 'Recovery709 round, run, or control preimage differs'; END IF;
  SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=parent_id FOR UPDATE;
  IF encode(extensions.digest(to_jsonb(parent)::text,'sha256'),'hex') <>
        '72e3b24a0e75db8faaae0a9073809c66e208dd420fd86177728d1dac2a427627'
     OR parent.kind<>'execute' OR parent.status<>'failed' OR parent.terminal_cause<>'provider_error'
     OR parent.output_ref IS NOT NULL OR parent.per_icp_score IS NOT NULL OR parent.terminal_doc IS NOT NULL
     OR public.lab_arena__baseline_recovery_run_v1(jsonb_populate_record(parent,
          jsonb_build_object('run_id',parent.assignment_id||':3','attempt',3))) IS DISTINCT FROM true
     OR encode(extensions.digest(pg_get_functiondef('public.lab_arena_close_stage(text,smallint)'::regprocedure),'sha256'),'hex')
        <> 'd4261afcdc7a08c474ffd3e27befd824fc0d1915d595fcf2f662eed6944e2513'
  THEN RAISE EXCEPTION 'Recovery709 parent or installed close rule differs'; END IF;
  cost:=public.lab_arena__successful_icp_cost_state(days[1],parent.submission_id,1);
  IF (cost->>'inflight_calls')::bigint IS DISTINCT FROM 0::bigint
     OR (cost->>'success_unresolved_calls')::bigint IS DISTINCT FROM 0::bigint THEN
    RAISE EXCEPTION 'Recovery709 provider work requires reconciliation';
  END IF;

  FOR slot IN 1..2 LOOP
    SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=days[slot];
    IF encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex') IS DISTINCT FROM
        (ARRAY['01cdb85fd6342d2b3584ef72867fd9eab7ee0a9d0d1d847abc7fd98ddf8d9b1a',
               'f683ac7f9fa3ea5f5e2aa94de761600dc346178823129f91131dace4ec088409'])[slot]
       OR (SELECT encode(extensions.digest(jsonb_agg(to_jsonb(s) ORDER BY submission_id)::text,'sha256'),'hex')
            FROM public.lab_arena_submissions s WHERE round_id=r.round_id) IS DISTINCT FROM
         (ARRAY['e19a8bd452c01066d3b0741351ab05bd3af1f2f68126c9af8946c44c7e8aae99',
                '40b22eee07ca179609807625894986622dfa53d2786bea1e00c27ef63a1d17ab'])[slot]
       OR r.status IS DISTINCT FROM (ARRAY['stage1_judged','open'])[slot]
       OR r.published_at IS NOT NULL OR r.reward_activated_at IS NOT NULL
       OR (SELECT encode(extensions.digest(icps::text,'sha256'),'hex') FROM public.qualification_private_icp_sets
            WHERE set_id=(ARRAY[20261120,20261121])[slot]) IS DISTINCT FROM
         (ARRAY['696d512f5734d6d3de93806251505548f5c0a4ff0735670ea3eb6cad82f4b8e9',
                '3512ae5ddfc3f922ec3e5ad51eadfd2ef67d34132a80c02d3bf08f4ea7f31898'])[slot]
    THEN RAISE EXCEPTION 'Recovery709 saved round, cohort or ICP bank differs for %',r.round_id; END IF;

    -- These are upper bounds. Successful earlier completion follows the normal driver.
    -- Give the second day its complete execution window even if its predecessor
    -- publishes at the end of its own recovery window. Intake never reopens.
    SELECT jsonb_object_agg(key,to_char(activation + delta,'YYYY-MM-DD"T"HH24:MI:SS"Z"')) INTO windows
    FROM (VALUES
      ('stage_1_close',       interval '14 hours',                2),
      ('stage_1_scoring_close',interval '16 hours',               2),
      ('stage_2_start',       interval '16 hours 1 second',       2),
      ('stage_2_close',       interval '20 hours',                2),
      ('final_scoring_close', interval '24 hours',                2),
      ('publication_deadline',interval '24 hours 1 second',       2),
      ('stage_2_close',       interval '6 hours',                 1),
      ('final_scoring_close', interval '9 hours',                 1),
      ('publication_deadline',interval '9 hours 1 second',        1)
    ) AS revised(key,delta,which) WHERE which=slot;
    n:=r;
    n.configuration_doc:=jsonb_set(r.configuration_doc,'{schedule}',(r.configuration_doc->'schedule')||windows,false);
    evidence:=jsonb_build_object('operation','round_operational_update_v1',
      'reason','nov21_provider_close_and_deferred_nov22','windows',windows,'activation',activation);
    IF slot=1 THEN
      evidence:=evidence||jsonb_build_object('closure_marker_run_id',parent_id,
        'previous_terminal_doc',parent.terminal_doc,'new_terminal_doc',jsonb_build_object('infrastructure_incomplete',true));
    END IF;
    INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
      actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
    SELECT r.round_id,coalesce(max(generation_floor),0)+1,request,
      'sha256:'||encode(extensions.digest(evidence::text,'sha256'),'hex'),'authorized-nov21-nov22-recovery',
      txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',evidence
    FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
    UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc WHERE round_id=r.round_id;
    IF (SELECT to_jsonb(check_row)-'updated_at' FROM public.lab_arena_rounds check_row
         WHERE round_id=r.round_id) IS DISTINCT FROM to_jsonb(n)-'updated_at' THEN
      RAISE EXCEPTION 'Recovery709 schedule readback differs';
    END IF;
  END LOOP;

  -- Apply exactly the marker 708 now records when this assignment closes.
  UPDATE public.lab_arena_runs SET terminal_doc='{"infrastructure_incomplete":true}'::jsonb WHERE run_id=parent_id;
  IF (SELECT to_jsonb(check_row)-ARRAY['terminal_doc','updated_at'] FROM public.lab_arena_runs check_row
       WHERE run_id=parent_id) IS DISTINCT FROM to_jsonb(parent)-ARRAY['terminal_doc','updated_at']
     OR public.lab_arena__publication_execution_incomplete_v1(days[1],parent.submission_id) IS DISTINCT FROM true
     OR public.lab_arena__baseline_final_ready_v1(days[1]) IS DISTINCT FROM false
     OR (SELECT public.lab_arena__baseline_source_current_v1(next_round) FROM public.lab_arena_rounds next_round
          WHERE next_round.round_id=days[2]) IS DISTINCT FROM false
  THEN RAISE EXCEPTION 'Recovery709 closure, finality or champion guard readback differs'; END IF;
END;
$recovery709$;
COMMIT;
