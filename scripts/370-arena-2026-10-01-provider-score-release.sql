-- After independent provider recovery proof, requeue only failed October 1
-- baseline judges from the unchanged accepted executions, then release the
-- migration-369 claim hold. Old runs, events, outputs, and costs stay intact.
DO $oct01_release$
DECLARE
  v_round public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_target public.lab_arena_runs;
  v_count INTEGER := 0;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF NOT FOUND OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> ''
     OR v_control.guard_expires_at IS NOT NULL
     OR v_control.restart_phase <> ''
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'Oct01 release conflicts with restart guard';
  END IF;
  IF NOT v_control.operator_paused
     AND v_control.actor_ref = 'oct01-provider-recovery370'
     AND v_control.pause_reason = '' THEN
    RETURN; -- Exact-owner replay after successful atomic release.
  END IF;
  IF NOT v_control.operator_paused
     OR v_control.actor_ref <> 'oct01-provider-recovery369'
     OR v_control.pause_reason <> 'oct01_deepline_outage' THEN
    RAISE EXCEPTION 'Oct01 release does not own provider hold';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-10-01' FOR UPDATE;
  IF NOT FOUND OR v_round.status <> 'stage1_scoring'
     OR v_round.status_generation <> 6 OR v_round.stage_generation <> 5
     OR v_round.benchmark_ref <> 'arena/arena-2026-10-01/benchmark.json'
     OR pg_catalog.clock_timestamp() >=
        (v_round.configuration_doc #>> '{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
     OR pg_catalog.encode(extensions.digest(v_round.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM
        'e7cd2d1d2fea2a1f2ae41b313889545d36ca3e9b09e5d3d4021fec36a99b0a7b'
     OR pg_catalog.encode(extensions.digest(v_round.participants::TEXT,'sha256'),'hex') IS DISTINCT FROM
        '872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19'
     OR (SELECT pg_catalog.encode(extensions.digest(icps::TEXT,'sha256'),'hex')
         FROM public.qualification_private_icp_sets WHERE set_id=20260930) IS DISTINCT FROM
        '8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)::TEXT,
           'sha256'),'hex') FROM public.lab_arena_submissions s
           WHERE round_id='arena-2026-10-01') IS DISTINCT FROM
        '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
             'run_id',run_id,'submission_id',submission_id,'stage',stage,
             'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,
             'output_ref',output_ref,'stage_generation',stage_generation)
             ORDER BY icp_position)::TEXT,'sha256'),'hex')
           FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01'
             AND kind='execute') IS DISTINCT FROM
        '24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND kind='execute'
             AND status='accepted' AND output_ref IS NOT NULL) <> 10
     OR (SELECT pg_catalog.count(DISTINCT icp_position) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND kind='score') <> 10
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND kind='score'
             AND (submission_id <> 'baseline-2026-10-01' OR stage <> 1
                  OR stage_generation <> 5))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND kind='score'
             AND status IN ('leased','submitted')) THEN
    RAISE EXCEPTION 'Oct01 release frozen source or drained scoring preimage changed';
  END IF;
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds r
      WHERE r.round_id <> v_round.round_id
        AND r.configuration_doc->>'mode'='live'
        AND r.status IN ('stage1','stage1_closed','stage1_scoring','stage1_judged',
                         'stage1_scored','stage2','stage2_closed','stage2_scoring',
                         'stage2_judged'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
      WHERE r.round_id <> v_round.round_id AND r.status IN ('leased','submitted')) THEN
    RAISE EXCEPTION 'Oct01 release would overlap another active round';
  END IF;

  -- The latest score row per assignment is the only retry candidate. Existing
  -- automatic retries remain pending; accepted judgments remain authoritative.
  FOR v_target IN
    SELECT DISTINCT ON (assignment_id) * FROM public.lab_arena_runs
    WHERE round_id='arena-2026-10-01' AND kind='score'
    ORDER BY assignment_id,attempt DESC
  LOOP
    IF v_target.status <> 'failed' THEN
      CONTINUE;
    END IF;
    IF v_target.attempt >= 5 OR v_target.output_ref IS NOT NULL
       OR v_target.per_icp_score IS NOT NULL
       OR NOT (
         (v_target.terminal_cause = 'credential_error'
           AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events t
             WHERE t.run_id=v_target.run_id
               AND t.event_kind='provider.response'
               AND t.content->>'operation_id'='scrapingdog.scrape'
               AND t.content->'call'->>'error_code'=
                   'miner_credentials_unavailable'
               AND t.content->'call'->>'provider_status'='403'
               AND (v_target.run_id,t.content->>'action_sequence',
                    t.content->'call'->>'call_identity') IN (
                 ('arena-2026-10-01:baseline-2026-10-01:1:0:score:1',
                  '56','sha256:0ecfb67d5dee9938a55a0362dc5a37893191ac883485f403b0d8eab06e56c09d'),
                 ('arena-2026-10-01:baseline-2026-10-01:1:3:score:2',
                  '11','sha256:f513b2588cd3f4ee4a02aea4beb1b71105737b79674f4f1e5aa305cc242fd1e8')
               )))
         OR (v_target.terminal_cause = 'judge_error'
           AND v_target.result_doc #>> '{failure_diagnostic,reason}' =
               'provider_error')
       )
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs e
           WHERE e.run_id=v_target.scored_run_id AND e.kind='execute'
             AND e.status='accepted' AND e.output_ref IS NOT NULL
             AND e.round_id=v_target.round_id
             AND e.submission_id=v_target.submission_id
             AND e.icp_position=v_target.icp_position)
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs a
           WHERE a.assignment_id=v_target.assignment_id
             AND a.status='accepted') THEN
      RAISE EXCEPTION 'Oct01 release found non-provider or foreign failed judge %',
        v_target.run_id;
    END IF;
    INSERT INTO public.lab_arena_runs (
      run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
      icp_position,attempt,status,lease_generation,stage_generation,kind,
      scored_run_id,previous_runner_hotkey,judgment_cache_key,
      judgment_input_hash,judgment_scope_doc,judgment_group_leader,
      judgment_group_miner_hotkeys,company_judgment_refs
    ) VALUES (
      v_target.assignment_id||':'||(v_target.attempt+1)::TEXT,
      v_target.assignment_id,v_target.round_id,v_target.submission_id,
      v_target.miner_hotkey,v_target.stage,v_target.icp_position,
      v_target.attempt+1,'pending',v_target.lease_generation,
      v_round.stage_generation,v_target.kind,v_target.scored_run_id,
      v_target.runner_hotkey,v_target.judgment_cache_key,
      v_target.judgment_input_hash,v_target.judgment_scope_doc,
      v_target.judgment_group_leader,v_target.judgment_group_miner_hotkeys,
      v_target.company_judgment_refs
    );
    v_count := v_count+1;
  END LOOP;
  IF v_count=0 THEN
    RAISE EXCEPTION 'Oct01 release found no failed provider judges to requeue';
  END IF;
  UPDATE public.lab_arena_restart_claim_control
  SET operator_paused=FALSE,pause_reason='',
      actor_ref='oct01-provider-recovery370',updated_at=pg_catalog.clock_timestamp()
  WHERE singleton AND operator_paused AND actor_ref='oct01-provider-recovery369';
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Oct01 release claim control changed';
  END IF;
END;
$oct01_release$;
