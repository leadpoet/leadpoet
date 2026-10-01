-- Recover the October 1 round after its scoring deadline advanced through the
-- provider claim hold. The provider hold remains in force. This is an exact
-- one-round recovery, not a general scoring reset or claim release.
--
-- Preserve the premature stage-2 state in a cancelled shadow audit round.
-- Only never-started miner execution rows move there. The original accepted
-- baseline executions, judge attempts, receipts, events and costs remain live.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control', 0));
LOCK TABLE public.lab_arena_restart_claim_control IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE MODE;
LOCK TABLE public.lab_arena_trajectory_events IN SHARE MODE;
LOCK TABLE public.qualification_private_icp_sets IN SHARE MODE;

DO $oct01_recover$
DECLARE
  v_round public.lab_arena_rounds;
  v_archive public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_target public.lab_arena_runs;
  v_archive_config JSONB;
  v_archive_participants JSONB;
  v_new_config JSONB;
  v_count INTEGER;
  v_retried INTEGER := 0;
  v_round_id CONSTANT TEXT := 'arena-2026-10-01';
  v_archive_id CONSTANT TEXT := 'arena-2026-10-01-r372archive';
  v_suffix CONSTANT TEXT := ':r372archive';
  v_old_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-01T23:00:02Z","publication_deadline":"2026-10-01T23:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-01T14:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-01T17:00:02Z","stage_2_start":"2026-10-01T14:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
  v_new_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-02T12:00:02Z","publication_deadline":"2026-10-02T12:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-01T20:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-02T08:00:02Z","stage_2_start":"2026-10-01T20:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
  WHERE round_id = v_archive_id FOR UPDATE;
  IF v_archive.round_id IS NOT NULL THEN
    IF v_archive.status <> 'cancelled'
       OR v_archive.configuration_doc ->> 'mode' <> 'shadow'
       OR v_archive.configuration_doc ->> 'recovery_source_round_id' <> v_round_id
       OR v_archive.configuration_doc ->> 'recovery_preimage_score_rows_sha256' <>
            'dad688e5ab73c5f5e6671299f10d914d5462a5b8333d24b9ddb53b467bf58a63'
       OR v_archive.configuration_doc ->> 'recovery_preimage_stage2_rows_sha256' <>
            '2b0443f2ba2cf021628d34bebd575886be427c4749536b2c8879d3b9bc10239a'
       OR v_archive.configuration_doc -> 'schedule' IS DISTINCT FROM v_old_schedule
       OR v_round.configuration_doc -> 'schedule' IS DISTINCT FROM v_new_schedule
       OR (SELECT COUNT(*) FROM public.lab_arena_submissions
           WHERE round_id=v_archive_id) <> 14
       OR (SELECT COUNT(*) FROM public.lab_arena_runs
           WHERE round_id=v_archive_id AND stage=1 AND kind='execute'
             AND status='accepted' AND per_icp_score=0) <> 10
       OR (SELECT COUNT(*) FROM public.lab_arena_runs
           WHERE round_id=v_archive_id AND stage=2 AND kind='execute'
             AND status='pending') <> 130
       OR v_archive.configuration_doc->>'recovery_archive_runs_sha256'
            IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(
              pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)::TEXT,
              'sha256'),'hex') FROM public.lab_arena_runs r
              WHERE round_id=v_archive_id)
       OR v_archive.configuration_doc->>'recovery_archive_submissions_sha256'
            IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(
              pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)::TEXT,
              'sha256'),'hex') FROM public.lab_arena_submissions s
              WHERE round_id=v_archive_id)
       OR EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events
           WHERE round_id=v_archive_id)
       OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
           WHERE round_id=v_archive_id) THEN
      RAISE EXCEPTION 'Oct01 deadline recovery archive replay differs'
        USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;

  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF NOT FOUND OR NOT v_control.operator_paused
     OR v_control.pause_reason <> 'oct01_deepline_outage'
     OR v_control.actor_ref <> 'oct01-provider-recovery369'
     OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> ''
     OR v_control.guard_expires_at IS NOT NULL
     OR v_control.restart_phase <> ''
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'Oct01 deadline recovery does not own claim hold'
      USING ERRCODE='55000';
  END IF;
  IF v_round.round_id IS NULL OR v_round.status <> 'stage2'
     OR v_round.status_generation <> 9 OR v_round.stage_generation <> 7
     OR v_round.configuration_doc -> 'schedule' IS DISTINCT FROM v_old_schedule
     OR v_round.configuration_doc ->> 'execution_sequence_policy' <>
          'baseline_scored_first_v1'
     OR v_round.configuration_doc ->> 'mode' <> 'live'
     OR v_round.configuration_doc ->> 'scorer_image_digest' <>
          'sha256:f8ab912f739a1c9e30cc33fb4a5f4ea86b7ef4680af571dc203574f6473f13eb'
     OR v_round.benchmark_ref <> 'arena/arena-2026-10-01/benchmark.json'
     OR v_round.evaluation_date <> '2026-10-01'
     OR v_round.icp_set_date <> DATE '2026-09-30'
     OR v_round.stage1_scoring_plan_doc IS NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage1_scoring_plan_doc->>'round_id' IS DISTINCT FROM v_round_id
     OR v_round.stage1_scoring_plan_doc->>'stage' IS DISTINCT FROM '1'
     OR v_round.stage1_scoring_plan_doc->>'execution_sequence_policy' IS DISTINCT FROM
          'baseline_scored_first_v1'
     OR v_round.stage1_scoring_plan_doc->'zero_rows' IS DISTINCT FROM '[]'::JSONB
     OR pg_catalog.jsonb_array_length(v_round.stage1_scoring_plan_doc->'work_items') <> 10
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(
         v_round.stage1_scoring_plan_doc->'work_items') AS item
       WHERE NOT EXISTS (SELECT 1 FROM public.lab_arena_runs e
         WHERE e.run_id=item->>'scored_run_id'
           AND e.round_id=v_round_id AND e.kind='execute' AND e.stage=1
           AND e.status='accepted' AND e.output_ref=item->>'output_ref'
           AND e.submission_id=item->>'submission_id'
           AND e.icp_position=(item->>'icp_position')::INTEGER))
     OR pg_catalog.jsonb_array_length(v_round.finalists) <> 13
     OR v_round.publication_doc IS NOT NULL OR v_round.published_at IS NOT NULL
     OR v_round.king_outcome IS NOT NULL OR v_round.reward_activated_at IS NOT NULL
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
           WHERE round_id=v_round_id) IS DISTINCT FROM
        '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
             'run_id',run_id,'submission_id',submission_id,'stage',stage,
             'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,
             'output_ref',output_ref,'stage_generation',stage_generation)
             ORDER BY icp_position)::TEXT,'sha256'),'hex')
           FROM public.lab_arena_runs WHERE round_id=v_round_id
             AND kind='execute' AND stage=1) IS DISTINCT FROM
        '24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa'
  THEN
    RAISE EXCEPTION 'Oct01 deadline recovery round or frozen source differs'
      USING ERRCODE='55000';
  END IF;
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds r
      WHERE r.round_id <> v_round_id
        AND r.configuration_doc->>'mode'='live'
        AND r.status IN ('committed','stage1','stage1_closed','stage1_scoring',
                         'stage1_judged','stage1_scored','stage2','stage2_closed',
                         'stage2_scoring','stage2_judged','scored'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
      WHERE r.round_id <> v_round_id AND r.status IN ('leased','submitted')) THEN
    RAISE EXCEPTION 'Oct01 deadline recovery overlaps active other round'
      USING ERRCODE='55000';
  END IF;
  IF (SELECT COUNT(*) FROM public.lab_arena_submissions
      WHERE round_id=v_round_id) <> 14
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
      WHERE round_id=v_round_id) <> 152
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND kind='execute' AND stage=1
        AND submission_id='baseline-2026-10-01' AND status='accepted'
        AND output_ref IS NOT NULL AND per_icp_score=0) <> 10
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND kind='execute' AND stage=1
        AND (submission_id<>'baseline-2026-10-01' OR status<>'accepted'))
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND kind='score' AND stage=1
        AND submission_id='baseline-2026-10-01') <> 12
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND kind='score' AND stage<>1)
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND kind='score' AND status='accepted'
        AND icp_position=1 AND output_ref IS NOT NULL
        AND terminal_cause='accepted' AND result_doc IS NOT NULL
        AND per_icp_score IS NULL) <> 1
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND kind='score' AND status='failed') <> 11
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND kind='execute' AND stage=2
        AND status='pending' AND stage_generation=7) <> 130
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
      WHERE r.round_id=v_round_id AND r.stage=2 AND (
        r.kind<>'execute' OR r.status<>'pending' OR r.attempt<>1
        OR r.stage_generation<>7 OR r.lease_generation<>0
        OR r.runner_hotkey IS NOT NULL OR r.lease_token_hash IS NOT NULL
        OR r.lease_expires_at IS NOT NULL OR r.claim_request_id IS NOT NULL
        OR r.claim_request_hash IS NOT NULL OR r.claim_response IS NOT NULL
        OR r.result_doc IS NOT NULL OR r.output_ref IS NOT NULL
        OR r.terminal_cause IS NOT NULL OR r.terminal_doc IS NOT NULL
        OR r.scored_run_id IS NOT NULL OR r.previous_runner_hotkey IS NOT NULL
        OR r.per_icp_score IS NOT NULL OR r.run_id<>r.assignment_id||':1'
      ))
     OR EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events e
      WHERE e.run_id IN (SELECT run_id FROM public.lab_arena_runs
        WHERE round_id=v_round_id AND stage=2))
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger l
      WHERE l.run_id IN (SELECT run_id FROM public.lab_arena_runs
        WHERE round_id=v_round_id AND stage=2))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
      WHERE round_id=v_archive_id OR run_id LIKE '%:r372archive'
        OR assignment_id LIKE '%:r372archive')
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
      WHERE round_id=v_archive_id OR submission_id LIKE '%:r372archive')
  THEN
    RAISE EXCEPTION 'Oct01 deadline recovery runs or archive identities differ'
      USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.encode(extensions.digest(
        pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(
          run_id,assignment_id,icp_position,attempt,status,terminal_cause,
          stage_generation,scored_run_id) ORDER BY run_id)::TEXT,'sha256'),'hex')
      FROM public.lab_arena_runs WHERE round_id=v_round_id AND kind='score')
        IS DISTINCT FROM
        'dad688e5ab73c5f5e6671299f10d914d5462a5b8333d24b9ddb53b467bf58a63'
     OR (SELECT pg_catalog.encode(extensions.digest(
        pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(
          run_id,assignment_id,submission_id,icp_position,attempt,status,
          stage_generation) ORDER BY run_id)::TEXT,'sha256'),'hex')
      FROM public.lab_arena_runs WHERE round_id=v_round_id
        AND kind='execute' AND stage=2) IS DISTINCT FROM
        '2b0443f2ba2cf021628d34bebd575886be427c4749536b2c8879d3b9bc10239a'
  THEN
    RAISE EXCEPTION 'Oct01 deadline recovery score or miner preimage differs'
      USING ERRCODE='55000';
  END IF;
  IF pg_catalog.clock_timestamp() >=
      (v_new_schedule->>'stage_1_scoring_close')::TIMESTAMPTZ THEN
    RAISE EXCEPTION 'Oct01 deadline recovery scoring window already closed'
      USING ERRCODE='55000';
  END IF;

  SELECT pg_catalog.jsonb_agg(
    p || pg_catalog.jsonb_build_object(
      'submission_id',p->>'submission_id'||v_suffix) ORDER BY ordinal)
    INTO v_archive_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants)
       WITH ORDINALITY AS participant(p,ordinal);
  v_archive_config := v_round.configuration_doc || pg_catalog.jsonb_build_object(
    'round_id',v_archive_id,'mode','shadow','rewards_enabled',FALSE,
    'recovery_source_round_id',v_round_id,
    'recovery_preimage_round_sha256',
      pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(v_round)::TEXT,'sha256'),'hex'),
    'recovery_preimage_score_rows_sha256',
      'dad688e5ab73c5f5e6671299f10d914d5462a5b8333d24b9ddb53b467bf58a63',
    'recovery_preimage_stage2_rows_sha256',
      '2b0443f2ba2cf021628d34bebd575886be427c4749536b2c8879d3b9bc10239a');
  v_new_config := pg_catalog.jsonb_set(
    v_round.configuration_doc,'{schedule}',v_new_schedule,TRUE);

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;

  INSERT INTO public.lab_arena_rounds (
    round_id,status,status_generation,stage_generation,configuration_doc,
    rewards_enabled,participants,benchmark_ref,evaluation_date,
    stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
    king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
    reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
    cancel_reason,published_at,created_at,updated_at,promotion_required,
    promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
    confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
    champion_funding_frozen,champion_submission_id,champion_hotkey,
    champion_fallback_providers
  )
  SELECT round_id,status,status_generation,stage_generation,configuration_doc,
    rewards_enabled,participants,benchmark_ref,evaluation_date,
    stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
    king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
    reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
    cancel_reason,published_at,created_at,updated_at,promotion_required,
    promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
    confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
    champion_funding_frozen,champion_submission_id,champion_hotkey,
    champion_fallback_providers
  FROM pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_rounds,
    pg_catalog.to_jsonb(v_round) || pg_catalog.jsonb_build_object(
      'round_id',v_archive_id,'configuration_doc',v_archive_config,
      'participants',v_archive_participants,'status','cancelled',
      'rewards_enabled',FALSE,
      'cancel_reason','authorized_oct01_recovery372_archive'
    ));
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(s) || pg_catalog.jsonb_build_object(
      'round_id',v_archive_id,
      'submission_id',s.submission_id||v_suffix
    ))).*
  FROM public.lab_arena_submissions s WHERE s.round_id=v_round_id;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>14 THEN RAISE EXCEPTION 'Oct01 archive submissions incomplete'; END IF;

  INSERT INTO public.lab_arena_runs
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_runs,
    pg_catalog.to_jsonb(r) || pg_catalog.jsonb_build_object(
      'run_id',r.run_id||v_suffix,'assignment_id',r.assignment_id||v_suffix,
      'round_id',v_archive_id,'submission_id',r.submission_id||v_suffix
    ))).*
  FROM public.lab_arena_runs r
  WHERE r.round_id=v_round_id AND r.stage=1 AND r.kind='execute';
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>10 THEN RAISE EXCEPTION 'Oct01 archive derived scores incomplete'; END IF;

  UPDATE public.lab_arena_runs r
  SET run_id=r.run_id||v_suffix,assignment_id=r.assignment_id||v_suffix,
      round_id=v_archive_id,submission_id=r.submission_id||v_suffix
  WHERE r.round_id=v_round_id AND r.stage=2 AND r.kind='execute';
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>130 THEN RAISE EXCEPTION 'Oct01 archive pending miners incomplete'; END IF;

  UPDATE public.lab_arena_rounds a
  SET configuration_doc=a.configuration_doc || pg_catalog.jsonb_build_object(
    'recovery_archive_runs_sha256',
      (SELECT pg_catalog.encode(extensions.digest(
        pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)::TEXT,
        'sha256'),'hex') FROM public.lab_arena_runs r
        WHERE round_id=v_archive_id),
    'recovery_archive_submissions_sha256',
      (SELECT pg_catalog.encode(extensions.digest(
        pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)::TEXT,
        'sha256'),'hex') FROM public.lab_arena_submissions s
        WHERE round_id=v_archive_id))
  WHERE a.round_id=v_archive_id;

  UPDATE public.lab_arena_runs
  SET per_icp_score=NULL
  WHERE round_id=v_round_id AND kind='execute' AND stage=1
    AND submission_id='baseline-2026-10-01';
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>10 THEN RAISE EXCEPTION 'Oct01 invalid derived scores not cleared'; END IF;

  UPDATE public.lab_arena_rounds
  SET status='stage1_scoring',status_generation=status_generation+1,
      stage_generation=stage_generation+1,configuration_doc=v_new_config,
      finalists=NULL,updated_at=pg_catalog.clock_timestamp()
  WHERE round_id=v_round_id;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>1 THEN RAISE EXCEPTION 'Oct01 scoring round not restored'; END IF;

  FOR v_target IN
    SELECT DISTINCT ON (assignment_id) * FROM public.lab_arena_runs
    WHERE round_id=v_round_id AND kind='score'
    ORDER BY assignment_id,attempt DESC
  LOOP
    IF v_target.status='accepted' THEN CONTINUE; END IF;
    IF v_target.status<>'failed' OR v_target.attempt>=5
       OR v_target.output_ref IS NOT NULL OR v_target.per_icp_score IS NOT NULL
       OR v_target.terminal_cause NOT IN
            ('credential_error','judge_error','stage_closed')
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs e
           WHERE e.run_id=v_target.scored_run_id AND e.kind='execute'
             AND e.round_id=v_round_id AND e.status='accepted'
             AND e.output_ref IS NOT NULL
             AND e.submission_id=v_target.submission_id
             AND e.icp_position=v_target.icp_position)
    THEN
      RAISE EXCEPTION 'Oct01 failed judge cannot be retried: %',v_target.run_id;
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
      v_round.stage_generation+1,v_target.kind,v_target.scored_run_id,
      v_target.runner_hotkey,v_target.judgment_cache_key,
      v_target.judgment_input_hash,v_target.judgment_scope_doc,
      v_target.judgment_group_leader,v_target.judgment_group_miner_hotkeys,
      v_target.company_judgment_refs
    );
    v_retried:=v_retried+1;
  END LOOP;
  IF v_retried<>9 THEN RAISE EXCEPTION 'Oct01 judge retry count differs'; END IF;

  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  IF EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
      WHERE t.tgrelid IN ('public.lab_arena_rounds'::REGCLASS,
                         'public.lab_arena_submissions'::REGCLASS,
                         'public.lab_arena_runs'::REGCLASS)
        AND NOT t.tgisinternal AND t.tgenabled IS DISTINCT FROM 'O')
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND kind='execute' AND stage=2) <> 0
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND kind='execute' AND stage=1
           AND per_icp_score IS NULL) <> 10
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND kind='score' AND status='accepted') <> 1
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND kind='score' AND status='pending'
           AND stage_generation=8) <> 9
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_archive_id AND stage=2 AND kind='execute') <> 130
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_archive_id AND stage=1 AND kind='execute'
           AND per_icp_score=0) <> 10
     OR (SELECT status FROM public.lab_arena_rounds
         WHERE round_id=v_round_id) <> 'stage1_scoring'
     OR (SELECT operator_paused FROM public.lab_arena_restart_claim_control
         WHERE singleton) IS DISTINCT FROM TRUE
  THEN
    RAISE EXCEPTION 'Oct01 deadline recovery postcondition differs'
      USING ERRCODE='55000';
  END IF;
END;
$oct01_recover$;
COMMIT;
