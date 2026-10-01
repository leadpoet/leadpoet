-- Recover the exact October 1 baseline execution after the frozen image could
-- not be materialized. All twenty attempts failed before model execution.
-- The immutable benchmark and all fourteen frozen source archives stay bound
-- to the same live round. Failed attempts and their sixty trajectory events
-- remain in a cancelled audit round; no provider ledger entry is moved.
--
-- Evidence: production-initial.json and trajectories-initial.json captured
-- after cancellation; benchmark.json SHA-256:
-- 55c2250f43a04b534d969789f7084446d2ddb93a447e95cdcba177a7729169c8.
-- This is a one-round transition. It is not a general cancelled-round reset.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control', 0)
);
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_trajectory_events IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.qualification_private_icp_sets IN SHARE MODE;

DO $recover_oct01_baseline$
DECLARE
  v_round public.lab_arena_rounds;
  v_archive public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_baseline public.lab_arena_submissions;
  v_archive_config JSONB;
  v_archive_participants JSONB;
  v_new_config JSONB;
  v_bank JSONB;
  v_count BIGINT;
  v_position INTEGER;
  v_assignment TEXT;
  v_round_id CONSTANT TEXT := 'arena-2026-10-01';
  v_archive_id CONSTANT TEXT := 'arena-2026-10-01-r368archive';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-10-01';
  v_suffix CONSTANT TEXT := ':r368archive';
  v_old_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-01T20:30:02Z","publication_deadline":"2026-10-01T20:30:03Z","stage_1_close":"2026-10-01T04:30:01Z","stage_1_scoring_close":"2026-10-01T11:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-01T14:00:02Z","stage_2_start":"2026-10-01T11:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
  v_new_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-01T23:00:02Z","publication_deadline":"2026-10-01T23:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-01T14:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-01T17:00:02Z","stage_2_start":"2026-10-01T14:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Oct01 recovery round missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
  WHERE round_id = v_archive_id FOR UPDATE;

  -- Reapplication is a read-only check. Normal progression may have added
  -- retries, score rows, or a publication, so only the frozen controls and
  -- immutable archive are compared after the first application.
  IF v_archive.round_id IS NOT NULL THEN
    IF v_round.status NOT IN (
         'stage1','stage1_closed','stage1_scoring','stage1_judged',
         'stage1_scored','stage2','stage2_closed','stage2_scoring',
         'stage2_judged','scored','published','cancelled'
       )
       OR v_round.configuration_doc -> 'schedule' IS DISTINCT FROM v_new_schedule
       OR pg_catalog.encode(extensions.digest(
            v_round.configuration_doc::TEXT, 'sha256'), 'hex')
          IS DISTINCT FROM
          'e7cd2d1d2fea2a1f2ae41b313889545d36ca3e9b09e5d3d4021fec36a99b0a7b'
       OR pg_catalog.encode(extensions.digest(
            v_round.participants::TEXT, 'sha256'), 'hex')
          IS DISTINCT FROM
          '872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19'
       OR pg_catalog.encode(extensions.digest(
            pg_catalog.to_jsonb(v_archive)::TEXT, 'sha256'), 'hex')
          IS DISTINCT FROM
          '05b787490ac66170bb64dc1292a8e371f4bbcd3fe9ed7f028cfee02d3143ea5c'
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
           WHERE round_id = v_archive_id) <> 14
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id = v_archive_id) <> 20
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_trajectory_events
           WHERE round_id = v_archive_id) <> 60
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
           WHERE round_id = v_archive_id) <> 0
       OR (SELECT pg_catalog.encode(extensions.digest(
             pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s)
               ORDER BY s.submission_id)::TEXT, 'sha256'), 'hex')
           FROM public.lab_arena_submissions s
           WHERE s.round_id = v_archive_id) IS DISTINCT FROM
          'd9282fda079e8669cf67643c574cc634091423a30f1516125591868c78bcbe15'
       OR (SELECT pg_catalog.encode(extensions.digest(
             pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)
               ORDER BY r.run_id)::TEXT, 'sha256'), 'hex')
           FROM public.lab_arena_runs r
           WHERE r.round_id = v_archive_id) IS DISTINCT FROM
          'a678e3f3da6cc58b66b494e848e7d0d2c265e820c5209ff7ea273107802a871f'
       OR (SELECT pg_catalog.encode(extensions.digest(
             pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e)
               ORDER BY e.trajectory_id)::TEXT, 'sha256'), 'hex')
           FROM public.lab_arena_trajectory_events e
           WHERE e.round_id = v_archive_id) IS DISTINCT FROM
          '903d4594445baf8290d9b88b04a09ae02dce2f18afc98c5d45f86df3ff6543bc'
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND submission_id = v_baseline_id
             AND stage = 1 AND kind = 'execute' AND attempt = 1
             AND stage_generation = 3
             AND run_id = assignment_id || ':1'
             AND assignment_id LIKE
               'arena-2026-10-01:baseline-2026-10-01:1:%:r368') <> 10
    THEN
      RAISE EXCEPTION 'Oct01 recovery replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;

  -- These digests cover every column, including frozen scorer image/policy,
  -- all fourteen source identities, failed terminal documents, and private
  -- trajectory contents. Any extra row changes the ordered digest or count.
  IF v_round.status IS DISTINCT FROM 'cancelled'
     OR v_round.status_generation IS DISTINCT FROM 3
     OR v_round.stage_generation IS DISTINCT FROM 2
     OR v_round.cancel_reason IS DISTINCT FROM 'execution_incomplete:stage1:10'
     OR v_round.configuration_doc -> 'schedule' IS DISTINCT FROM v_old_schedule
     OR v_round.configuration_doc ->> 'round_id' IS DISTINCT FROM v_round_id
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc ->> 'execution_sequence_policy'
        IS DISTINCT FROM 'baseline_scored_first_v1'
     OR v_round.configuration_doc #>> '{scorer_policy,env_bindings,ARENA_SCORE_NORMALIZATION}'
        IS DISTINCT FROM 'available_intent_cap_v1'
     OR v_round.benchmark_ref IS DISTINCT FROM
        'arena/arena-2026-10-01/benchmark.json'
     OR v_round.icp_set_date IS DISTINCT FROM DATE '2026-09-30'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-10-01'
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 14
     OR pg_catalog.encode(extensions.digest(
          pg_catalog.to_jsonb(v_round)::TEXT, 'sha256'), 'hex')
        IS DISTINCT FROM
        'dbbaf18ff42f77aa56c8dc6f91780a8887195763e43827f3de93f57f80dd3623'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = v_round_id) <> 14
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s)
             ORDER BY s.submission_id)::TEXT, 'sha256'), 'hex')
         FROM public.lab_arena_submissions s
         WHERE s.round_id = v_round_id) IS DISTINCT FROM
        '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id) <> 20
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)
             ORDER BY r.run_id)::TEXT, 'sha256'), 'hex')
         FROM public.lab_arena_runs r
         WHERE r.round_id = v_round_id) IS DISTINCT FROM
        'a28d65a9d8e350b081473855e5d239f5d27bc92c8fe9afc083288a3a7f7ac2d8'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_trajectory_events
         WHERE round_id = v_round_id) <> 60
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e)
             ORDER BY e.trajectory_id)::TEXT, 'sha256'), 'hex')
         FROM public.lab_arena_trajectory_events e
         WHERE e.round_id = v_round_id) IS DISTINCT FROM
        'de57bf99060cfe8cf5170436a5a25ac9fa3bbf71d66dfb3e8a167116f6cc1d0b'
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
         WHERE round_id = v_round_id OR submission_id IN (
           SELECT submission_id FROM public.lab_arena_submissions
           WHERE round_id = v_round_id
         ) OR run_id IN (
           SELECT run_id FROM public.lab_arena_runs
           WHERE round_id = v_round_id
         ))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE submission_id = v_baseline_id AND round_id <> v_round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events
         WHERE run_id IN (SELECT run_id FROM public.lab_arena_runs
                          WHERE round_id = v_round_id)
           AND round_id <> v_round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
         WHERE submission_id LIKE '%:r368archive')
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE assignment_id LIKE '%:r368')
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
         WHERE t.tgrelid IN (
           'public.lab_arena_rounds'::REGCLASS,
           'public.lab_arena_submissions'::REGCLASS,
           'public.lab_arena_runs'::REGCLASS,
           'public.lab_arena_ledger'::REGCLASS,
           'public.lab_arena_trajectory_events'::REGCLASS
         ) AND NOT t.tgisinternal AND t.tgenabled IS DISTINCT FROM 'O')
  THEN
    RAISE EXCEPTION 'Oct01 recovery terminal preimage differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT s.* INTO v_baseline FROM public.lab_arena_submissions s
  WHERE s.round_id = v_round_id AND s.submission_id = v_baseline_id;
  IF NOT FOUND OR v_baseline.status IS DISTINCT FROM 'frozen'
     OR v_baseline.is_king IS NOT TRUE THEN
    RAISE EXCEPTION 'Oct01 frozen baseline differs' USING ERRCODE = '55000';
  END IF;
  SELECT icps INTO v_bank FROM public.qualification_private_icp_sets
  WHERE set_id = 20260930;
  IF NOT FOUND OR pg_catalog.jsonb_array_length(v_bank) <> 10
     OR pg_catalog.encode(extensions.digest(v_bank::TEXT, 'sha256'), 'hex')
        IS DISTINCT FROM
        '8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61'
  THEN
    RAISE EXCEPTION 'Oct01 frozen ICP bank differs' USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.clock_timestamp() + INTERVAL '150 minutes'
       >= (v_new_schedule ->> 'stage_1_close')::TIMESTAMPTZ THEN
    RAISE EXCEPTION 'Oct01 recovery stage1 retry window closed'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.jsonb_agg(
    p || pg_catalog.jsonb_build_object(
      'submission_id', p ->> 'submission_id' || v_suffix
    ) ORDER BY ordinal
  ) INTO v_archive_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants)
       WITH ORDINALITY AS participant(p, ordinal);
  v_archive_config := v_round.configuration_doc ||
    pg_catalog.jsonb_build_object(
      'round_id', v_archive_id,
      'mode', 'shadow',
      'rewards_enabled', FALSE,
      'recovery_source_round_id', v_round_id,
      'recovery_preimage_sha256',
        'sha256:dbbaf18ff42f77aa56c8dc6f91780a8887195763e43827f3de93f57f80dd3623'
    );
  v_new_config := pg_catalog.jsonb_set(
    v_round.configuration_doc, '{schedule}', v_new_schedule, TRUE
  );

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
      'round_id', v_archive_id,
      'configuration_doc', v_archive_config,
      'participants', v_archive_participants,
      'rewards_enabled', FALSE,
      'cancel_reason', 'authorized_oct01_recovery368_archive'
    )
  );
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(s) || pg_catalog.jsonb_build_object(
      'round_id', v_archive_id,
      'submission_id', s.submission_id || v_suffix
    )
  )).*
  FROM public.lab_arena_submissions s WHERE s.round_id = v_round_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 14 THEN
    RAISE EXCEPTION 'Oct01 archive submissions incomplete';
  END IF;

  UPDATE public.lab_arena_runs
  SET round_id = v_archive_id,
      submission_id = v_baseline_id || v_suffix
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 20 THEN
    RAISE EXCEPTION 'Oct01 archive runs incomplete';
  END IF;
  UPDATE public.lab_arena_trajectory_events
  SET round_id = v_archive_id,
      submission_id = v_baseline_id || v_suffix
  WHERE round_id = v_round_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 60 THEN
    RAISE EXCEPTION 'Oct01 archive trajectories incomplete';
  END IF;

  UPDATE public.lab_arena_rounds
  SET status = 'stage1',
      status_generation = status_generation + 1,
      stage_generation = stage_generation + 1,
      configuration_doc = v_new_config,
      cancel_reason = NULL,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round_id;

  FOR v_position IN 0..9 LOOP
    v_assignment := v_round_id || ':' || v_baseline_id || ':1:'
      || v_position::TEXT || ':r368';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, kind, status, stage_generation
    ) VALUES (
      v_assignment || ':1', v_assignment, v_round_id, v_baseline_id,
      v_baseline.miner_hotkey, 1, v_position, 1, 'execute', 'pending',
      v_round.stage_generation + 1
    );
  END LOOP;

  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  SELECT * INTO v_after FROM public.lab_arena_rounds
  WHERE round_id = v_round_id;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
  WHERE round_id = v_archive_id;
  IF v_after.status IS DISTINCT FROM 'stage1'
     OR v_after.status_generation IS DISTINCT FROM 4
     OR v_after.stage_generation IS DISTINCT FROM 3
     OR v_after.cancel_reason IS NOT NULL
     OR v_after.configuration_doc IS DISTINCT FROM v_new_config
     OR (pg_catalog.to_jsonb(v_after) - 'status' - 'status_generation'
         - 'stage_generation' - 'configuration_doc' - 'cancel_reason'
         - 'updated_at') IS DISTINCT FROM
        (pg_catalog.to_jsonb(v_round) - 'status' - 'status_generation'
         - 'stage_generation' - 'configuration_doc' - 'cancel_reason'
         - 'updated_at')
     OR pg_catalog.encode(extensions.digest(
          pg_catalog.to_jsonb(v_archive)::TEXT, 'sha256'), 'hex')
        IS DISTINCT FROM
        '05b787490ac66170bb64dc1292a8e371f4bbcd3fe9ed7f028cfee02d3143ea5c'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s)
             ORDER BY s.submission_id)::TEXT, 'sha256'), 'hex')
         FROM public.lab_arena_submissions s
         WHERE s.round_id = v_round_id) IS DISTINCT FROM
        '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s)
             ORDER BY s.submission_id)::TEXT, 'sha256'), 'hex')
         FROM public.lab_arena_submissions s
         WHERE s.round_id = v_archive_id) IS DISTINCT FROM
        'd9282fda079e8669cf67643c574cc634091423a30f1516125591868c78bcbe15'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)
             ORDER BY r.run_id)::TEXT, 'sha256'), 'hex')
         FROM public.lab_arena_runs r
         WHERE r.round_id = v_archive_id) IS DISTINCT FROM
        'a678e3f3da6cc58b66b494e848e7d0d2c265e820c5209ff7ea273107802a871f'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e)
             ORDER BY e.trajectory_id)::TEXT, 'sha256'), 'hex')
         FROM public.lab_arena_trajectory_events e
         WHERE e.round_id = v_archive_id) IS DISTINCT FROM
        '903d4594445baf8290d9b88b04a09ae02dce2f18afc98c5d45f86df3ff6543bc'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND kind = 'execute'
           AND status = 'pending' AND stage = 1
           AND stage_generation = 3) <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id) <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_trajectory_events
         WHERE round_id = v_round_id) <> 0
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
         WHERE t.tgrelid IN (
           'public.lab_arena_rounds'::REGCLASS,
           'public.lab_arena_submissions'::REGCLASS,
           'public.lab_arena_runs'::REGCLASS
         ) AND NOT t.tgisinternal AND t.tgenabled IS DISTINCT FROM 'O')
  THEN
    RAISE EXCEPTION 'Oct01 recovery postcondition differs' USING ERRCODE = '55000';
  END IF;
END;
$recover_oct01_baseline$;
COMMIT;
