-- Rejudge the accepted October 1 baseline outputs under the new scorer.
-- Preserve every old score attempt, cost and trajectory in a cancelled audit
-- round. Miner executions have not started; normal stage 2 runs them once.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $oct01_scoring_namespace$
DECLARE
  v_definition TEXT;
  v_hash TEXT;
  v_anchor CONSTANT TEXT :=
    '    v_status := CASE WHEN v_cache.cache_key IS NULL THEN ''pending'' ELSE ''accepted'' END;';
  v_insert CONSTANT TEXT :=
    '    -- Oct01 rerun376 score namespace' || E'\n' ||
    '    IF p_round_id = ''arena-2026-10-01'' AND EXISTS (' || E'\n' ||
    '      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = ''arena-2026-10-01-r376archive'') THEN' || E'\n' ||
    '      v_assignment := p_round_id || '':'' || v_scored.submission_id || '':'' || p_stage::TEXT || '':'' || v_scored.icp_position::TEXT || '':score:rerun376'';' || E'\n' ||
    '    END IF;' || E'\n';
BEGIN
  v_definition := pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::REGPROCEDURE);
  v_hash := pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
  IF pg_catalog.strpos(v_definition,'Oct01 rerun376 score namespace') > 0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
        pg_catalog.replace(v_definition,v_insert,''))) <> pg_catalog.length(v_insert)
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds
           WHERE round_id='arena-2026-10-01-r376archive') THEN
      RAISE EXCEPTION 'Oct01 scoring namespace replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF v_hash <> '20fb97257eb151be8352c2104aa2f1fa75d4577faab37bdd2ef130e682cc7216'
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_anchor,''))) <> pg_catalog.length(v_anchor)
  THEN
    RAISE EXCEPTION 'Oct01 scoring initializer differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_anchor,v_insert || v_anchor);
END;
$oct01_scoring_namespace$;

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_trajectory_events IN ACCESS EXCLUSIVE MODE;

DO $oct01_rejudge$
DECLARE
  v_round public.lab_arena_rounds;
  v_archive public.lab_arena_rounds;
  v_archive_config JSONB;
  v_archive_participants JSONB;
  v_score_hash TEXT;
  v_ledger_hash TEXT;
  v_ledger_max BIGINT;
  v_events_hash TEXT;
  v_count INTEGER;
  v_empty CONSTANT JSONB := '{"companies":[]}'::JSONB;
  v_round_id CONSTANT TEXT := 'arena-2026-10-01';
  v_archive_id CONSTANT TEXT := 'arena-2026-10-01-r376archive';
  v_suffix CONSTANT TEXT := ':r376archive';
  v_archive_baseline_id CONSTANT TEXT := 'baseline-2026-10-01-r376archive';
  v_new_digest CONSTANT TEXT :=
    'sha256:342645a42b52363cb907c7b48627ead707e09547a7197b4b0d803e7e6e57a7ba';
  v_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-03T08:00:02Z","publication_deadline":"2026-10-03T08:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-02T14:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-03T02:00:02Z","stage_2_start":"2026-10-02T14:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id=v_round_id FOR UPDATE;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
  WHERE round_id=v_archive_id FOR UPDATE;
  IF v_archive.round_id IS NOT NULL THEN
    IF v_archive.status <> 'cancelled'
       OR v_archive.rewards_enabled IS DISTINCT FROM FALSE
       OR v_archive.configuration_doc->>'mode' IS DISTINCT FROM 'live'
       OR v_archive.king_hotkey IS NOT NULL
       OR v_archive.effective_reward_epoch IS NOT NULL
       OR v_archive.promotion_required IS DISTINCT FROM FALSE
       OR v_archive.champion_funding_frozen IS DISTINCT FROM FALSE
       OR v_archive.champion_hotkey IS NOT NULL
       OR v_archive.cancel_reason IS DISTINCT FROM
         'authorized_oct01_new_scorer_rejudge_archive'
       OR v_archive.configuration_doc->>'recovery_source_round_id'
         IS DISTINCT FROM v_round_id
       OR v_archive.configuration_doc->>'recovery_score_runs_sha256'
         IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ||
             pg_catalog.jsonb_build_object('round_id',v_round_id,
               'submission_id','baseline-2026-10-01') ORDER BY run_id)::TEXT,
           'sha256'),'hex') FROM public.lab_arena_runs s
           WHERE round_id=v_archive_id AND kind='score')
       OR v_archive.configuration_doc->>'recovery_score_ledger_sha256'
         IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(COALESCE(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ||
             pg_catalog.jsonb_build_object('round_id',v_round_id,
               'submission_id','baseline-2026-10-01') ORDER BY entry_id)::TEXT,''),
           'sha256'),'hex') FROM public.lab_arena_ledger l
           WHERE round_id=v_archive_id AND EXISTS
             (SELECT 1 FROM public.lab_arena_runs s
              WHERE s.round_id=v_archive_id AND s.kind='score'
                AND s.run_id=l.run_id)
             AND l.entry_id <=
               (v_archive.configuration_doc->>'recovery_score_ledger_max_entry_id')::BIGINT)
       OR v_archive.configuration_doc->>'recovery_score_events_sha256'
         IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(COALESCE(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e) ||
             pg_catalog.jsonb_build_object('round_id',v_round_id,
               'submission_id','baseline-2026-10-01') ORDER BY trajectory_id)::TEXT,''),
           'sha256'),'hex') FROM public.lab_arena_trajectory_events e
           WHERE round_id=v_archive_id AND run_kind='score')
       OR v_round.configuration_doc->>'scorer_image_digest'
         IS DISTINCT FROM v_new_digest
       OR v_round.configuration_doc->'schedule' IS DISTINCT FROM v_schedule
       OR (SELECT COUNT(*) FROM public.lab_arena_runs
           WHERE round_id=v_archive_id AND kind='score') <> 21
       OR (SELECT COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs
           WHERE round_id=v_round_id AND stage=1 AND kind='score'
             AND assignment_id LIKE '%:score:rerun376') NOT BETWEEN 0 AND 10
    THEN
      RAISE EXCEPTION 'Oct01 rejudge archive replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF v_round.round_id IS NULL OR v_round.status <> 'stage1_judged'
     OR v_round.status_generation <> 11 OR v_round.stage_generation <> 9
     OR v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
       'sha256:f8ab912f739a1c9e30cc33fb4a5f4ea86b7ef4680af571dc203574f6473f13eb'
     OR v_round.configuration_doc->'schedule'->>'stage_1_scoring_close'
        IS DISTINCT FROM
       '2026-10-01T20:00:01Z'
     OR v_round.benchmark_ref IS NULL OR v_round.evaluation_date IS NULL
     OR v_round.published_at IS NOT NULL OR v_round.reward_activated_at IS NOT NULL
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='execute'
           AND submission_id='baseline-2026-10-01' AND status='accepted'
           AND per_icp_score IS NULL AND qualification_doc=v_empty) <> 10
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='score') <> 21
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='score'
           AND status='accepted') <> 10
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='score'
           AND status='failed') <> 11
     OR (SELECT COUNT(DISTINCT icp_position) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND kind='score' AND status='accepted') <> 10
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs s
         LEFT JOIN public.lab_arena_runs e ON e.run_id=s.scored_run_id
         WHERE s.round_id=v_round_id AND s.kind='score'
           AND (e.run_id IS NULL OR e.round_id<>v_round_id
             OR e.kind<>'execute' OR e.status<>'accepted'
             OR e.icp_position<>s.icp_position
             OR e.submission_id<>s.submission_id))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=2)
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id='arena-2026-10-01-r372archive'
           AND kind='execute' AND stage=1 AND per_icp_score=0) <> 10
  THEN
    RAISE EXCEPTION 'Oct01 rejudge precondition differs' USING ERRCODE='55000';
  END IF;

  SELECT pg_catalog.encode(extensions.digest(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY run_id)::TEXT,
    'sha256'),'hex') INTO v_score_hash
  FROM public.lab_arena_runs s WHERE round_id=v_round_id AND kind='score';
  SELECT pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)::TEXT,''),
    'sha256'),'hex') INTO v_ledger_hash
  FROM public.lab_arena_ledger l WHERE round_id=v_round_id
    AND EXISTS (SELECT 1 FROM public.lab_arena_runs s
      WHERE s.round_id=v_round_id AND s.kind='score' AND s.run_id=l.run_id);
  SELECT pg_catalog.max(l.entry_id) INTO v_ledger_max
  FROM public.lab_arena_ledger l WHERE l.round_id=v_round_id
    AND EXISTS (SELECT 1 FROM public.lab_arena_runs s
      WHERE s.round_id=v_round_id AND s.kind='score' AND s.run_id=l.run_id);
  SELECT pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e) ORDER BY trajectory_id)::TEXT,''),
    'sha256'),'hex') INTO v_events_hash
  FROM public.lab_arena_trajectory_events e WHERE round_id=v_round_id
    AND e.run_kind='score';

  SELECT pg_catalog.jsonb_agg(item.value || pg_catalog.jsonb_build_object(
    'submission_id',CASE WHEN item.value->>'submission_id'='baseline-2026-10-01'
      THEN v_archive_baseline_id ELSE (item.value->>'submission_id')||v_suffix END)
    ORDER BY item.ordinal)
  INTO v_archive_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants)
    WITH ORDINALITY AS item(value,ordinal);
  -- Keep this cancelled audit round in live mode so the existing closed-call
  -- billing reconciler can settle its one unresolved historical judge call.
  v_archive_config := v_round.configuration_doc || pg_catalog.jsonb_build_object(
    'round_id',v_archive_id,'mode','live','rewards_enabled',FALSE,
    'recovery_source_round_id',v_round_id,
    'recovery_original_champion_submission_id',v_round.champion_submission_id,
    'recovery_original_champion_hotkey',v_round.champion_hotkey,
    'recovery_score_runs_sha256',v_score_hash,
    'recovery_score_ledger_sha256',v_ledger_hash,
    'recovery_score_ledger_max_entry_id',v_ledger_max,
    'recovery_score_events_sha256',v_events_hash);

  IF EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
      WHERE t.tgrelid IN ('public.lab_arena_rounds'::REGCLASS,
                         'public.lab_arena_submissions'::REGCLASS,
                         'public.lab_arena_runs'::REGCLASS,
                         'public.lab_arena_ledger'::REGCLASS,
                         'public.lab_arena_trajectory_events'::REGCLASS)
        AND NOT t.tgisinternal AND t.tgenabled IS DISTINCT FROM 'O') THEN
    RAISE EXCEPTION 'Oct01 rejudge trigger state differs' USING ERRCODE='55000';
  END IF;
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_trajectory_events DISABLE TRIGGER USER;

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
    champion_fallback_providers)
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
  FROM pg_catalog.jsonb_populate_record(NULL::public.lab_arena_rounds,
    pg_catalog.to_jsonb(v_round) || pg_catalog.jsonb_build_object(
      'round_id',v_archive_id,'configuration_doc',v_archive_config,
      'participants',v_archive_participants,'status','cancelled',
      'rewards_enabled',FALSE,'reward_basis_hash',NULL,'reward_basis_doc',NULL,
      'signing_key_doc',NULL,'effective_reward_epoch',NULL,
      'reward_activated_at',NULL,'king_outcome',NULL,'king_hotkey',NULL,
      'king_start_epoch',NULL,'promotion_required',FALSE,'promotion_doc',NULL,
      'baseline_promoted_at',NULL,'champion_funding_frozen',FALSE,
      'champion_submission_id',NULL,'champion_hotkey',NULL,
      'champion_fallback_providers','[]'::JSONB,
      'cancel_reason','authorized_oct01_new_scorer_rejudge_archive'));
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(s) || pg_catalog.jsonb_build_object(
      'submission_id',CASE WHEN s.submission_id='baseline-2026-10-01'
        THEN v_archive_baseline_id ELSE s.submission_id||v_suffix END,
      'round_id',v_archive_id))).*
  FROM public.lab_arena_submissions s WHERE s.round_id=v_round_id;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>14 THEN RAISE EXCEPTION 'Oct01 rejudge submissions differ'; END IF;

  UPDATE public.lab_arena_runs s
  SET round_id=v_archive_id,
    submission_id=CASE WHEN s.submission_id='baseline-2026-10-01'
      THEN v_archive_baseline_id ELSE s.submission_id||v_suffix END
  WHERE s.round_id=v_round_id AND s.kind='score';
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>21 THEN RAISE EXCEPTION 'Oct01 rejudge score archive differs'; END IF;
  UPDATE public.lab_arena_ledger l
  SET round_id=v_archive_id,
    submission_id=CASE WHEN l.submission_id='baseline-2026-10-01'
      THEN v_archive_baseline_id ELSE l.submission_id||v_suffix END
  WHERE l.round_id=v_round_id AND EXISTS (SELECT 1 FROM public.lab_arena_runs s
    WHERE s.round_id=v_archive_id AND s.kind='score' AND s.run_id=l.run_id);
  UPDATE public.lab_arena_trajectory_events e
  SET round_id=v_archive_id,
    submission_id=CASE WHEN e.submission_id='baseline-2026-10-01'
      THEN v_archive_baseline_id ELSE e.submission_id||v_suffix END
  WHERE e.round_id=v_round_id AND e.run_kind='score';

  IF v_score_hash IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(
      pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) || pg_catalog.jsonb_build_object(
        'round_id',v_round_id,'submission_id','baseline-2026-10-01') ORDER BY run_id)::TEXT,
      'sha256'),'hex') FROM public.lab_arena_runs s
      WHERE round_id=v_archive_id AND kind='score')
     OR v_ledger_hash IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(COALESCE(
      pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) || pg_catalog.jsonb_build_object(
        'round_id',v_round_id,'submission_id','baseline-2026-10-01') ORDER BY entry_id)::TEXT,''),
      'sha256'),'hex') FROM public.lab_arena_ledger l
      WHERE round_id=v_archive_id AND EXISTS (SELECT 1 FROM public.lab_arena_runs s
        WHERE s.round_id=v_archive_id AND s.kind='score' AND s.run_id=l.run_id))
     OR v_events_hash IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(COALESCE(
      pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e) || pg_catalog.jsonb_build_object(
        'round_id',v_round_id,'submission_id','baseline-2026-10-01') ORDER BY trajectory_id)::TEXT,''),
      'sha256'),'hex') FROM public.lab_arena_trajectory_events e
      WHERE round_id=v_archive_id AND run_kind='score')
  THEN
    RAISE EXCEPTION 'Oct01 rejudge archive payload differs' USING ERRCODE='55000';
  END IF;

  UPDATE public.lab_arena_runs SET qualification_doc=NULL
  WHERE round_id=v_round_id AND stage=1 AND kind='execute';
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>10 THEN RAISE EXCEPTION 'Oct01 rejudge receipt reset differs'; END IF;
  UPDATE public.lab_arena_rounds SET status='stage1_closed',
    status_generation=status_generation+1,stage_generation=stage_generation+1,
    configuration_doc=configuration_doc || pg_catalog.jsonb_build_object(
      'schedule',v_schedule,'scorer_image_digest',v_new_digest,
      'scorer_image_reference',
      '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'||v_new_digest),
    updated_at=pg_catalog.clock_timestamp()
  WHERE round_id=v_round_id;
  ALTER TABLE public.lab_arena_trajectory_events ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  IF (SELECT status FROM public.lab_arena_rounds WHERE round_id=v_round_id)
       <> 'stage1_closed'
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND kind='score') <> 0
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_archive_id AND kind='score') <> 21
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND kind='execute' AND stage=1
           AND (per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL))
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
         WHERE t.tgrelid IN ('public.lab_arena_rounds'::REGCLASS,
                            'public.lab_arena_submissions'::REGCLASS,
                            'public.lab_arena_runs'::REGCLASS,
                            'public.lab_arena_ledger'::REGCLASS,
                            'public.lab_arena_trajectory_events'::REGCLASS)
           AND NOT t.tgisinternal AND t.tgenabled IS DISTINCT FROM 'O')
  THEN
    RAISE EXCEPTION 'Oct01 rejudge postcondition differs' USING ERRCODE='55000';
  END IF;
END;
$oct01_rejudge$;
NOTIFY pgrst, 'reload schema';
COMMIT;
