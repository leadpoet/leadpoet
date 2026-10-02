-- Rejudge accepted October 1 baseline outputs after the semantic precedence fix.
-- Preserve prior judgments, source costs, and all 130 miner executions.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $oct01_scoring_namespace$
DECLARE
  v_definition TEXT;
  v_anchor CONSTANT TEXT :=
    '    v_status := CASE WHEN v_cache.cache_key IS NULL THEN ''pending'' ELSE ''accepted'' END;';
  v_prior CONSTANT TEXT :=
    '    -- Oct01 rerun381 score namespace' || E'\n' ||
    '    IF p_round_id = ''arena-2026-10-01'' AND EXISTS (' || E'\n' ||
    '      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = ''arena-2026-10-01-r381archive'') THEN' || E'\n' ||
    '      v_assignment := p_round_id || '':'' || v_scored.submission_id || '':'' || p_stage::TEXT || '':'' || v_scored.icp_position::TEXT || '':score:rerun381'';' || E'\n' ||
    '    END IF;' || E'\n';
  v_insert CONSTANT TEXT :=
    '    -- Oct01 rerun383 score namespace' || E'\n' ||
    '    IF p_round_id = ''arena-2026-10-01'' AND EXISTS (' || E'\n' ||
    '      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = ''arena-2026-10-01-r383archive'') THEN' || E'\n' ||
    '      v_assignment := p_round_id || '':'' || v_scored.submission_id || '':'' || p_stage::TEXT || '':'' || v_scored.icp_position::TEXT || '':score:rerun383'';' || E'\n' ||
    '    END IF;' || E'\n';
BEGIN
  v_definition := pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::REGPROCEDURE);
  IF pg_catalog.strpos(v_definition,'Oct01 rerun383 score namespace') > 0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
        pg_catalog.replace(v_definition,v_insert,''))) <> pg_catalog.length(v_insert)
       OR (pg_catalog.length(v_definition)-pg_catalog.length(
         pg_catalog.replace(v_definition,v_prior,''))) <> pg_catalog.length(v_prior)
       OR (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(
         v_definition,'Oct01 rerun381 score namespace',''))) <>
         pg_catalog.length('Oct01 rerun381 score namespace')
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds
           WHERE round_id='arena-2026-10-01-r383archive') THEN
      RAISE EXCEPTION 'Oct01 semantic precedence scoring namespace replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition,'Oct01 rerun379 score namespace')=0
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_prior,''))) <> pg_catalog.length(v_prior)
     OR (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(
       v_definition,'Oct01 rerun381 score namespace',''))) <>
       pg_catalog.length('Oct01 rerun381 score namespace')
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_anchor,''))) <> pg_catalog.length(v_anchor)
  THEN
    RAISE EXCEPTION 'Oct01 semantic precedence scoring initializer differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_anchor,v_insert || v_anchor);
END;
$oct01_scoring_namespace$;

-- Retain the existing bounded stage-2 resume branch, but bind its scorer check
-- to the verified new image so the accepted miner outputs can be reused.
DO $oct01_stage2_resume_function$
DECLARE
  v_definition TEXT;
  v_old CONSTANT TEXT :=
    '  -- Oct01 rerun379 preserved miner execution resume' || E'\n' ||
    '  IF p_round_id = ''arena-2026-10-01'' AND p_stage = 2' || E'\n' ||
    '     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds' || E'\n' ||
    '       WHERE round_id = ''arena-2026-10-01-r379archive'') THEN' || E'\n' ||
    '    IF v_round.configuration_doc->>''scorer_image_digest'' IS DISTINCT FROM' || E'\n' ||
    '       ''sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65''' || E'\n' ||
    '       OR EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(p_participants) p' || E'\n' ||
    '         WHERE NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s' || E'\n' ||
    '           WHERE s.round_id=p_round_id AND s.submission_id=p->>''submission_id''' || E'\n' ||
    '             AND s.miner_hotkey=p->>''miner_hotkey''' || E'\n' ||
    '             AND s.status=''frozen'' AND NOT s.is_king))' || E'\n' ||
    '       OR (SELECT COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs' || E'\n' ||
    '         WHERE round_id=p_round_id AND stage=2 AND kind=''execute'') <> 130' || E'\n' ||
    '       OR EXISTS (SELECT 1 FROM public.lab_arena_runs r' || E'\n' ||
    '         WHERE r.round_id=p_round_id AND r.stage=2 AND' || E'\n' ||
    '           (r.kind<>''execute'' OR r.status IN (''leased'',''submitted'')' || E'\n' ||
    '             OR r.icp_position NOT BETWEEN 0 AND 9' || E'\n' ||
    '             OR r.assignment_id IS DISTINCT FROM p_round_id||'':''||r.submission_id||'':2:''||r.icp_position::TEXT' || E'\n' ||
    '             OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s' || E'\n' ||
    '               WHERE s.round_id=p_round_id AND s.submission_id=r.submission_id' || E'\n' ||
    '                 AND s.miner_hotkey=r.miner_hotkey AND s.status=''frozen'' AND NOT s.is_king' || E'\n' ||
    '                 AND EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(p_participants) p' || E'\n' ||
    '                   WHERE p->>''submission_id''=r.submission_id' || E'\n' ||
    '                     AND p->>''miner_hotkey''=r.miner_hotkey)))) THEN' || E'\n' ||
    '      RAISE EXCEPTION ''Oct01 preserved miner execution differs'' USING ERRCODE=''55000'';' || E'\n' ||
    '    END IF;' || E'\n' ||
    '    UPDATE public.lab_arena_runs SET stage_generation=v_generation' || E'\n' ||
    '      WHERE round_id=p_round_id AND stage=2 AND kind=''execute'' AND status=''pending'';' || E'\n' ||
    '    UPDATE public.lab_arena_rounds SET status=''stage2'',' || E'\n' ||
    '      status_generation=status_generation+1,stage_generation=v_generation' || E'\n' ||
    '      WHERE round_id=p_round_id;' || E'\n' ||
    '    RETURN pg_catalog.jsonb_build_object(''status'',''ok'',''round_status'',''stage2'',' || E'\n' ||
    '      ''stage_generation'',v_generation,''assignments'',130,''resumed'',TRUE);' || E'\n' ||
    '  END IF;' || E'\n';
  v_new CONSTANT TEXT := pg_catalog.replace(
    v_old,
    'sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65',
    'sha256:7d07a692c1af9221f74be9fbeb876987c41f4fdbc71222089c42c85c8d3d157b');
BEGIN
  v_definition := pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::REGPROCEDURE);
  IF pg_catalog.strpos(v_definition,'Oct01 rerun381 score namespace')>0
     OR (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(
       v_definition,'Oct01 rerun379 preserved miner execution resume',''))) <>
       pg_catalog.length('Oct01 rerun379 preserved miner execution resume')
     OR (pg_catalog.length(v_old)-pg_catalog.length(pg_catalog.replace(
       v_old,'sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65',''))) <>
       pg_catalog.length('sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65') THEN
    RAISE EXCEPTION 'Oct01 semantic precedence stage2 branch differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_definition,v_new)>0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
        pg_catalog.replace(v_definition,v_new,''))) <> pg_catalog.length(v_new)
       OR pg_catalog.strpos(v_definition,v_old)>0
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-10-01-r383archive') THEN
      RAISE EXCEPTION 'Oct01 semantic precedence stage2 resume replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old)
     OR pg_catalog.strpos(v_definition,'sha256:de2b69fba175c716d2e69b8cadec71cd42ab301ce34425b6dfd5a7df9f2427ea')>0
  THEN
    RAISE EXCEPTION 'Oct01 semantic precedence stage2 initializer differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
END;
$oct01_stage2_resume_function$;

SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control', 0));
LOCK TABLE public.lab_arena_restart_claim_control IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_trajectory_events IN ACCESS EXCLUSIVE MODE;

DO $oct01_rejudge$
DECLARE
  v_round public.lab_arena_rounds;
  v_archive public.lab_arena_rounds;
  v_previous_archive public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_archive_config JSONB;
  v_archive_participants JSONB;
  v_receipts JSONB;
  v_score_hash TEXT;
  v_ledger_hash TEXT;
  v_ledger_max BIGINT;
  v_events_hash TEXT;
  v_miner_hash TEXT;
  v_miner_ledger_hash TEXT;
  v_miner_events_hash TEXT;
  v_count INTEGER;
  v_round_id CONSTANT TEXT := 'arena-2026-10-01';
  v_archive_id CONSTANT TEXT := 'arena-2026-10-01-r383archive';
  v_suffix CONSTANT TEXT := ':r383archive';
  v_archive_baseline_id CONSTANT TEXT := 'baseline-2026-10-01-r383archive';
  v_new_digest CONSTANT TEXT :=
    'sha256:7d07a692c1af9221f74be9fbeb876987c41f4fdbc71222089c42c85c8d3d157b';
  v_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-03T08:00:02Z","publication_deadline":"2026-10-03T08:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-02T14:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-03T02:00:02Z","stage_2_start":"2026-10-02T14:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
BEGIN
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id=v_round_id FOR UPDATE;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
  WHERE round_id=v_archive_id FOR UPDATE;
  SELECT * INTO v_previous_archive FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-10-01-r381archive' FOR UPDATE;
  IF v_archive.round_id IS NOT NULL THEN
    IF v_archive.status <> 'cancelled'
       OR v_archive.rewards_enabled IS DISTINCT FROM FALSE
       OR v_archive.configuration_doc->>'mode' IS DISTINCT FROM 'live'
       OR v_archive.king_hotkey IS NOT NULL
       OR v_archive.effective_reward_epoch IS NOT NULL
       OR v_archive.promotion_required IS DISTINCT FROM FALSE
       OR v_archive.champion_funding_frozen IS DISTINCT FROM TRUE
       OR v_archive.champion_submission_id IS DISTINCT FROM
         v_archive.configuration_doc->>'recovery_original_champion_submission_id'
       OR v_archive.champion_hotkey IS DISTINCT FROM
         v_archive.configuration_doc->>'recovery_original_champion_hotkey'
       OR pg_catalog.to_jsonb(v_archive.champion_fallback_providers) IS DISTINCT FROM
         v_archive.configuration_doc->'recovery_original_champion_fallback_providers'
       OR v_archive.cancel_reason IS DISTINCT FROM
         'authorized_oct01_semantic_precedence_review_archive'
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
       OR v_archive.configuration_doc->>'recovery_baseline_receipts_sha256'
         IS DISTINCT FROM pg_catalog.encode(extensions.digest(
           (v_archive.configuration_doc->'recovery_baseline_receipts')::TEXT,
           'sha256'),'hex')
       OR pg_catalog.jsonb_array_length(
           v_archive.configuration_doc->'recovery_baseline_receipts') <> 10
       OR v_round.configuration_doc->>'scorer_image_digest'
         IS DISTINCT FROM v_new_digest
       OR v_round.configuration_doc->'schedule' IS DISTINCT FROM v_schedule
       OR (SELECT COUNT(*) FROM public.lab_arena_runs
           WHERE round_id=v_archive_id AND kind='score') <> 10
       OR (SELECT COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs
           WHERE round_id=v_round_id AND stage=1 AND kind='score'
             AND assignment_id LIKE '%:score:rerun383') NOT BETWEEN 0 AND 10
    THEN
      RAISE EXCEPTION 'Oct01 semantic precedence rejudge archive replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF v_control.singleton IS NULL OR v_control.operator_paused IS DISTINCT FROM TRUE
     OR v_control.pause_reason IS DISTINCT FROM 'oct01_semantic_precedence_review'
     OR v_control.actor_ref NOT IN
       ('oct01-semantic-precedence-hold382',
        'canonical-active-release:1d3954af027b0addc648176fe61083df07265972')
     OR v_control.guard_commitment IS DISTINCT FROM ''
     OR v_control.owner_commitment IS DISTINCT FROM ''
     OR v_control.guard_expires_at IS NOT NULL
     OR v_control.restart_phase IS DISTINCT FROM ''
     OR v_control.captured_leases IS DISTINCT FROM '[]'::JSONB THEN
    RAISE EXCEPTION 'Oct01 semantic precedence evidence rejudge does not own claim hold'
      USING ERRCODE='55000';
  END IF;
  IF v_round.round_id IS NULL OR v_round.status <> 'stage2'
     OR v_round.status_generation <> 23 OR v_round.stage_generation <> 19
     OR v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
       'sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65'
     OR v_round.configuration_doc->'schedule' IS DISTINCT FROM v_schedule
     OR v_round.configuration_doc->>'scorer_image_reference' IS DISTINCT FROM
       '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65'
     OR v_previous_archive.round_id IS NULL
     OR v_previous_archive.status IS DISTINCT FROM 'cancelled'
     OR v_previous_archive.configuration_doc->>'recovery_source_round_id'
       IS DISTINCT FROM v_round_id
     OR v_round.configuration_doc - ARRAY[
       'round_id','rewards_enabled','schedule','scorer_image_digest',
       'scorer_image_reference'] IS DISTINCT FROM
       v_previous_archive.configuration_doc - ARRAY[
       'round_id','rewards_enabled','schedule','scorer_image_digest',
       'scorer_image_reference','recovery_source_round_id',
       'recovery_original_champion_submission_id',
       'recovery_original_champion_hotkey',
       'recovery_original_champion_fallback_providers',
       'recovery_score_runs_sha256','recovery_score_ledger_sha256',
       'recovery_score_ledger_max_entry_id','recovery_score_events_sha256',
       'recovery_baseline_receipts','recovery_baseline_receipts_sha256']
     OR pg_catalog.encode(extensions.digest(v_round.participants::TEXT,'sha256'),'hex')
        IS DISTINCT FROM '872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19'
     OR (SELECT pg_catalog.encode(extensions.digest(icps::TEXT,'sha256'),'hex')
         FROM public.qualification_private_icp_sets WHERE set_id=20260930)
        IS DISTINCT FROM '8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61'
     OR (SELECT pg_catalog.encode(extensions.digest(
         pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)::TEXT,
         'sha256'),'hex') FROM public.lab_arena_submissions s WHERE round_id=v_round_id)
        IS DISTINCT FROM '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3'
     OR (SELECT pg_catalog.encode(extensions.digest(
         pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
           'run_id',run_id,'submission_id',submission_id,'stage',stage,
           'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,
           'output_ref',output_ref,'stage_generation',stage_generation)
           ORDER BY icp_position)::TEXT,'sha256'),'hex')
         FROM public.lab_arena_runs WHERE round_id=v_round_id AND stage=1 AND kind='execute')
        IS DISTINCT FROM '24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa'
     OR v_round.benchmark_ref IS DISTINCT FROM 'arena/arena-2026-10-01/benchmark.json'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-10-01'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-09-30'
     OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-10-01-r376archive' AND status='cancelled'
           AND configuration_doc->>'recovery_source_round_id'=v_round_id)
     OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-10-01-r381archive' AND status='cancelled'
           AND configuration_doc->>'recovery_source_round_id'=v_round_id)
     OR v_round.published_at IS NOT NULL OR v_round.reward_activated_at IS NOT NULL
     OR v_round.champion_funding_frozen IS DISTINCT FROM TRUE
     OR v_round.champion_submission_id IS NULL OR v_round.champion_hotkey IS NULL
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='execute'
           AND submission_id='baseline-2026-10-01' AND status='accepted'
           AND output_ref IS NOT NULL AND per_icp_score IS NOT NULL
           AND qualification_doc IS NOT NULL) <> 10
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='score') <> 10
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='score'
           AND status <> 'accepted')
     OR (SELECT COUNT(DISTINCT icp_position) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='score') <> 10
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=1 AND kind='score'
           AND assignment_id NOT LIKE '%:score:rerun381')
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs s
         LEFT JOIN public.lab_arena_runs e ON e.run_id=s.scored_run_id
         WHERE s.round_id=v_round_id AND s.kind='score'
           AND (e.run_id IS NULL OR e.round_id<>v_round_id
             OR e.kind<>'execute' OR e.status<>'accepted'
             OR e.icp_position<>s.icp_position
             OR e.submission_id<>s.submission_id))
     OR (SELECT COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=2 AND kind='execute') <> 130
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
         WHERE r.round_id=v_round_id AND r.stage=2 AND
           (r.kind<>'execute' OR r.status IN ('leased','submitted')
            OR r.icp_position NOT BETWEEN 0 AND 9
            OR r.assignment_id IS DISTINCT FROM
               v_round_id||':'||r.submission_id||':2:'||r.icp_position::TEXT
            OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s
               WHERE s.round_id=v_round_id AND s.submission_id=r.submission_id
                 AND s.status='frozen' AND NOT s.is_king)))
     OR EXISTS (SELECT 1 FROM (
         SELECT DISTINCT ON (l.call_identity) l.entry_kind
         FROM public.lab_arena_ledger l
         WHERE l.round_id=v_round_id AND l.call_identity IS NOT NULL
         ORDER BY l.call_identity,l.entry_id DESC) heads
         WHERE heads.entry_kind IN ('reservation','dispatch'))
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
  THEN
    RAISE EXCEPTION 'Oct01 semantic precedence rejudge precondition differs' USING ERRCODE='55000';
  END IF;

  SELECT pg_catalog.encode(extensions.digest(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)::TEXT,
    'sha256'),'hex') INTO v_miner_hash
  FROM public.lab_arena_runs r WHERE round_id=v_round_id AND stage=2;
  SELECT pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)::TEXT,''),
    'sha256'),'hex') INTO v_miner_ledger_hash
  FROM public.lab_arena_ledger l WHERE round_id=v_round_id AND stage=2;
  SELECT pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e) ORDER BY trajectory_id)::TEXT,''),
    'sha256'),'hex') INTO v_miner_events_hash
  FROM public.lab_arena_trajectory_events e WHERE round_id=v_round_id AND stage=2;

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
  SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
    'run_id',run_id,'per_icp_score',per_icp_score,
    'qualification_doc',qualification_doc) ORDER BY icp_position)
  INTO v_receipts FROM public.lab_arena_runs
  WHERE round_id=v_round_id AND stage=1 AND kind='execute';

  SELECT pg_catalog.jsonb_agg(item.value || pg_catalog.jsonb_build_object(
    'submission_id',CASE WHEN item.value->>'submission_id'='baseline-2026-10-01'
      THEN v_archive_baseline_id ELSE (item.value->>'submission_id')||v_suffix END)
    ORDER BY item.ordinal)
  INTO v_archive_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants)
    WITH ORDINALITY AS item(value,ordinal);
  -- Keep this cancelled audit round in live mode so the existing closed-call
  -- billing reconciler can settle any late historical judge call.
  v_archive_config := v_round.configuration_doc || pg_catalog.jsonb_build_object(
    'round_id',v_archive_id,'mode','live','rewards_enabled',FALSE,
    'recovery_source_round_id',v_round_id,
    'recovery_original_champion_submission_id',v_round.champion_submission_id,
    'recovery_original_champion_hotkey',v_round.champion_hotkey,
    'recovery_original_champion_fallback_providers',v_round.champion_fallback_providers,
    'recovery_score_runs_sha256',v_score_hash,
    'recovery_score_ledger_sha256',v_ledger_hash,
    'recovery_score_ledger_max_entry_id',v_ledger_max,
    'recovery_score_events_sha256',v_events_hash,
    'recovery_baseline_receipts',v_receipts,
    'recovery_baseline_receipts_sha256',
      pg_catalog.encode(extensions.digest(v_receipts::TEXT,'sha256'),'hex'));

  IF EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
      WHERE t.tgrelid IN ('public.lab_arena_rounds'::REGCLASS,
                         'public.lab_arena_submissions'::REGCLASS,
                         'public.lab_arena_runs'::REGCLASS,
                         'public.lab_arena_ledger'::REGCLASS,
                         'public.lab_arena_trajectory_events'::REGCLASS)
        AND NOT t.tgisinternal AND t.tgenabled IS DISTINCT FROM 'O') THEN
    RAISE EXCEPTION 'Oct01 semantic precedence rejudge trigger state differs' USING ERRCODE='55000';
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
      'baseline_promoted_at',NULL,
      'cancel_reason','authorized_oct01_semantic_precedence_review_archive'));
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(s) || pg_catalog.jsonb_build_object(
      'submission_id',CASE WHEN s.submission_id='baseline-2026-10-01'
        THEN v_archive_baseline_id ELSE s.submission_id||v_suffix END,
      'round_id',v_archive_id))).*
  FROM public.lab_arena_submissions s WHERE s.round_id=v_round_id;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>14 THEN RAISE EXCEPTION 'Oct01 semantic precedence rejudge submissions differ'; END IF;

  UPDATE public.lab_arena_runs s
  SET round_id=v_archive_id,
    submission_id=CASE WHEN s.submission_id='baseline-2026-10-01'
      THEN v_archive_baseline_id ELSE s.submission_id||v_suffix END
  WHERE s.round_id=v_round_id AND s.kind='score';
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>10 THEN RAISE EXCEPTION 'Oct01 semantic precedence rejudge score archive differs'; END IF;
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
    RAISE EXCEPTION 'Oct01 semantic precedence rejudge archive payload differs' USING ERRCODE='55000';
  END IF;

  UPDATE public.lab_arena_runs SET qualification_doc=NULL,per_icp_score=NULL
  WHERE round_id=v_round_id AND stage=1 AND kind='execute';
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>10 THEN RAISE EXCEPTION 'Oct01 semantic precedence rejudge receipt reset differs'; END IF;
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
         WHERE round_id=v_archive_id AND kind='score') <> 10
     OR (SELECT COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=2 AND kind='execute') <> 130
     OR v_miner_hash IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(
         pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)::TEXT,
         'sha256'),'hex') FROM public.lab_arena_runs r
         WHERE round_id=v_round_id AND stage=2)
     OR v_miner_ledger_hash IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(COALESCE(
         pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)::TEXT,''),
         'sha256'),'hex') FROM public.lab_arena_ledger l
         WHERE round_id=v_round_id AND stage=2)
     OR v_miner_events_hash IS DISTINCT FROM (SELECT pg_catalog.encode(extensions.digest(COALESCE(
         pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e) ORDER BY trajectory_id)::TEXT,''),
         'sha256'),'hex') FROM public.lab_arena_trajectory_events e
         WHERE round_id=v_round_id AND stage=2)
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
    RAISE EXCEPTION 'Oct01 semantic precedence rejudge postcondition differs' USING ERRCODE='55000';
  END IF;
  UPDATE public.lab_arena_restart_claim_control
  SET operator_paused=FALSE,pause_reason='',
      actor_ref='oct01-semantic-precedence-rejudge383',
      updated_at=pg_catalog.clock_timestamp()
  WHERE singleton AND operator_paused
    AND pause_reason='oct01_semantic_precedence_review'
    AND actor_ref IN ('oct01-semantic-precedence-hold382',
      'canonical-active-release:1d3954af027b0addc648176fe61083df07265972');
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Oct01 semantic precedence hold release differs' USING ERRCODE='55000';
  END IF;
END;
$oct01_rejudge$;
NOTIFY pgrst, 'reload schema';
COMMIT;
