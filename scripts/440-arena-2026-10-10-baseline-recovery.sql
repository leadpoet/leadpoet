-- Exact October 10 host-failure recovery and corrected-image saved-output rejudge.
-- Preserve bank, source, 69 miners, old attempts and costs. Extend only baseline
-- execution close from 04:30:01 to 05:30:01 UTC; all later deadlines stay fixed.
-- Recovery ends when these four attempts and the corrected judgments complete.
-- First apply needs the full frozen 4500-second lease before the new close.
-- The four exceptional attempt-3 rows do not change the ordinary two-attempt policy.
-- Corrected scorer source: a2e0e67f6c37754a76d2fcf4bfe4ad9219813e76 (PR323).
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_restart_claim_control IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_trajectory_events IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.qualification_private_icp_sets IN SHARE MODE;
DO $oct10_recovery440$
DECLARE
  v_round public.lab_arena_rounds;
  v_archive public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_config JSONB;
  v_receipts JSONB;
  v_scores JSONB;
  v_score_ledger JSONB;
  v_score_events JSONB;
  v_executions JSONB;
  v_ledger JSONB;
  v_events JSONB;
  v_triggers JSONB;
  v_definition TEXT;
  v_count INTEGER;
  v_round_id CONSTANT TEXT := 'arena-2026-10-10';
  v_baseline CONSTANT TEXT := 'baseline-2026-10-10';
  v_archive_id CONSTANT TEXT := 'arena-2026-10-10-r440archive';
  v_archive_baseline CONSTANT TEXT := 'baseline-2026-10-10-r440archive';
  v_new_digest CONSTANT TEXT := 'sha256:3be7e227ea62eddcdd6871af28bbacd8623db929f5fe393769bc98dad34a35ba';
  v_config_hash CONSTANT TEXT := 'a0e29abee1790aa5d525a2145c59b4272695fbd7bc80bf098b4d90993ecc31a3';
  v_participants_hash CONSTANT TEXT := 'e272d0d25813b1d4c60f62cee8753c96d7f58b93660ea3086a46d03fd252acb8';
  v_submissions_hash CONSTANT TEXT := 'f4608f08670b523fccc7de0708c7c24208dfb3afbfb3ad5f0fda6afe137a5e4a';
  v_runs_hash CONSTANT TEXT := '6fd4541aa2858cb1413351ffc5227e72d1b244d21bbac83e44ecc2f64a4dcb45';
  v_bank_hash CONSTANT TEXT := '30b673c4c32d2ef3716701aa65d543b70cc87c6414fe281a45d6fb7e8d84dfb8';
  v_anchor CONSTANT TEXT :=
    '    v_status := CASE WHEN v_cache.cache_key IS NULL THEN ''pending'' ELSE ''accepted'' END;';
  v_namespace CONSTANT TEXT := $namespace$    -- Oct10 recovery440 score namespace
    IF p_round_id = 'arena-2026-10-10' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds
      WHERE round_id = 'arena-2026-10-10-r440archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:recovery440';
    END IF;
$namespace$;
BEGIN
  IF pg_catalog.current_setting('session_replication_role') <> 'origin'
     OR v_new_digest !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'Oct10 recovery origin execution and reviewed image required';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
    WHERE round_id=v_round_id FOR UPDATE;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
    WHERE round_id=v_archive_id FOR UPDATE;
  IF v_round.round_id IS NULL THEN
    RAISE EXCEPTION 'Oct10 recovery round missing';
  END IF;
  IF v_archive.round_id IS NOT NULL THEN
    IF v_archive.status <> 'cancelled' OR v_archive.rewards_enabled
       OR v_archive.cancel_reason IS DISTINCT FROM 'authorized_oct10_baseline_recovery440'
       OR v_archive.configuration_doc->>'recovery_source_round_id' IS DISTINCT FROM v_round_id
       OR v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM v_new_digest
       OR v_round.configuration_doc->'schedule' IS DISTINCT FROM jsonb_set(v_archive.configuration_doc->'schedule','{stage_1_close}','"2026-10-10T05:30:01Z"'::JSONB)
       OR v_round.configuration_doc->>'max_attempts_per_assignment' IS DISTINCT FROM '2'
       OR encode(extensions.digest((v_round.configuration_doc-'scorer_image_digest'-'scorer_image_reference')::TEXT,'sha256'),'hex') IS DISTINCT FROM v_archive.configuration_doc->>'recovery_configuration_body_sha256'
       OR encode(extensions.digest(v_round.participants::TEXT,'sha256'),'hex') IS DISTINCT FROM v_participants_hash
       OR encode(extensions.digest((SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id=v_round_id)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_submissions_hash
       OR encode(extensions.digest((SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20261009)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_bank_hash
       OR encode(extensions.digest((SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at' ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=v_round_id AND kind='execute' AND attempt<=2)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_archive.configuration_doc->>'recovery_execution_body_sha256'
       OR (SELECT count(*) FROM public.lab_arena_runs
           WHERE round_id=v_round_id AND kind='execute' AND attempt=3
             AND submission_id=v_baseline AND stage=1 AND icp_position IN (2,5,7,8)
             AND stage_generation=5) <> 4
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id=v_round_id AND r.attempt=3
            AND (r.submission_id<>v_baseline OR r.stage<>1 OR r.kind<>'execute'
              OR r.icp_position NOT IN (2,5,7,8) OR r.stage_generation<>5
              OR r.assignment_id IS DISTINCT FROM v_round_id||':'||v_baseline||':1:'||r.icp_position::TEXT
              OR r.run_id IS DISTINCT FROM r.assignment_id||':3'))
       OR (SELECT count(*) FROM public.lab_arena_runs
           WHERE round_id=v_archive_id AND kind='score') <> 6
       OR encode(extensions.digest((v_archive.configuration_doc->'recovery_baseline_receipts')::TEXT,'sha256'),'hex') IS DISTINCT FROM v_archive.configuration_doc->>'recovery_baseline_receipts_sha256'
       OR encode(extensions.digest(pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),'sha256'),'hex') IS DISTINCT FROM v_archive.configuration_doc->>'recovery_scoring_function_sha256'
       OR v_archive.configuration_doc->>'recovery_score_runs_sha256' IS DISTINCT FROM
         (SELECT encode(extensions.digest(jsonb_agg(to_jsonb(s)||jsonb_build_object('round_id',v_round_id,
             'submission_id',v_baseline) ORDER BY run_id)::TEXT,'sha256'),'hex')
          FROM public.lab_arena_runs s WHERE round_id=v_archive_id AND kind='score')
       OR v_archive.configuration_doc->>'recovery_score_ledger_sha256' IS DISTINCT FROM
         (SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(l)||jsonb_build_object(
            'round_id',v_round_id,'submission_id',v_baseline) ORDER BY entry_id),'[]'::JSONB)::TEXT,
            'sha256'),'hex') FROM public.lab_arena_ledger l WHERE round_id=v_archive_id
            AND entry_id <= (v_archive.configuration_doc->>'recovery_score_ledger_max_entry_id')::BIGINT)
       OR v_archive.configuration_doc->>'recovery_score_events_sha256' IS DISTINCT FROM
         (SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(e)||jsonb_build_object(
            'round_id',v_round_id,'submission_id',v_baseline) ORDER BY trajectory_id),'[]'::JSONB)::TEXT,
            'sha256'),'hex') FROM public.lab_arena_trajectory_events e WHERE round_id=v_archive_id
            AND trajectory_id <= (v_archive.configuration_doc->>'recovery_score_events_max_id')::BIGINT) THEN
      RAISE EXCEPTION 'Oct10 recovery replay differs';
    END IF;
    RETURN;
  END IF;
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF NOT FOUND OR v_control.operator_paused OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> '' OR v_control.restart_scope <> ''
     OR v_control.restart_phase <> '' OR v_control.guard_expires_at IS NOT NULL
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'Oct10 recovery restart or operator guard active';
  END IF;
  IF v_round.status IS DISTINCT FROM 'stage1_judged'
     OR v_round.status_generation <> 5 OR v_round.stage_generation <> 4
     OR v_round.configuration_doc->>'max_attempts_per_assignment' IS DISTINCT FROM '2'
     OR v_round.published_at IS NOT NULL OR v_round.publication_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL OR v_round.effective_reward_epoch IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.configuration_doc#>>'{schedule,stage_1_close}' IS DISTINCT FROM '2026-10-10T04:30:01Z'
     OR v_round.configuration_doc->>'checkpoint_deadline_policy' IS DISTINCT FROM 'atomic_checkpoint_60m_v1'
     OR v_round.configuration_doc->>'icp_wall_clock_seconds' IS DISTINCT FROM '3600'
     OR v_round.configuration_doc->>'lease_ttl_seconds' IS DISTINCT FROM '4500'
     OR pg_catalog.clock_timestamp()+INTERVAL '4500 seconds' >= '2026-10-10T05:30:01Z'::TIMESTAMPTZ
     OR encode(extensions.digest(v_round.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM v_config_hash
     OR encode(extensions.digest(v_round.participants::TEXT,'sha256'),'hex') IS DISTINCT FROM v_participants_hash
     OR jsonb_array_length(v_round.participants) <> 70
     OR encode(extensions.digest((SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id)
          FROM public.lab_arena_submissions s WHERE round_id=v_round_id)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_submissions_hash
     OR encode(extensions.digest((SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id)
          FROM public.lab_arena_runs r WHERE round_id=v_round_id)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_runs_hash
     OR encode(extensions.digest((SELECT icps FROM public.qualification_private_icp_sets
          WHERE set_id=20261009)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_bank_hash
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=v_round_id
          AND (stage<>1 OR status IN ('pending','leased','submitted')))
     OR (SELECT count(*) FROM public.lab_arena_submissions
          WHERE round_id=v_round_id AND NOT is_king AND status='frozen') <> 69
     OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
          AND kind='execute' AND status='accepted' AND output_ref IS NOT NULL) <> 6
     OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
          AND kind='score' AND status='accepted' AND output_ref IS NOT NULL) <> 6
     OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
          AND kind='execute' AND attempt=2 AND icp_position IN (2,5,7,8)
          AND status='failed' AND terminal_cause='lease_expired' AND output_ref IS NULL) <> 4
  THEN
    RAISE EXCEPTION 'Oct10 recovery exact state or full execution window differs';
  END IF;
  -- Do not overlap outstanding charged work or erase uncertain accounting.
  IF EXISTS (SELECT 1 FROM (
       SELECT DISTINCT ON (call_identity) entry_kind
       FROM public.lab_arena_ledger WHERE round_id=v_round_id
         AND run_id IN (SELECT run_id FROM public.lab_arena_runs WHERE round_id=v_round_id AND submission_id=v_baseline)
         AND call_identity IS NOT NULL ORDER BY call_identity,entry_id DESC
     ) head WHERE entry_kind IN ('reservation','dispatch','uncertain')) THEN
    RAISE EXCEPTION 'Oct10 recovery provider accounting remains open';
  END IF;
  SELECT jsonb_agg(to_jsonb(t) ORDER BY tgrelid,tgname) INTO v_triggers
    FROM pg_trigger t WHERE tgrelid IN ('public.lab_arena_rounds'::regclass,
      'public.lab_arena_submissions'::regclass,'public.lab_arena_runs'::regclass,
      'public.lab_arena_ledger'::regclass,'public.lab_arena_trajectory_events'::regclass)
      AND NOT tgisinternal;
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid IN (
       'public.lab_arena_rounds'::regclass,'public.lab_arena_runs'::regclass,
       'public.lab_arena_ledger'::regclass) AND NOT tgisinternal AND tgenabled<>'O') THEN
    RAISE EXCEPTION 'Oct10 recovery trigger state differs';
  END IF;
  IF (SELECT encode(extensions.digest(jsonb_agg(jsonb_build_array(c.relname,t.tgname,
       t.tgenabled,pg_get_triggerdef(t.oid),pg_get_functiondef(t.tgfoid),owner.rolname,
       p.proacl::text,p.prosecdef,p.proconfig) ORDER BY c.relname,t.tgname)::text,'sha256'),'hex')
       FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_proc p ON p.oid=t.tgfoid
       JOIN pg_roles owner ON owner.oid=p.proowner WHERE t.tgrelid IN (
         'public.lab_arena_rounds'::regclass,'public.lab_arena_runs'::regclass,
         'public.lab_arena_ledger'::regclass) AND NOT t.tgisinternal AND t.tgname IN (
         'lab_arena_rounds_write_once','lab_arena_runs_terminal','lab_arena_integrity_run_guard',
         'lab_arena_runs_participation','lab_arena_ledger_append_only')) IS DISTINCT FROM
       '5b74d744d9aaae0bbc1b18c12e6a943b07c83ee5b8cad92aacc094117cc52067' THEN
    RAISE EXCEPTION 'Oct10 recovery protected guard shape differs';
  END IF;
  IF (SELECT jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
       FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
       WHERE p.oid='public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)
       IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Oct10 recovery scoring initializer security differs';
  END IF;
  SELECT pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)
    INTO v_definition;
  IF encode(extensions.digest(v_definition,'sha256'),'hex') <>
       '847e3ad30de957162f59ec3cd58d52e99199e6a68c098484406d5830012e1b5c'
     OR length(v_definition)-length(replace(v_definition,v_anchor,''))<>length(v_anchor) THEN
    RAISE EXCEPTION 'Oct10 recovery score initializer differs';
  END IF;
  SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) INTO v_scores
    FROM public.lab_arena_runs r WHERE round_id=v_round_id AND kind='score';
  SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),'[]'::JSONB) INTO v_score_ledger
    FROM public.lab_arena_ledger l WHERE round_id=v_round_id
      AND run_id IN (SELECT run_id FROM public.lab_arena_runs WHERE round_id=v_round_id AND kind='score');
  SELECT coalesce(jsonb_agg(to_jsonb(e) ORDER BY trajectory_id),'[]'::JSONB) INTO v_score_events
    FROM public.lab_arena_trajectory_events e WHERE round_id=v_round_id AND run_kind='score';
  SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'per_icp_score',per_icp_score,
      'qualification_doc',qualification_doc) ORDER BY run_id) INTO v_receipts
    FROM public.lab_arena_runs WHERE round_id=v_round_id AND kind='execute';
  SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-'qualification_doc' ORDER BY run_id) INTO v_executions
    FROM public.lab_arena_runs r WHERE round_id=v_round_id AND kind='execute';
  SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),'[]'::JSONB) INTO v_ledger
    FROM public.lab_arena_ledger l WHERE round_id=v_round_id AND NOT EXISTS
      (SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id=v_round_id AND r.kind='score' AND r.run_id=l.run_id);
  SELECT coalesce(jsonb_agg(to_jsonb(e) ORDER BY trajectory_id),'[]'::JSONB) INTO v_events
    FROM public.lab_arena_trajectory_events e WHERE round_id=v_round_id AND run_kind<>'score';
  v_config := jsonb_set(v_round.configuration_doc,'{schedule,stage_1_close}',
    '"2026-10-10T05:30:01Z"'::JSONB,FALSE)||jsonb_build_object(
    'scorer_image_digest',v_new_digest,'scorer_image_reference',
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'||v_new_digest);

  -- Disable only the guards which prohibit the exact archive/receipt correction.
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_integrity_run_guard;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_participation;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER lab_arena_ledger_append_only;
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
  SELECT
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
  FROM jsonb_populate_record(NULL::public.lab_arena_rounds,to_jsonb(v_round)||jsonb_build_object(
    'round_id',v_archive_id,'status','cancelled','rewards_enabled',FALSE,
    'participants',jsonb_build_array((SELECT p||jsonb_build_object('submission_id',v_archive_baseline)
       FROM jsonb_array_elements(v_round.participants) p WHERE p->>'submission_id'=v_baseline)),
    'configuration_doc',v_round.configuration_doc||jsonb_build_object('round_id',v_archive_id,
       'rewards_enabled',FALSE,'recovery_source_round_id',v_round_id,
       'recovery_baseline_receipts',v_receipts,
       'recovery_baseline_receipts_sha256',encode(extensions.digest(v_receipts::TEXT,'sha256'),'hex'),
       'recovery_scoring_function_sha256',encode(extensions.digest(replace(v_definition,v_anchor,v_namespace||v_anchor),'sha256'),'hex'),
       'recovery_configuration_body_sha256',encode(extensions.digest((v_config-'scorer_image_digest'-'scorer_image_reference')::TEXT,'sha256'),'hex'),
       'recovery_execution_body_sha256',(SELECT encode(extensions.digest(jsonb_agg(to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at' ORDER BY run_id)::TEXT,'sha256'),'hex') FROM public.lab_arena_runs r WHERE round_id=v_round_id AND kind='execute'),
       'recovery_score_ledger_max_entry_id',(SELECT coalesce(max(entry_id),0) FROM public.lab_arena_ledger WHERE round_id=v_round_id),
       'recovery_score_events_max_id',(SELECT coalesce(max(trajectory_id),0) FROM public.lab_arena_trajectory_events WHERE round_id=v_round_id),
       'recovery_score_runs_sha256',encode(extensions.digest(v_scores::TEXT,'sha256'),'hex'),
       'recovery_score_ledger_sha256',encode(extensions.digest(v_score_ledger::TEXT,'sha256'),'hex'),
       'recovery_score_events_sha256',encode(extensions.digest(v_score_events::TEXT,'sha256'),'hex')),
    'promotion_required',FALSE,'promotion_doc',NULL,'baseline_promoted_at',NULL,
    'cancel_reason','authorized_oct10_baseline_recovery440','reward_basis_doc',NULL,
    'reward_basis_hash',NULL,'signing_key_doc',NULL,'effective_reward_epoch',NULL,
    'reward_activated_at',NULL,'king_outcome',NULL,'king_hotkey',NULL,'king_start_epoch',NULL));
  INSERT INTO public.lab_arena_submissions
    SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,to_jsonb(s)||
      jsonb_build_object('round_id',v_archive_id,'submission_id',v_archive_baseline))).*
    FROM public.lab_arena_submissions s WHERE round_id=v_round_id AND submission_id=v_baseline;
  UPDATE public.lab_arena_runs SET round_id=v_archive_id,submission_id=v_archive_baseline
    WHERE round_id=v_round_id AND kind='score';
  UPDATE public.lab_arena_ledger SET round_id=v_archive_id,submission_id=v_archive_baseline
    WHERE round_id=v_round_id AND run_id IN
      (SELECT run_id FROM public.lab_arena_runs WHERE round_id=v_archive_id);
  UPDATE public.lab_arena_trajectory_events SET round_id=v_archive_id,submission_id=v_archive_baseline
    WHERE round_id=v_round_id AND run_kind='score';
  UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
    WHERE round_id=v_round_id AND kind='execute';
  UPDATE public.lab_arena_rounds SET status='stage1',status_generation=6,
    stage_generation=5,stage1_scoring_plan_doc=NULL,configuration_doc=v_config
    WHERE round_id=v_round_id;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER lab_arena_ledger_append_only;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_participation;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_integrity_run_guard;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;
  INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,
    miner_hotkey,stage,icp_position,attempt,status,stage_generation,kind,previous_runner_hotkey)
    SELECT assignment_id||':3',assignment_id,round_id,submission_id,miner_hotkey,
      stage,icp_position,3,'pending',5,'execute',runner_hotkey
    FROM public.lab_arena_runs WHERE round_id=v_round_id AND kind='execute'
      AND attempt=2 AND icp_position IN (2,5,7,8);
  GET DIAGNOSTICS v_count=ROW_COUNT;
  EXECUTE replace(v_definition,v_anchor,v_namespace||v_anchor);

  IF v_count<>4
     OR (SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=v_round_id) IS DISTINCT FROM v_config
     OR (SELECT to_jsonb(r)-'configuration_doc'-'status'-'status_generation'-'stage_generation'-'stage1_scoring_plan_doc'-'updated_at' FROM public.lab_arena_rounds r WHERE round_id=v_round_id) IS DISTINCT FROM (to_jsonb(v_round)-'configuration_doc'-'status'-'status_generation'-'stage_generation'-'stage1_scoring_plan_doc'-'updated_at')
     OR (SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-'qualification_doc' ORDER BY run_id)
          FROM public.lab_arena_runs r WHERE round_id=v_round_id AND kind='execute' AND attempt<>3)
          IS DISTINCT FROM v_executions
     OR (SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),'[]'::JSONB)
          FROM public.lab_arena_ledger l WHERE round_id=v_round_id) IS DISTINCT FROM v_ledger
     OR (SELECT coalesce(jsonb_agg(to_jsonb(e) ORDER BY trajectory_id),'[]'::JSONB)
          FROM public.lab_arena_trajectory_events e WHERE round_id=v_round_id) IS DISTINCT FROM v_events
     OR (SELECT jsonb_agg(to_jsonb(s)||jsonb_build_object('round_id',v_round_id,
          'submission_id',v_baseline) ORDER BY run_id)
          FROM public.lab_arena_runs s WHERE round_id=v_archive_id) IS DISTINCT FROM v_scores
     OR (SELECT coalesce(jsonb_agg(to_jsonb(l)||jsonb_build_object('round_id',v_round_id,
          'submission_id',v_baseline) ORDER BY entry_id),'[]'::JSONB)
          FROM public.lab_arena_ledger l WHERE round_id=v_archive_id) IS DISTINCT FROM v_score_ledger
     OR (SELECT coalesce(jsonb_agg(to_jsonb(e)||jsonb_build_object('round_id',v_round_id,
          'submission_id',v_baseline) ORDER BY trajectory_id),'[]'::JSONB)
          FROM public.lab_arena_trajectory_events e WHERE round_id=v_archive_id) IS DISTINCT FROM v_score_events
     OR (SELECT jsonb_agg(to_jsonb(t) ORDER BY tgrelid,tgname) FROM pg_trigger t
          WHERE tgrelid IN ('public.lab_arena_rounds'::regclass,'public.lab_arena_submissions'::regclass,
            'public.lab_arena_runs'::regclass,'public.lab_arena_ledger'::regclass,
            'public.lab_arena_trajectory_events'::regclass) AND NOT tgisinternal) IS DISTINCT FROM v_triggers
     OR encode(extensions.digest((SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id)
          FROM public.lab_arena_submissions s WHERE round_id=v_round_id)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_submissions_hash
     OR encode(extensions.digest((SELECT icps FROM public.qualification_private_icp_sets
          WHERE set_id=20261009)::TEXT,'sha256'),'hex') IS DISTINCT FROM v_bank_hash THEN
    RAISE EXCEPTION 'Oct10 recovery preservation or trigger restoration differs';
  END IF;
END;
$oct10_recovery440$;
COMMIT;
