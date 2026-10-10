-- PREPARATION TEMPLATE ONLY. Required: REVIEWED_HOLD_INVENTORY_JSON.
-- Apply only after normal stage2 closure marked infrastructure-incomplete work.
-- Seal all immutable execution attempts; 690 counts assignments, never attempts.
-- Judge progress may change between review and hold application. Existing judges
-- may finish. This hold opens no work and changes no round, run, output or cost.
-- Render and review the exact terminal snapshot before applying.
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
LOCK TABLE public.qualification_private_icp_sets IN SHARE MODE;

DO $oct10_uniform_hold444$
DECLARE
  v_round public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_round_id CONSTANT TEXT := 'arena-2026-10-10';
  v_archive_id CONSTANT TEXT := 'arena-2026-10-10-r445archive';
  v_archive_baseline_id CONSTANT TEXT := 'baseline-2026-10-10-r445archive';
  v_actor CONSTANT TEXT := 'oct10-uniform-pe-hold444';
  v_reason CONSTANT TEXT := 'oct10_uniform_pe_boundary_recovery';
  v_expected CONSTANT JSONB := '{"bank_sha256": "30b673c4c32d2ef3716701aa65d543b70cc87c6414fe281a45d6fb7e8d84dfb8", "execution_attempts": 764, "execution_sha256": "c5cfc22a173bb2ef7bae38f9a66eb1864ecaadd8bc4301ee7d3a00bded992e49", "functions_security_sha256": "a53d4ad26c0a68379ee93307701e3ccb8a27acca0e21ba249a50f8f5cae6dae1", "hold_generation": 418, "hold_sha256": "666f5118a6dd2ef186edf2b071f42302d12aaa165c0b16d6abf4727a2a5ba010", "participants_sha256": "e272d0d25813b1d4c60f62cee8753c96d7f58b93660ea3086a46d03fd252acb8", "round_identity_sha256": "c16415455500cd59bff0def17e98ea9c3f98cd20aed5a5281db56cc557e89af0", "round_stage_generation": 10, "round_status": "stage2_closed", "round_status_generation": 12, "scoring_function_sha256": "8777bd9b757a774623233aa5c5dc2a5e5ca1666fb127ce6132ffe0ce789b6b61", "stage_function_sha256": "35adefcc70140ea9378d19413f5a6619beaf4ddc7bc41431ec8d08a80b5b477c", "submissions_sha256": "f4608f08670b523fccc7de0708c7c24208dfb3afbfb3ad5f0fda6afe137a5e4a", "triggers_sha256": "cdd4b970ab4e074af3c82db94b2be3a7a95399af716638245cb28318a55c5cc1"}'::JSONB;
  v_frozen JSONB;
  v_actual JSONB;
  v_count INTEGER;
BEGIN
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id=v_round_id FOR UPDATE;
  IF current_setting('session_replication_role')<>'origin'
     OR jsonb_typeof(v_expected)<>'object' OR v_round.round_id IS NULL
     OR v_control.singleton IS NULL OR v_control.guard_commitment<>''
     OR v_control.owner_commitment<>'' OR v_control.restart_scope<>''
     OR v_control.restart_phase<>'' OR v_control.guard_expires_at IS NOT NULL
     OR v_control.captured_leases<>'[]'::JSONB
     OR v_control.guard_generation IS DISTINCT FROM (v_expected->>'hold_generation')::BIGINT
     OR (v_control.operator_paused AND (v_control.actor_ref,v_control.pause_reason)
          IS DISTINCT FROM (v_actor,v_reason)) THEN
    RAISE EXCEPTION 'Oct10 uniform hold444 owner or restart guard differs';
  END IF;
  SELECT jsonb_build_object(
    'triggers_sha256',(SELECT encode(extensions.digest(coalesce(jsonb_agg(jsonb_build_array(c.relname,t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),pg_get_functiondef(t.tgfoid),owner.rolname,p.proacl::text,p.prosecdef,p.proconfig) ORDER BY c.relname,t.tgname),'[]'::jsonb)::text,'sha256'),'hex') FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_proc p ON p.oid=t.tgfoid JOIN pg_roles owner ON owner.oid=p.proowner WHERE t.tgrelid IN ('public.lab_arena_rounds'::regclass,'public.lab_arena_submissions'::regclass,'public.lab_arena_runs'::regclass,'public.lab_arena_ledger'::regclass,'public.lab_arena_trajectory_events'::regclass) AND NOT t.tgisinternal),
    'functions_security_sha256',(SELECT encode(extensions.digest(jsonb_agg(jsonb_build_array(p.oid::regprocedure::text,owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig) ORDER BY p.oid::regprocedure::text)::text,'sha256'),'hex') FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid IN('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure,'public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure)),
    'participants_sha256',encode(extensions.digest(v_round.participants::text,'sha256'),'hex'),
    'submissions_sha256',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(s) ORDER BY submission_id),'[]'::jsonb)::text,'sha256'),'hex') FROM public.lab_arena_submissions s WHERE round_id=v_round_id),
    'bank_sha256',(SELECT encode(extensions.digest(icps::text,'sha256'),'hex') FROM public.qualification_private_icp_sets WHERE set_id=20261009),
    -- Stable resume identity permits only normal score writes and their timestamp.
    'execution_sha256',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(r)-'qualification_doc'-'per_icp_score'-'updated_at' ORDER BY run_id),'[]'::jsonb)::text,'sha256'),'hex') FROM public.lab_arena_runs r WHERE round_id=v_round_id AND kind='execute'),
    'execution_attempts',(SELECT count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id AND kind='execute')
  ) INTO v_frozen;
  SELECT v_frozen || jsonb_build_object(
    'round_identity_sha256',encode(extensions.digest((to_jsonb(v_round)-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','stage2_scoring_plan_doc','finalists','updated_at'])::text,'sha256'),'hex'),
    'hold_generation',v_control.guard_generation,
    'hold_sha256',encode(extensions.digest(to_jsonb(v_control)::text,'sha256'),'hex'),
    'scoring_function_sha256',encode(extensions.digest(pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),'sha256'),'hex'),
    'stage_function_sha256',encode(extensions.digest(pg_get_functiondef('public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure),'sha256'),'hex'),
    'triggers_sha256',(SELECT encode(extensions.digest(coalesce(jsonb_agg(jsonb_build_array(c.relname,t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),pg_get_functiondef(t.tgfoid),owner.rolname,p.proacl::text,p.prosecdef,p.proconfig) ORDER BY c.relname,t.tgname),'[]'::jsonb)::text,'sha256'),'hex') FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_proc p ON p.oid=t.tgfoid JOIN pg_roles owner ON owner.oid=p.proowner WHERE t.tgrelid IN ('public.lab_arena_rounds'::regclass,'public.lab_arena_submissions'::regclass,'public.lab_arena_runs'::regclass,'public.lab_arena_ledger'::regclass,'public.lab_arena_trajectory_events'::regclass) AND NOT t.tgisinternal)
  ) INTO v_actual;
  IF (v_control.operator_paused AND (v_actual-'hold_sha256') IS DISTINCT FROM (v_expected-ARRAY['hold_sha256','round_status','round_stage_generation','round_status_generation']))
     OR (NOT v_control.operator_paused AND v_actual IS DISTINCT FROM (v_expected-ARRAY['round_status','round_stage_generation','round_status_generation'])) THEN
    RAISE EXCEPTION 'Oct10 uniform hold444 reviewed preimage differs';
  END IF;
  IF array_position(ARRAY['stage2_closed','stage2_scoring','stage2_judged','scored'],v_round.status)
       < array_position(ARRAY['stage2_closed','stage2_scoring','stage2_judged','scored'],v_expected->>'round_status')
     OR v_round.stage_generation IS DISTINCT FROM ((v_expected->>'round_stage_generation')::INTEGER
       + (CASE v_round.status WHEN 'stage2_closed' THEN 0 WHEN 'stage2_scoring' THEN 1 ELSE 2 END)
       - (CASE v_expected->>'round_status' WHEN 'stage2_closed' THEN 0 WHEN 'stage2_scoring' THEN 1 ELSE 2 END))
     OR v_round.status_generation IS DISTINCT FROM ((v_expected->>'round_status_generation')::INTEGER
       + array_position(ARRAY['stage2_closed','stage2_scoring','stage2_judged','scored'],v_round.status)
       - array_position(ARRAY['stage2_closed','stage2_scoring','stage2_judged','scored'],v_expected->>'round_status'))
     OR v_round.status NOT IN ('stage2_closed','stage2_scoring','stage2_judged','scored')
     OR v_round.configuration_doc->>'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-10-10'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-10-09'
     OR v_round.published_at IS NOT NULL OR v_round.publication_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL OR v_round.effective_reward_epoch IS NOT NULL
     OR EXISTS (SELECT 1 FROM public.lab_arena_rounds WHERE round_id=v_archive_id)
     OR jsonb_array_length(v_round.participants)<>70
     OR (SELECT jsonb_array_length(icps) FROM public.qualification_private_icp_sets WHERE set_id=20261009)<>10
     OR (SELECT count(*) FROM public.lab_arena_submissions WHERE round_id=v_round_id AND status='frozen' AND NOT is_king)<>69
     OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id AND stage=1 AND kind='execute' AND status='accepted' AND output_ref IS NOT NULL)<>10
     OR (SELECT count(DISTINCT assignment_id) FROM public.lab_arena_runs WHERE round_id=v_round_id AND stage=2 AND kind='execute')<>690
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r LEFT JOIN public.lab_arena_submissions s ON s.submission_id=r.submission_id AND s.round_id=r.round_id
        WHERE r.round_id=v_round_id AND r.kind='execute' AND (r.status NOT IN('accepted','failed') OR r.icp_position NOT BETWEEN 0 AND 9
          OR r.assignment_id IS DISTINCT FROM v_round_id||':'||r.submission_id||':'||r.stage||':'||r.icp_position
          OR (r.status='accepted' AND r.output_ref IS NULL) OR s.status IS DISTINCT FROM 'frozen'
          OR s.miner_hotkey IS DISTINCT FROM r.miner_hotkey OR (r.stage=2 AND (r.stage_generation<>9 OR s.is_king))))
     OR EXISTS (SELECT 1 FROM public.lab_arena_rounds WHERE round_id<>v_round_id AND configuration_doc->>'mode'='live' AND status NOT IN('open','committed','published','cancelled'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs s LEFT JOIN public.lab_arena_runs e ON e.run_id=s.scored_run_id WHERE s.round_id=v_round_id AND s.kind='score' AND (s.status NOT IN('accepted','failed','pending','leased','submitted') OR e.round_id IS DISTINCT FROM v_round_id OR e.kind IS DISTINCT FROM 'execute' OR e.status IS DISTINCT FROM 'accepted' OR e.submission_id IS DISTINCT FROM s.submission_id OR e.miner_hotkey IS DISTINCT FROM s.miner_hotkey OR e.stage IS DISTINCT FROM s.stage OR e.icp_position IS DISTINCT FROM s.icp_position))
     OR EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid IN('public.lab_arena_rounds'::regclass,'public.lab_arena_submissions'::regclass,'public.lab_arena_runs'::regclass,'public.lab_arena_ledger'::regclass,'public.lab_arena_trajectory_events'::regclass) AND NOT tgisinternal AND tgenabled<>'O')
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r JOIN public.lab_arena_rounds rr USING(round_id)
          WHERE r.round_id<>v_round_id AND rr.configuration_doc->>'mode'='live'
            AND r.status IN('leased','submitted')) THEN
    RAISE EXCEPTION 'Oct10 uniform hold444 terminal inventory or foreign live activity differs';
  END IF;
  IF v_control.operator_paused THEN
    RETURN; -- Same-owner replay does not mutate hold time or active judge leases.
  END IF;
  UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE,
    pause_reason=v_reason,actor_ref=v_actor,updated_at=clock_timestamp()
    WHERE singleton AND NOT operator_paused AND guard_generation=v_control.guard_generation;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count<>1 THEN RAISE EXCEPTION 'Oct10 uniform hold444 claim control changed'; END IF;
END;
$oct10_uniform_hold444$;
COMMIT;
