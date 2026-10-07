-- Recover the exact Oct07 infrastructure cutoff under full-round recovery scope.
-- All 80 retries use one uniformly extended schedule. Old rows and costs stay
-- immutable; the 10 interrupted executions restart, without checkpoint merge.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control', 0));
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_restart_claim_control IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $oct07_cutoff_419$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_config JSONB;
  v_before JSONB;
  v_triggers JSONB;
  v_runs_hash TEXT;
  v_submissions_hash TEXT;
  v_ledger_count BIGINT;
  v_ledger_max BIGINT;
  v_ledger_guard JSONB;
  v_target_hash TEXT;
  v_count INTEGER;
  v_old_round_hash CONSTANT TEXT := '917c007fcc697dbc52ecfb80650af90ae528b97c8c0df888c81024fbac290cd1';
  v_old_targets_hash CONSTANT TEXT := '4358942ab1c09fb04fd3d0a57d56e3f7eaf0be723fa481479a2b8703a9fa87ed';
  v_old_config_hash CONSTANT TEXT := 'bcc605522feb7e6d4be5ed94fd120b15a79f624db0191c2492e7b265d28878a3';
  v_new_config_hash CONSTANT TEXT := '12e3f0e058bb3d0c88847ff459b762e7f17b79c02214b3e143033cd8bc57d0ba';
BEGIN
  IF pg_catalog.current_setting('session_replication_role') IS DISTINCT FROM 'origin' THEN
    RAISE EXCEPTION 'Oct07 cutoff origin trigger execution is required';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger AS t
    JOIN pg_catalog.pg_proc AS f ON f.oid = t.tgfoid
    JOIN pg_catalog.pg_roles AS owner ON owner.oid = f.proowner
    WHERE t.tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND t.tgname = 'lab_arena_rounds_write_once'
      AND NOT t.tgisinternal AND t.tgenabled = 'O' AND t.tgtype = 27
      AND t.tgfoid = 'public.lab_arena_rounds_write_once_v1()'::pg_catalog.regprocedure
      AND pg_catalog.encode(extensions.digest(pg_catalog.pg_get_triggerdef(t.oid), 'sha256'), 'hex')
          = 'ab24e958f8cc240576d5e03e77de92182f83f720f352becc30ca52cea2cc7216'
      AND pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(f.oid), 'sha256'), 'hex')
          = '0e5347bb11dc6cbfe5949d4b2e43ae2ae56264713d5990d2ce9533352d440cb3'
      AND owner.rolname = 'lab_arena_owner'
      AND f.proacl::TEXT = '{lab_arena_owner=X/lab_arena_owner}'
      AND NOT f.prosecdef AND f.provolatile = 'v'
      AND f.proconfig = ARRAY['search_path=pg_catalog']
  ) THEN
    RAISE EXCEPTION 'Oct07 schedule write-once guard differs' USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(
      t.tgname,t.tgenabled,pg_catalog.pg_get_triggerdef(t.oid),
      pg_catalog.pg_get_functiondef(t.tgfoid)) ORDER BY t.tgname)
    INTO v_triggers FROM pg_catalog.pg_trigger t
    WHERE t.tgrelid='public.lab_arena_rounds'::regclass AND NOT t.tgisinternal;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_trigger
     WHERE tgrelid='public.lab_arena_rounds'::regclass
       AND tgname='lab_arena_rounds_operator_hold_transition_guard' AND tgenabled='O') THEN
    RAISE EXCEPTION 'Oct07 cutoff operator guard differs';
  END IF;
  -- Ledger rows cannot change: keep the exact append-only guard enabled,
  -- hold the ledger write lock, and reject any accidental append by count/max.
  -- Do not serialize provider-response payloads in this recovery transaction.
  SELECT pg_catalog.jsonb_build_array(
      pg_catalog.pg_get_triggerdef(t.oid),t.tgenabled,t.tgtype,
      pg_catalog.pg_get_functiondef(f.oid),owner.rolname,f.proacl::TEXT,
      f.proconfig,f.provolatile,f.prosecdef)
    INTO v_ledger_guard
    FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_proc f ON f.oid=t.tgfoid
      JOIN pg_catalog.pg_roles owner ON owner.oid=f.proowner
    WHERE t.tgrelid='public.lab_arena_ledger'::regclass
      AND t.tgname='lab_arena_ledger_append_only' AND NOT t.tgisinternal
      AND t.tgenabled='O' AND t.tgtype=27
      AND t.tgfoid='public.lab_arena_append_only_v1()'::regprocedure
      AND pg_catalog.encode(extensions.digest(pg_catalog.pg_get_triggerdef(t.oid),'sha256'),'hex')
        ='ab24bb7b732014633a7c03cbf13873f011c178bec0b6cad8364de0146de33b5f'
      AND pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(f.oid),'sha256'),'hex')
        ='2c8df4e925a3f343b80fa6ca6327debcc522a9709d60f1f10991c746e5c3524d'
      AND owner.rolname='lab_arena_owner'
      AND f.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
      AND f.proconfig=ARRAY['search_path=pg_catalog']
      AND f.provolatile='v' AND NOT f.prosecdef;
  IF v_ledger_guard IS NULL THEN
    RAISE EXCEPTION 'Oct07 cutoff ledger append-only guard differs';
  END IF;
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF NOT FOUND OR v_control.operator_paused OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> '' OR v_control.restart_scope <> ''
     OR v_control.restart_phase <> '' OR v_control.guard_expires_at IS NOT NULL
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'Oct07 cutoff restart or operator guard is active';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
    WHERE round_id='arena-2026-10-07' FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'Oct07 cutoff round missing'; END IF;

  CREATE TEMP TABLE oct07_cutoff_targets ON COMMIT DROP AS
    SELECT r.* FROM public.lab_arena_runs r
    WHERE r.round_id=v_round.round_id AND r.kind='execute' AND r.stage=2
      AND r.attempt=1 AND r.status='failed' AND r.terminal_cause='stage_closed'
      AND r.stage_generation=5;
  SELECT pg_catalog.count(*), pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.jsonb_agg(to_jsonb(t) ORDER BY run_id),'[]'::JSONB)::TEXT,
    'sha256'),'hex') INTO v_count,v_target_hash FROM oct07_cutoff_targets t;
  IF v_count <> 80 OR v_target_hash IS DISTINCT FROM v_old_targets_hash THEN
    RAISE EXCEPTION 'Oct07 cutoff exact targets differ';
  END IF;

  -- A replay permits normal progress of the 80 new attempts. It never updates
  -- their state, dates, leases, accepted outputs, or later round generations.
  IF EXISTS (SELECT 1 FROM public.lab_arena_runs r JOIN oct07_cutoff_targets t
      ON r.assignment_id=t.assignment_id WHERE r.attempt=2) THEN
    IF v_round.status NOT IN ('stage2','stage2_closed','stage2_scoring',
         'stage2_judged','scored','published','cancelled')
       OR v_round.status_generation < 9 OR v_round.stage_generation < 7
       OR pg_catalog.encode(extensions.digest(v_round.configuration_doc::TEXT,'sha256'),'hex')
            IS DISTINCT FROM v_new_config_hash
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs r
           JOIN oct07_cutoff_targets t ON r.assignment_id=t.assignment_id
           WHERE r.run_id=t.assignment_id||':2' AND r.round_id=t.round_id
             AND r.submission_id=t.submission_id AND r.miner_hotkey=t.miner_hotkey
             AND r.stage=t.stage AND r.icp_position=t.icp_position
             AND r.attempt=2 AND r.kind=t.kind AND r.stage_generation=7)
            IS DISTINCT FROM 80::BIGINT
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs r JOIN oct07_cutoff_targets t
            ON r.assignment_id=t.assignment_id WHERE r.attempt>2) THEN
      RAISE EXCEPTION 'Oct07 cutoff replay state differs';
    END IF;
    RETURN;
  END IF;

  IF pg_catalog.clock_timestamp() > '2026-10-07T15:00:00Z'::TIMESTAMPTZ
     OR pg_catalog.clock_timestamp() < '2026-10-07T14:00:04Z'::TIMESTAMPTZ THEN
    RAISE EXCEPTION 'Oct07 cutoff first apply outside bounded recovery window';
  END IF;
  IF v_round.status IS DISTINCT FROM 'cancelled'
     OR v_round.cancel_reason IS DISTINCT FROM 'execution_incomplete:stage2:80'
     OR v_round.status_generation IS DISTINCT FROM 8
     OR v_round.stage_generation IS DISTINCT FROM 6
     OR pg_catalog.encode(extensions.digest(to_jsonb(v_round)::TEXT,'sha256'),'hex')
          IS DISTINCT FROM v_old_round_hash
     OR pg_catalog.encode(extensions.digest(v_round.configuration_doc::TEXT,'sha256'),'hex')
          IS DISTINCT FROM v_old_config_hash
     OR (v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
          IS DISTINCT FROM 2
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.publication_doc IS NOT NULL OR v_round.published_at IS NOT NULL THEN
    RAISE EXCEPTION 'Oct07 cutoff cancelled preimage differs';
  END IF;
  IF EXISTS (SELECT 1 FROM oct07_cutoff_targets t WHERE t.result_doc IS NOT NULL
       OR t.output_ref IS NOT NULL OR t.run_id <> t.assignment_id||':1'
       OR t.per_icp_score IS NOT NULL)
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r JOIN oct07_cutoff_targets t
       ON r.assignment_id=t.assignment_id
       WHERE r.attempt>t.attempt OR r.status='accepted')
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id=v_round.round_id
       AND r.status IN ('pending','leased','submitted')) THEN
    RAISE EXCEPTION 'Oct07 cutoff target attempt or accepted state differs';
  END IF;
  SELECT pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.jsonb_agg(pg_catalog.encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex') ORDER BY run_id),'[]'::JSONB)::TEXT,'sha256'),'hex')
    INTO v_runs_hash FROM public.lab_arena_runs r WHERE round_id=v_round.round_id;
  SELECT pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.jsonb_agg(pg_catalog.encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') ORDER BY submission_id),'[]'::JSONB)::TEXT,'sha256'),'hex')
    INTO v_submissions_hash FROM public.lab_arena_submissions s WHERE round_id=v_round.round_id;
  SELECT (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger),
         (SELECT pg_catalog.max(entry_id) FROM public.lab_arena_ledger)
    INTO v_ledger_count,v_ledger_max;
  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round.round_id AND status='accepted') IS DISTINCT FROM 690::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round.round_id) IS DISTINCT FROM 775::BIGINT
     OR EXISTS (SELECT 1 FROM oct07_cutoff_targets t LEFT JOIN public.lab_arena_submissions s ON s.submission_id=t.submission_id AND s.round_id=t.round_id WHERE s.status IS DISTINCT FROM 'frozen' OR s.source_ref IS NULL) THEN
    RAISE EXCEPTION 'Oct07 cutoff accepted or frozen source state differs';
  END IF;
  v_before := to_jsonb(v_round)-'configuration_doc'-'status'-'status_generation'
    -'stage_generation'-'cancel_reason'-'updated_at';
  v_config := pg_catalog.jsonb_set(pg_catalog.jsonb_set(pg_catalog.jsonb_set(
    v_round.configuration_doc,
    '{schedule,stage_2_close}','"2026-10-07T17:00:02Z"'::JSONB,FALSE),
    '{schedule,final_scoring_close}','"2026-10-07T23:30:02Z"'::JSONB,FALSE),
    '{schedule,publication_deadline}','"2026-10-07T23:30:03Z"'::JSONB,FALSE);
  IF pg_catalog.encode(extensions.digest(v_config::TEXT,'sha256'),'hex')
       IS DISTINCT FROM v_new_config_hash
     OR (v_config-'schedule') IS DISTINCT FROM (v_round.configuration_doc-'schedule') THEN
    RAISE EXCEPTION 'Oct07 cutoff schedule postimage differs';
  END IF;
  INSERT INTO public.lab_arena_runs (
    run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
    attempt,status,lease_generation,stage_generation,kind,previous_runner_hotkey)
  SELECT assignment_id||':2',assignment_id,round_id,submission_id,miner_hotkey,
    stage,icp_position,2,'pending',lease_generation,7,kind,runner_hotkey
    FROM oct07_cutoff_targets;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  IF v_count <> 80 THEN RAISE EXCEPTION 'Oct07 cutoff partial retry insertion'; END IF;
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds SET status='stage2',status_generation=9,
    stage_generation=7,cancel_reason=NULL,configuration_doc=v_config
    WHERE round_id=v_round.round_id;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;
  SELECT * INTO v_after FROM public.lab_arena_rounds WHERE round_id=v_round.round_id;
  IF pg_catalog.clock_timestamp() > '2026-10-07T15:00:00Z'::TIMESTAMPTZ
     OR v_count<>1 OR v_after.status<>'stage2' OR v_after.status_generation<>9
     OR v_after.stage_generation<>7 OR v_after.cancel_reason IS NOT NULL
     OR v_after.configuration_doc IS DISTINCT FROM v_config
     OR (to_jsonb(v_after)-'configuration_doc'-'status'-'status_generation'
           -'stage_generation'-'cancel_reason'-'updated_at') IS DISTINCT FROM v_before
     OR (SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(
          t.tgname,t.tgenabled,pg_catalog.pg_get_triggerdef(t.oid),
          pg_catalog.pg_get_functiondef(t.tgfoid)) ORDER BY t.tgname)
          FROM pg_catalog.pg_trigger t
          WHERE t.tgrelid='public.lab_arena_rounds'::regclass AND NOT t.tgisinternal)
          IS DISTINCT FROM v_triggers
     OR (SELECT pg_catalog.encode(extensions.digest(COALESCE(
          pg_catalog.jsonb_agg(pg_catalog.encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex') ORDER BY run_id),'[]'::JSONB)::TEXT,'sha256'),'hex')
          FROM public.lab_arena_runs r WHERE round_id=v_round.round_id
            AND run_id NOT IN (SELECT assignment_id||':2' FROM oct07_cutoff_targets))
          IS DISTINCT FROM v_runs_hash
     OR (SELECT pg_catalog.encode(extensions.digest(COALESCE(
          pg_catalog.jsonb_agg(pg_catalog.encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') ORDER BY submission_id),'[]'::JSONB)::TEXT,'sha256'),'hex')
          FROM public.lab_arena_submissions s WHERE round_id=v_round.round_id)
          IS DISTINCT FROM v_submissions_hash
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger)
          IS DISTINCT FROM v_ledger_count
     OR (SELECT pg_catalog.max(entry_id) FROM public.lab_arena_ledger)
          IS DISTINCT FROM v_ledger_max
     OR (SELECT pg_catalog.jsonb_build_array(
          pg_catalog.pg_get_triggerdef(t.oid),t.tgenabled,t.tgtype,
          pg_catalog.pg_get_functiondef(f.oid),owner.rolname,f.proacl::TEXT,
          f.proconfig,f.provolatile,f.prosecdef)
          FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_proc f ON f.oid=t.tgfoid
            JOIN pg_catalog.pg_roles owner ON owner.oid=f.proowner
          WHERE t.tgrelid='public.lab_arena_ledger'::regclass
            AND t.tgname='lab_arena_ledger_append_only' AND NOT t.tgisinternal)
          IS DISTINCT FROM v_ledger_guard THEN
    RAISE EXCEPTION 'Oct07 cutoff immutable preservation or trigger restoration failed';
  END IF;
END;
$oct07_cutoff_419$;
COMMIT;
