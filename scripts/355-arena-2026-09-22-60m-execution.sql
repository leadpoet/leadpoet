-- Move only the unstarted Sep22 execution deadline to the 60-minute profile.
-- Historical round documents and published scores remain untouched.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $sep22_60m_prerequisites$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::REGCLASS
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     )
     OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::REGPROCEDURE
     ), 'sha256'), 'hex') IS DISTINCT FROM
       '09f708260fdaeca445e3b7b98fcd8d367146de8833e7cb718802b7b64c8d42ae'
     OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_mark_uncertain(text,text,text,jsonb,integer)'::REGPROCEDURE
     ), 'sha256'), 'hex') IS DISTINCT FROM
       'ed7046ece252cd97252b2828c1087318d9da7b995d41e66b635190130bdb5b4e'
     OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::REGPROCEDURE
     ), 'sha256'), 'hex') IS DISTINCT FROM
       '4b96b984aa6c471588e84f08ab3a832300de4cb9e8cda386cd46a884fb431422'
     OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)'::REGPROCEDURE
     ), 'sha256'), 'hex') IS DISTINCT FROM
       'b8cb76f7838ae6f5daffaa801127808879f0d18281aec903f1a906137a6b3303' THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 355';
  END IF;
END;
$sep22_60m_prerequisites$;

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;

DO $sep22_60m$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_expected JSONB;
  v_count INTEGER;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-22'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep22 round is missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
       IS DISTINCT FROM 'arena-2026-09-22'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
       IS DISTINCT FROM '2026-09-22T00:00:00Z' THEN
    RAISE EXCEPTION 'Sep22 execution configuration identity differs';
  END IF;

  -- Replay is safe after the round starts, but no in-flight round may switch.
  IF v_round.configuration_doc ->> 'checkpoint_deadline_policy'
       = 'atomic_checkpoint_60m_v1'
     AND (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER = 3600
     AND (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER = 4500 THEN
    RETURN;
  END IF;
  IF v_round.configuration_doc ->> 'checkpoint_deadline_policy'
       IS DISTINCT FROM 'atomic_checkpoint_45m_v1'
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
       IS DISTINCT FROM 2700
     OR (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
       IS DISTINCT FROM 3600
     OR v_round.status IS DISTINCT FROM 'open'
     OR v_round.stage_generation IS DISTINCT FROM 0
     OR v_round.participants IS NOT NULL
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-22'
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = 'arena-2026-09-22'
         AND (status = 'frozen' OR frozen_at IS NOT NULL)
     ) THEN
    RAISE EXCEPTION 'Sep22 execution deadline requires an open, unfrozen, unstarted round';
  END IF;

  v_expected := v_round.configuration_doc || pg_catalog.jsonb_build_object(
    'checkpoint_deadline_policy', 'atomic_checkpoint_60m_v1',
    'icp_wall_clock_seconds', 3600,
    'lease_ttl_seconds', 4500
  );
  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET configuration_doc = v_expected,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-22';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep22 execution deadline update count differs';
  END IF;

  SELECT * INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-22';
  IF v_after.configuration_doc IS DISTINCT FROM v_expected
     OR (pg_catalog.to_jsonb(v_after) - 'configuration_doc' - 'updated_at')
       IS DISTINCT FROM
       (pg_catalog.to_jsonb(v_round) - 'configuration_doc' - 'updated_at')
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::REGCLASS
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'Sep22 execution deadline changed protected state';
  END IF;
END;
$sep22_60m$;

COMMIT;
