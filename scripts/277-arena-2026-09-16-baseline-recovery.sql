-- Exact sixth baseline rerun recovery for arena-2026-09-16.
-- Render only after the fifth native rerun is terminal and all active provider
-- calls have closed. The migration seals that failed rerun; the operator RPC
-- archives it with its original source, selects the separately sealed latest
-- source for twenty fresh assignments, and keeps the policy, scoring, and
-- two-attempt contract unchanged. Deployed runtime identity remains
-- a separate release-operator precondition; this database seal does not claim it.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $requires_sep16_recovery$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_prepare_sep16_baseline_rerun_v1(bigint,text,text,text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_open_sep16_baseline_scoring_v1(text,smallint,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_rerun_audit'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_rerun_release_authority'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_recovery_authority'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_recovery_audit'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep16_recovery_archive_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_recovery275_authority'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_recovery275_audit'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep16_recovery272_archive_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep16_recovery273_archive_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep16_recovery275_archive_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_parallel_execution_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply current Arena migrations through 275 before 277';
  END IF;
END;
$requires_sep16_recovery$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_sep16_baseline_recovery277_authority (
  round_id TEXT PRIMARY KEY CHECK (round_id = 'arena-2026-09-16'),
  archive_round_id TEXT NOT NULL UNIQUE CHECK (
    archive_round_id = 'arena-2026-09-16-rerun275archive'
  ),
  archive_submission_id TEXT NOT NULL UNIQUE CHECK (
    archive_submission_id = 'baseline-2026-09-16-native-rerun275-archive'
  ),
  execute_namespace TEXT NOT NULL CHECK (execute_namespace = 'rerun277'),
  terminal_source_ref TEXT NOT NULL CHECK (
    terminal_source_ref = 'arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery275.tar.gz'
  ),
  terminal_source_size_bytes BIGINT NOT NULL CHECK (
    terminal_source_size_bytes = 556500
  ),
  terminal_source_sha256 TEXT NOT NULL CHECK (
    terminal_source_sha256 = 'c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104'
  ),
  terminal_source_commit TEXT NOT NULL CHECK (
    terminal_source_commit = '2b8386b835de02739ddef6c182bdeba43c6ad467'
  ),
  recovery_source_ref TEXT NOT NULL CHECK (
    recovery_source_ref = 'arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery277.tar.gz'
  ),
  recovery_source_size_bytes BIGINT NOT NULL CHECK (
    recovery_source_size_bytes = 561102
  ),
  recovery_source_sha256 TEXT NOT NULL CHECK (
    recovery_source_sha256 = '962ef646cd63e51e1f7fcdac85cc47a510389c42ce4ed787b685927bbf436425'
  ),
  recovery_source_commit TEXT NOT NULL CHECK (
    recovery_source_commit = 'f46e03cc8dba1d1b7ca0292bad0e1a71f849b4d7'
  ),
  bank_sha256 TEXT NOT NULL CHECK (
    bank_sha256 = '42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390'
  ),
  scorer_image_digest TEXT NOT NULL CHECK (
    scorer_image_digest = 'sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385'
  ),
  terminal_round_hash TEXT NOT NULL CHECK (
    terminal_round_hash = 'sha256:5f3687d016eefe35a51bc829734950ff742357bc1160457f10f351496dd7d03e'
  ),
  terminal_baseline_submission_hash TEXT NOT NULL CHECK (
    terminal_baseline_submission_hash = 'sha256:34431a508f9e5cba9c37792e968353356d4c2833adb59422368e54730296faeb'
  ),
  terminal_baseline_runs_hash TEXT NOT NULL CHECK (
    terminal_baseline_runs_hash = 'sha256:2d70f3b40393b831736b1a02e59a5acc3559eaa77edadaa6e404aa93bed95a27'
  ),
  terminal_baseline_ledger_hash TEXT NOT NULL CHECK (
    terminal_baseline_ledger_hash = 'sha256:62f70bea4f0b22588d514961fe7130f8e5f8d7ca79db386581cab12d355da415'
  ),
  terminal_baseline_run_count BIGINT NOT NULL CHECK (
    terminal_baseline_run_count = 22
  ),
  terminal_baseline_ledger_count BIGINT NOT NULL CHECK (
    terminal_baseline_ledger_count = 2691
  ),
  terminal_baseline_ledger_max_entry_id BIGINT NOT NULL CHECK (
    terminal_baseline_ledger_max_entry_id = 462143
  ),
  terminal_baseline_settled_microusd BIGINT NOT NULL CHECK (
    terminal_baseline_settled_microusd = 7384837
  ),
  terminal_baseline_uncertain_microusd BIGINT NOT NULL CHECK (
    terminal_baseline_uncertain_microusd = 451909
  ),
  challenger_runs_hash TEXT NOT NULL CHECK (
    challenger_runs_hash = 'sha256:518533eface43776db8b7a15871a684f976e379c106466cea97c8030561f3768'
  ),
  challenger_submissions_hash TEXT NOT NULL CHECK (
    challenger_submissions_hash = 'sha256:dd73d5d02a6d50b7790172f83149e89cadefb2b527dac2a8f6fbde57f482d89b'
  ),
  challenger_ledger_hash TEXT NOT NULL CHECK (
    challenger_ledger_hash = 'sha256:3c6aeabb33649104ac2b59621f5ec29324e10cbea987d89e972bc91d51444367'
  ),
  challenger_ledger_max_entry_id BIGINT NOT NULL CHECK (
    challenger_ledger_max_entry_id = 434377
  ),
  prior_rerun_audit_hash TEXT NOT NULL CHECK (
    prior_rerun_audit_hash = 'sha256:ed1412ab5cb0be2aa60a9903952fb67173f6038528f2384cec43b48ffa562c33'
  ),
  release_authority_hash TEXT NOT NULL CHECK (
    release_authority_hash = 'sha256:4edca10bfc6ad93d120688efdfa15f50e30bc02e51c416e78d5505e9a0766a7f'
  ),
  prior_recovery_authority_hash TEXT NOT NULL CHECK (
    prior_recovery_authority_hash = 'sha256:008dd1dd68730c4670bd10f03c8bf3c72ab2dc97524252deacb61bfb010e281b'
  ),
  prior_recovery_audit_hash TEXT NOT NULL CHECK (
    prior_recovery_audit_hash = 'sha256:a7bd1ef201627f0755623c99f2ad1c7ddf448293bc22d10d30ba0278b2dfe0b1'
  ),
  terminal_status_generation BIGINT NOT NULL CHECK (
    terminal_status_generation = 22
  ),
  terminal_stage_generation BIGINT NOT NULL CHECK (
    terminal_stage_generation = 18
  ),
  terminal_cancel_reason TEXT NOT NULL CHECK (
    terminal_cancel_reason = 'operator'
  ),
  old_schedule JSONB NOT NULL CHECK (
    old_schedule = $old_schedule${"benchmark_deadline":"2026-09-16T23:15:00Z","final_scoring_close":"2026-09-17T08:15:00Z","publication_deadline":"2026-09-17T08:45:00Z","stage_1_close":"2026-09-17T05:00:00Z","stage_1_scoring_close":"2026-09-17T06:30:00Z","stage_1_start":"2026-09-16T23:15:01Z","stage_2_close":"2026-09-17T06:45:00Z","stage_2_start":"2026-09-17T06:35:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"}$old_schedule$::JSONB
  ),
  forward_schedule JSONB NOT NULL CHECK (
    forward_schedule = $new_schedule${"benchmark_deadline":"2026-09-17T04:00:00Z","final_scoring_close":"2026-09-17T16:20:00Z","publication_deadline":"2026-09-17T16:50:00Z","stage_1_close":"2026-09-17T12:00:00Z","stage_1_scoring_close":"2026-09-17T14:00:00Z","stage_1_start":"2026-09-17T04:00:01Z","stage_2_close":"2026-09-17T14:20:00Z","stage_2_start":"2026-09-17T14:05:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"}$new_schedule$::JSONB
  ),
  authorized_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.lab_arena_sep16_baseline_recovery277_authority
  OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_sep16_baseline_recovery277_authority
  ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_sep16_baseline_recovery277_authority
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE TABLE IF NOT EXISTS public.lab_arena_sep16_baseline_recovery277_audit (
  round_id TEXT PRIMARY KEY CHECK (round_id = 'arena-2026-09-16'),
  archive_round_id TEXT NOT NULL UNIQUE,
  archive_submission_id TEXT NOT NULL UNIQUE,
  execute_namespace TEXT NOT NULL,
  terminal_round_doc JSONB NOT NULL,
  terminal_baseline_submission_doc JSONB NOT NULL,
  terminal_baseline_runs_hash TEXT NOT NULL,
  terminal_baseline_ledger_hash TEXT NOT NULL,
  terminal_baseline_run_count BIGINT NOT NULL,
  terminal_baseline_ledger_count BIGINT NOT NULL,
  terminal_baseline_ledger_max_entry_id BIGINT NOT NULL,
  terminal_baseline_settled_microusd BIGINT NOT NULL,
  terminal_baseline_uncertain_microusd BIGINT NOT NULL,
  challenger_runs_hash TEXT NOT NULL,
  challenger_submissions_hash TEXT NOT NULL,
  challenger_ledger_hash TEXT NOT NULL,
  challenger_ledger_max_entry_id BIGINT NOT NULL,
  prior_rerun_audit_hash TEXT NOT NULL,
  release_authority_hash TEXT NOT NULL,
  prior_recovery_authority_hash TEXT NOT NULL,
  prior_recovery_audit_hash TEXT NOT NULL,
  bank_sha256 TEXT NOT NULL,
  terminal_source_ref TEXT NOT NULL,
  terminal_source_size_bytes BIGINT NOT NULL,
  terminal_source_sha256 TEXT NOT NULL,
  terminal_source_commit TEXT NOT NULL,
  recovery_source_ref TEXT NOT NULL,
  recovery_source_size_bytes BIGINT NOT NULL,
  recovery_source_sha256 TEXT NOT NULL,
  recovery_source_commit TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.lab_arena_sep16_baseline_recovery277_audit
  OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_sep16_baseline_recovery277_audit ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_sep16_baseline_recovery277_audit
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

INSERT INTO public.lab_arena_sep16_baseline_recovery277_authority (
  round_id, archive_round_id, archive_submission_id, execute_namespace,
  terminal_source_ref, terminal_source_size_bytes, terminal_source_sha256,
  terminal_source_commit, recovery_source_ref, recovery_source_size_bytes,
  recovery_source_sha256, recovery_source_commit, bank_sha256,
  scorer_image_digest, terminal_round_hash,
  terminal_baseline_submission_hash, terminal_baseline_runs_hash,
  terminal_baseline_ledger_hash, terminal_baseline_run_count,
  terminal_baseline_ledger_count, terminal_baseline_ledger_max_entry_id,
  terminal_baseline_settled_microusd, terminal_baseline_uncertain_microusd,
  challenger_runs_hash,
  challenger_submissions_hash, challenger_ledger_hash,
  challenger_ledger_max_entry_id, prior_rerun_audit_hash,
  release_authority_hash, prior_recovery_authority_hash,
  prior_recovery_audit_hash, terminal_status_generation,
  terminal_stage_generation, terminal_cancel_reason, old_schedule,
  forward_schedule
) VALUES (
  'arena-2026-09-16', 'arena-2026-09-16-rerun275archive',
  'baseline-2026-09-16-native-rerun275-archive', 'rerun277',
  'arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery275.tar.gz',
  556500, 'c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104',
  '2b8386b835de02739ddef6c182bdeba43c6ad467',
  'arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery277.tar.gz',
  561102, '962ef646cd63e51e1f7fcdac85cc47a510389c42ce4ed787b685927bbf436425',
  'f46e03cc8dba1d1b7ca0292bad0e1a71f849b4d7',
  '42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390',
  'sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385',
  'sha256:5f3687d016eefe35a51bc829734950ff742357bc1160457f10f351496dd7d03e',
  'sha256:34431a508f9e5cba9c37792e968353356d4c2833adb59422368e54730296faeb',
  'sha256:2d70f3b40393b831736b1a02e59a5acc3559eaa77edadaa6e404aa93bed95a27',
  'sha256:62f70bea4f0b22588d514961fe7130f8e5f8d7ca79db386581cab12d355da415',
  22,
  2691,
  462143,
  7384837,
  451909,
  'sha256:518533eface43776db8b7a15871a684f976e379c106466cea97c8030561f3768',
  'sha256:dd73d5d02a6d50b7790172f83149e89cadefb2b527dac2a8f6fbde57f482d89b',
  'sha256:3c6aeabb33649104ac2b59621f5ec29324e10cbea987d89e972bc91d51444367',
  434377,
  'sha256:ed1412ab5cb0be2aa60a9903952fb67173f6038528f2384cec43b48ffa562c33',
  'sha256:4edca10bfc6ad93d120688efdfa15f50e30bc02e51c416e78d5505e9a0766a7f',
  'sha256:008dd1dd68730c4670bd10f03c8bf3c72ab2dc97524252deacb61bfb010e281b',
  'sha256:a7bd1ef201627f0755623c99f2ad1c7ddf448293bc22d10d30ba0278b2dfe0b1',
  22,
  18,
  'operator',
  $old_schedule${"benchmark_deadline":"2026-09-16T23:15:00Z","final_scoring_close":"2026-09-17T08:15:00Z","publication_deadline":"2026-09-17T08:45:00Z","stage_1_close":"2026-09-17T05:00:00Z","stage_1_scoring_close":"2026-09-17T06:30:00Z","stage_1_start":"2026-09-16T23:15:01Z","stage_2_close":"2026-09-17T06:45:00Z","stage_2_start":"2026-09-17T06:35:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"}$old_schedule$::JSONB,
  $new_schedule${"benchmark_deadline":"2026-09-17T04:00:00Z","final_scoring_close":"2026-09-17T16:20:00Z","publication_deadline":"2026-09-17T16:50:00Z","stage_1_close":"2026-09-17T12:00:00Z","stage_1_scoring_close":"2026-09-17T14:00:00Z","stage_1_start":"2026-09-17T04:00:01Z","stage_2_close":"2026-09-17T14:20:00Z","stage_2_start":"2026-09-17T14:05:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"}$new_schedule$::JSONB
) ON CONFLICT (round_id) DO NOTHING;

DO $seal_terminal_sep16_rerun$
DECLARE
  v_authority public.lab_arena_sep16_baseline_recovery277_authority%ROWTYPE;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_round_hash TEXT;
  v_submission_hash TEXT;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_challenger_runs_hash TEXT;
  v_challenger_submissions_hash TEXT;
  v_challenger_ledger_hash TEXT;
  v_prior_audit_hash TEXT;
  v_release_hash TEXT;
  v_prior_recovery_authority_hash TEXT;
  v_prior_recovery_audit_hash TEXT;
  v_execute_cost JSONB;
  v_score_cost JSONB;
  v_count BIGINT;
  v_ledger_count BIGINT;
  v_ledger_max BIGINT;
  v_settled BIGINT;
  v_uncertain BIGINT;
  v_challenger_ledger_max BIGINT;
BEGIN
  SELECT * INTO STRICT v_authority
  FROM public.lab_arena_sep16_baseline_recovery277_authority
  WHERE round_id = 'arena-2026-09-16' FOR UPDATE;
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_sep16_baseline_recovery277_audit
    WHERE round_id = v_authority.round_id
  ) THEN
    RETURN;
  END IF;
  SELECT * INTO STRICT v_round FROM public.lab_arena_rounds
  WHERE round_id = v_authority.round_id FOR SHARE;
  SELECT * INTO STRICT v_baseline FROM public.lab_arena_submissions
  WHERE submission_id = 'baseline-2026-09-16' FOR SHARE;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_round)::TEXT, 'sha256'), 'hex') INTO v_round_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_baseline)::TEXT, 'sha256'), 'hex') INTO v_submission_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*)
  INTO v_runs_hash, v_count FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_authority.round_id
    AND submission_id = 'baseline-2026-09-16';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*), COALESCE(pg_catalog.max(entry_id), 0),
    COALESCE(pg_catalog.sum(amount_microusd)
      FILTER (WHERE entry_kind = 'settlement'), 0),
    COALESCE(pg_catalog.sum(amount_microusd)
      FILTER (WHERE entry_kind = 'uncertain'), 0)
  INTO v_ledger_hash, v_ledger_count, v_ledger_max, v_settled, v_uncertain
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_authority.round_id
    AND submission_id = 'baseline-2026-09-16';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_challenger_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_authority.round_id
    AND submission_id <> 'baseline-2026-09-16';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY submission_id), ''), 'sha256'), 'hex')
  INTO v_challenger_submissions_hash
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_authority.round_id
    AND submission_id <> 'baseline-2026-09-16';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY entry_id), ''), 'sha256'), 'hex'),
    COALESCE(pg_catalog.max(entry_id), 0)
  INTO v_challenger_ledger_hash, v_challenger_ledger_max
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_authority.round_id
    AND submission_id <> 'baseline-2026-09-16';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_audit_hash
  FROM public.lab_arena_sep16_baseline_rerun_audit AS row_value
  WHERE round_id = v_authority.round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_release_hash
  FROM public.lab_arena_sep16_rerun_release_authority AS row_value
  WHERE round_id = v_authority.round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_recovery_authority_hash
  FROM public.lab_arena_sep16_baseline_recovery275_authority AS row_value
  WHERE round_id = v_authority.round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_recovery_audit_hash
  FROM public.lab_arena_sep16_baseline_recovery275_audit AS row_value
  WHERE round_id = v_authority.round_id;
  v_execute_cost := public.lab_arena__successful_call_cost_state(
    'baseline-2026-09-16', 'execute', NULL
  );
  v_score_cost := public.lab_arena__successful_call_cost_state(
    'baseline-2026-09-16', 'score', NULL
  );
  IF v_round_hash IS DISTINCT FROM v_authority.terminal_round_hash
     OR v_submission_hash IS DISTINCT FROM
          v_authority.terminal_baseline_submission_hash
     OR v_runs_hash IS DISTINCT FROM v_authority.terminal_baseline_runs_hash
     OR v_ledger_hash IS DISTINCT FROM v_authority.terminal_baseline_ledger_hash
     OR v_count IS DISTINCT FROM v_authority.terminal_baseline_run_count
     OR v_ledger_count IS DISTINCT FROM
          v_authority.terminal_baseline_ledger_count
     OR v_ledger_max IS DISTINCT FROM
          v_authority.terminal_baseline_ledger_max_entry_id
     OR v_settled IS DISTINCT FROM
          v_authority.terminal_baseline_settled_microusd
     OR v_uncertain IS DISTINCT FROM
          v_authority.terminal_baseline_uncertain_microusd
     OR v_challenger_runs_hash IS DISTINCT FROM v_authority.challenger_runs_hash
     OR v_challenger_submissions_hash IS DISTINCT FROM
          v_authority.challenger_submissions_hash
     OR v_challenger_ledger_hash IS DISTINCT FROM
          v_authority.challenger_ledger_hash
     OR v_challenger_ledger_max IS DISTINCT FROM
          v_authority.challenger_ledger_max_entry_id
     OR v_prior_audit_hash IS DISTINCT FROM v_authority.prior_rerun_audit_hash
     OR v_release_hash IS DISTINCT FROM v_authority.release_authority_hash
     OR v_prior_recovery_authority_hash IS DISTINCT FROM
          v_authority.prior_recovery_authority_hash
     OR v_prior_recovery_audit_hash IS DISTINCT FROM
          v_authority.prior_recovery_audit_hash
     OR v_round.status <> 'cancelled'
     OR v_round.status_generation IS DISTINCT FROM
          v_authority.terminal_status_generation
     OR v_round.stage_generation IS DISTINCT FROM
          v_authority.terminal_stage_generation
     OR v_round.cancel_reason IS DISTINCT FROM v_authority.terminal_cancel_reason
     OR v_round.configuration_doc -> 'schedule' IS DISTINCT FROM
          v_authority.old_schedule
     OR v_round.configuration_doc ->> 'max_attempts_per_assignment' <> '2'
     OR v_round.configuration_doc #>> '{call_quotas,openrouter}' <> '60'
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution' <> 'true'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy' <>
          'successful_calls_v1'
     OR v_round.benchmark_ref <> 'arena/arena-2026-09-16/benchmark.json'
     OR v_round.icp_set_date <> DATE '2026-09-15'
     OR v_baseline.round_id <> v_authority.round_id
     OR v_baseline.status <> 'frozen' OR NOT v_baseline.is_king
     OR v_baseline.source_ref <> v_authority.terminal_source_ref
     OR v_baseline.source_size_bytes <>
          v_authority.terminal_source_size_bytes
     OR v_baseline.submission_doc ->> 'source_sha256' <>
          v_authority.terminal_source_sha256
     OR v_baseline.submission_doc ->> 'source_commit' <>
          v_authority.terminal_source_commit
     OR public.lab_arena_sep16_challenger_seals_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery272_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery273_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery275_archive_valid_v1() IS NOT TRUE
     OR COALESCE((v_execute_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_execute_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-16-rerun269archive'
           AND submission_id = 'baseline-2026-09-16-native-rerun269-archive'
           AND entry_kind = 'settlement'
           AND entry_doc ->>
                 'sep16_generic_http_catalog_reconciliation' = 'true') <> 2
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_authority.round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS candidate
       WHERE candidate.round_id = v_authority.round_id
         AND candidate.submission_id = 'baseline-2026-09-16'
         AND candidate.call_identity IS NOT NULL
         AND candidate.entry_id = (
           SELECT pg_catalog.max(later.entry_id)
           FROM public.lab_arena_ledger AS later
           WHERE later.call_identity = candidate.call_identity
         )
         AND candidate.entry_kind IN ('reservation', 'dispatch')
     )
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_authority.round_id
           AND submission_id = 'baseline-2026-09-16'
           AND kind = 'execute' AND assignment_id LIKE '%:rerun275') <> 20
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_authority.round_id
         AND submission_id = 'baseline-2026-09-16'
         AND (kind <> 'execute' OR attempt NOT IN (1, 2)
              OR assignment_id NOT LIKE '%:rerun275')
     ) THEN
    RAISE EXCEPTION 'Sep16 terminal native rerun seal differs before recovery277'
      USING ERRCODE = '55000';
  END IF;
END;
$seal_terminal_sep16_rerun$;

CREATE OR REPLACE FUNCTION public.lab_arena_sep16_recovery277_archive_valid_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep16_recovery277_archive_valid$
DECLARE
  v_audit public.lab_arena_sep16_baseline_recovery277_audit%ROWTYPE;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_submission_doc JSONB;
BEGIN
  SELECT * INTO v_audit
  FROM public.lab_arena_sep16_baseline_recovery277_audit
  WHERE round_id = 'arena-2026-09-16';
  IF NOT FOUND THEN
    RETURN FALSE;
  END IF;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_audit.archive_round_id
    AND submission_id = v_audit.archive_submission_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_audit.archive_round_id
    AND submission_id = v_audit.archive_submission_id
    AND entry_id <= v_audit.terminal_baseline_ledger_max_entry_id;
  SELECT pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id'
  INTO v_submission_doc FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_audit.archive_round_id
    AND submission_id = v_audit.archive_submission_id;
  RETURN v_runs_hash IS NOT DISTINCT FROM v_audit.terminal_baseline_runs_hash
     AND v_ledger_hash IS NOT DISTINCT FROM v_audit.terminal_baseline_ledger_hash
     AND v_submission_doc IS NOT DISTINCT FROM
          (v_audit.terminal_baseline_submission_doc - 'round_id' - 'submission_id')
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
          WHERE round_id = v_audit.archive_round_id
            AND submission_id = v_audit.archive_submission_id)
         = v_audit.terminal_baseline_run_count
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
          WHERE round_id = v_audit.archive_round_id
            AND submission_id = v_audit.archive_submission_id
            AND entry_id <= v_audit.terminal_baseline_ledger_max_entry_id)
         = v_audit.terminal_baseline_ledger_count
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_audit.archive_round_id AND status = 'cancelled'
         AND rewards_enabled IS FALSE
         AND cancel_reason = 'authorized_sep16_failed_native_rerun_archive'
     )
     AND NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS late
       WHERE late.round_id = v_audit.archive_round_id
         AND late.submission_id = v_audit.archive_submission_id
         AND late.entry_id > v_audit.terminal_baseline_ledger_max_entry_id
         AND (
           late.entry_kind <> 'settlement'
           OR late.entry_doc ->> 'late_reconciliation' IS DISTINCT FROM 'true'
           OR NOT EXISTS (
             SELECT 1 FROM public.lab_arena_ledger AS original
             WHERE original.entry_id <=
                     v_audit.terminal_baseline_ledger_max_entry_id
               AND original.entry_kind = 'uncertain'
               AND original.round_id = late.round_id
               AND original.submission_id = late.submission_id
               AND original.run_id = late.run_id
               AND original.miner_hotkey = late.miner_hotkey
               AND original.stage = late.stage
               AND original.call_identity = late.call_identity
               AND original.provider = late.provider
               AND original.operation_id = late.operation_id
               AND original.funding_source IS NOT DISTINCT FROM
                     late.funding_source
               AND original.entry_id =
                   (late.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
               AND (
                 (late.provider = 'deepline'
                  AND late.entry_doc ->> 'deepline_delayed_reconciliation' = 'true')
                 OR (late.provider = 'openrouter'
                  AND late.entry_doc ->> 'openrouter_delayed_reconciliation' = 'true')
               )
           )
         )
     );
END;
$sep16_recovery277_archive_valid$;
ALTER FUNCTION public.lab_arena_sep16_recovery277_archive_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep16_recovery277_archive_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep16_baseline_recovery277_v1(
  p_source_size_bytes BIGINT,
  p_source_sha256 TEXT,
  p_source_commit TEXT,
  p_forward_schedule JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $prepare_sep16_baseline_recovery277$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-16';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-16';
  v_authority public.lab_arena_sep16_baseline_recovery277_authority%ROWTYPE;
  v_audit public.lab_arena_sep16_baseline_recovery277_audit%ROWTYPE;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_archive_configuration JSONB;
  v_new_configuration JSONB;
  v_new_participants JSONB;
  v_execute_cost JSONB;
  v_score_cost JSONB;
  v_stage SMALLINT;
  v_position INTEGER;
  v_assignment TEXT;
  v_count BIGINT;
  v_round_hash TEXT;
  v_submission_hash TEXT;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_challenger_runs_hash TEXT;
  v_challenger_submissions_hash TEXT;
  v_challenger_ledger_hash TEXT;
  v_prior_audit_hash TEXT;
  v_release_hash TEXT;
  v_prior_recovery_authority_hash TEXT;
  v_prior_recovery_audit_hash TEXT;
  v_ledger_count BIGINT;
  v_ledger_max BIGINT;
  v_settled BIGINT;
  v_uncertain BIGINT;
  v_challenger_ledger_max BIGINT;
BEGIN
  PERFORM pg_catalog.set_config('lock_timeout', '5s', TRUE);
  PERFORM pg_catalog.set_config('statement_timeout', '120s', TRUE);
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-16-native-baseline-v1', 0)
  );
  SELECT * INTO STRICT v_authority
  FROM public.lab_arena_sep16_baseline_recovery277_authority
  WHERE round_id = v_round_id FOR SHARE;
  IF p_source_size_bytes IS DISTINCT FROM
       v_authority.recovery_source_size_bytes
     OR p_source_sha256 IS DISTINCT FROM v_authority.recovery_source_sha256
     OR p_source_commit IS DISTINCT FROM v_authority.recovery_source_commit THEN
    RAISE EXCEPTION 'Sep16 recovery277 source differs from authority'
      USING ERRCODE = '22023';
  END IF;
  IF p_forward_schedule IS DISTINCT FROM v_authority.forward_schedule THEN
    RAISE EXCEPTION 'Sep16 recovery277 schedule differs from authority'
      USING ERRCODE = '22023';
  END IF;
  LOCK TABLE public.lab_arena_sep16_baseline_recovery277_audit
    IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
  SELECT * INTO v_audit
  FROM public.lab_arena_sep16_baseline_recovery277_audit
  WHERE round_id = v_round_id FOR UPDATE;
  IF FOUND THEN
    IF v_audit.archive_round_id <> v_authority.archive_round_id
       OR v_audit.archive_submission_id <> v_authority.archive_submission_id
       OR v_audit.execute_namespace <> v_authority.execute_namespace
       OR v_audit.terminal_baseline_runs_hash <>
            v_authority.terminal_baseline_runs_hash
       OR v_audit.terminal_baseline_ledger_hash <>
            v_authority.terminal_baseline_ledger_hash
       OR v_audit.prior_rerun_audit_hash <> v_authority.prior_rerun_audit_hash
       OR v_audit.release_authority_hash <> v_authority.release_authority_hash
       OR v_audit.prior_recovery_authority_hash <>
            v_authority.prior_recovery_authority_hash
       OR v_audit.prior_recovery_audit_hash <>
            v_authority.prior_recovery_audit_hash
       OR v_audit.terminal_source_ref <> v_authority.terminal_source_ref
       OR v_audit.terminal_source_size_bytes <>
            v_authority.terminal_source_size_bytes
       OR v_audit.terminal_source_sha256 <> v_authority.terminal_source_sha256
       OR v_audit.terminal_source_commit <> v_authority.terminal_source_commit
       OR v_audit.recovery_source_ref <> v_authority.recovery_source_ref
       OR v_audit.recovery_source_size_bytes <>
            v_authority.recovery_source_size_bytes
       OR v_audit.recovery_source_sha256 <> v_authority.recovery_source_sha256
       OR v_audit.recovery_source_commit <> v_authority.recovery_source_commit
       OR public.lab_arena_sep16_recovery277_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep16_recovery_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep16_recovery272_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep16_recovery273_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep16_recovery275_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep16_challenger_seals_valid_v1() IS NOT TRUE
       OR (SELECT configuration_doc FROM public.lab_arena_rounds
           WHERE round_id = v_round_id) IS DISTINCT FROM
          pg_catalog.jsonb_set(
            pg_catalog.jsonb_set(
              v_audit.terminal_round_doc -> 'configuration_doc',
              '{schedule}', v_authority.forward_schedule, FALSE
            ),
            '{call_quotas,openrouter}', '200'::JSONB, FALSE
          )
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_submissions
         WHERE submission_id = v_baseline_id AND round_id = v_round_id
           AND source_ref = v_authority.recovery_source_ref
           AND source_size_bytes = v_authority.recovery_source_size_bytes
           AND submission_doc ->> 'source_ref' =
                 v_authority.recovery_source_ref
           AND submission_doc ->> 'source_sha256' =
                 v_authority.recovery_source_sha256
           AND submission_doc ->> 'source_commit' =
                 v_authority.recovery_source_commit
       )
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_rounds AS current_round,
           LATERAL pg_catalog.jsonb_array_elements(
             current_round.participants
           ) AS item
         WHERE current_round.round_id = v_round_id
           AND item ->> 'submission_id' = v_baseline_id
           AND item ->> 'source_ref' = v_authority.recovery_source_ref
           AND (item ->> 'source_size_bytes')::BIGINT =
                 v_authority.recovery_source_size_bytes
       )
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND submission_id = v_baseline_id
             AND kind = 'execute' AND assignment_id LIKE '%:rerun277') <> 20 THEN
      RAISE EXCEPTION 'Sep16 recovery277 replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'round_id', v_round_id,
      'baseline_execute_assignments', 20
    );
  END IF;
  SELECT * INTO STRICT v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  SELECT * INTO STRICT v_baseline FROM public.lab_arena_submissions
  WHERE submission_id = v_baseline_id FOR UPDATE;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_round)::TEXT, 'sha256'), 'hex') INTO v_round_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_baseline)::TEXT, 'sha256'), 'hex') INTO v_submission_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*)
  INTO v_runs_hash, v_count FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*), COALESCE(pg_catalog.max(entry_id), 0),
    COALESCE(pg_catalog.sum(amount_microusd)
      FILTER (WHERE entry_kind = 'settlement'), 0),
    COALESCE(pg_catalog.sum(amount_microusd)
      FILTER (WHERE entry_kind = 'uncertain'), 0)
  INTO v_ledger_hash, v_ledger_count, v_ledger_max, v_settled, v_uncertain
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_challenger_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY submission_id), ''), 'sha256'), 'hex')
  INTO v_challenger_submissions_hash
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY entry_id), ''), 'sha256'), 'hex'),
    COALESCE(pg_catalog.max(entry_id), 0)
  INTO v_challenger_ledger_hash, v_challenger_ledger_max
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_audit_hash
  FROM public.lab_arena_sep16_baseline_rerun_audit AS row_value
  WHERE round_id = v_round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_release_hash
  FROM public.lab_arena_sep16_rerun_release_authority AS row_value
  WHERE round_id = v_round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_recovery_authority_hash
  FROM public.lab_arena_sep16_baseline_recovery275_authority AS row_value
  WHERE round_id = v_round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_recovery_audit_hash
  FROM public.lab_arena_sep16_baseline_recovery275_audit AS row_value
  WHERE round_id = v_round_id;
  v_execute_cost := public.lab_arena__successful_call_cost_state(
    v_baseline_id, 'execute', NULL
  );
  v_score_cost := public.lab_arena__successful_call_cost_state(
    v_baseline_id, 'score', NULL
  );
  IF v_round_hash <> v_authority.terminal_round_hash
     OR v_submission_hash <> v_authority.terminal_baseline_submission_hash
     OR v_runs_hash <> v_authority.terminal_baseline_runs_hash
     OR v_ledger_hash <> v_authority.terminal_baseline_ledger_hash
     OR v_count <> v_authority.terminal_baseline_run_count
     OR v_ledger_count <> v_authority.terminal_baseline_ledger_count
     OR v_ledger_max <> v_authority.terminal_baseline_ledger_max_entry_id
     OR v_settled <> v_authority.terminal_baseline_settled_microusd
     OR v_uncertain <> v_authority.terminal_baseline_uncertain_microusd
     OR v_challenger_runs_hash <> v_authority.challenger_runs_hash
     OR v_challenger_submissions_hash <> v_authority.challenger_submissions_hash
     OR v_challenger_ledger_hash <> v_authority.challenger_ledger_hash
     OR v_challenger_ledger_max <> v_authority.challenger_ledger_max_entry_id
     OR v_prior_audit_hash <> v_authority.prior_rerun_audit_hash
     OR v_release_hash <> v_authority.release_authority_hash
     OR v_prior_recovery_authority_hash <>
          v_authority.prior_recovery_authority_hash
     OR v_prior_recovery_audit_hash <>
          v_authority.prior_recovery_audit_hash
     OR v_round.status <> 'cancelled'
     OR v_round.status_generation <> v_authority.terminal_status_generation
     OR v_round.stage_generation <> v_authority.terminal_stage_generation
     OR v_round.cancel_reason <> v_authority.terminal_cancel_reason
     OR v_round.configuration_doc -> 'schedule' <> v_authority.old_schedule
     OR COALESCE((v_execute_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_execute_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR public.lab_arena_sep16_challenger_seals_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery272_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery273_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery275_archive_valid_v1() IS NOT TRUE
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-16-rerun269archive'
           AND submission_id = 'baseline-2026-09-16-native-rerun269-archive'
           AND entry_kind = 'settlement'
           AND entry_doc ->>
                 'sep16_generic_http_catalog_reconciliation' = 'true') <> 2
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND status IN ('pending', 'leased', 'submitted')
     ) THEN
    RAISE EXCEPTION 'Sep16 terminal rerun changed before recovery277'
      USING ERRCODE = '55000';
  END IF;
  INSERT INTO public.lab_arena_sep16_baseline_recovery277_audit (
    round_id, archive_round_id, archive_submission_id, execute_namespace,
    terminal_round_doc, terminal_baseline_submission_doc,
    terminal_baseline_runs_hash, terminal_baseline_ledger_hash,
    terminal_baseline_run_count, terminal_baseline_ledger_count,
    terminal_baseline_ledger_max_entry_id,
    terminal_baseline_settled_microusd,
    terminal_baseline_uncertain_microusd,
    challenger_runs_hash, challenger_submissions_hash, challenger_ledger_hash,
    challenger_ledger_max_entry_id, prior_rerun_audit_hash,
    release_authority_hash, prior_recovery_authority_hash,
    prior_recovery_audit_hash, bank_sha256,
    terminal_source_ref, terminal_source_size_bytes,
    terminal_source_sha256, terminal_source_commit,
    recovery_source_ref, recovery_source_size_bytes,
    recovery_source_sha256, recovery_source_commit
  ) VALUES (
    v_round_id, v_authority.archive_round_id, v_authority.archive_submission_id,
    v_authority.execute_namespace, pg_catalog.to_jsonb(v_round),
    pg_catalog.to_jsonb(v_baseline), v_runs_hash, v_ledger_hash, v_count,
    v_ledger_count, v_ledger_max, v_settled, v_uncertain,
    v_challenger_runs_hash,
    v_challenger_submissions_hash, v_challenger_ledger_hash,
    v_challenger_ledger_max, v_prior_audit_hash, v_release_hash,
    v_prior_recovery_authority_hash, v_prior_recovery_audit_hash,
    v_authority.bank_sha256,
    v_authority.terminal_source_ref, v_authority.terminal_source_size_bytes,
    v_authority.terminal_source_sha256, v_authority.terminal_source_commit,
    v_authority.recovery_source_ref, v_authority.recovery_source_size_bytes,
    v_authority.recovery_source_sha256, v_authority.recovery_source_commit
  );
  v_archive_configuration := v_round.configuration_doc ||
    pg_catalog.jsonb_build_object(
      'round_id', v_authority.archive_round_id,
      'rewards_enabled', FALSE
    );
  INSERT INTO public.lab_arena_rounds (
    round_id, status, configuration_doc, rewards_enabled, cancel_reason
  ) VALUES (
    v_authority.archive_round_id, 'cancelled', v_archive_configuration, FALSE,
    'authorized_sep16_failed_native_rerun_archive'
  );
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(v_baseline) || pg_catalog.jsonb_build_object(
      'submission_id', v_authority.archive_submission_id,
      'round_id', v_authority.archive_round_id
    )
  )).*;
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
  UPDATE public.lab_arena_runs
  SET round_id = v_authority.archive_round_id,
      submission_id = v_authority.archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> v_authority.terminal_baseline_run_count THEN
    RAISE EXCEPTION 'Sep16 recovery277 archive run count differs';
  END IF;
  UPDATE public.lab_arena_ledger
  SET round_id = v_authority.archive_round_id,
      submission_id = v_authority.archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> v_authority.terminal_baseline_ledger_count THEN
    RAISE EXCEPTION 'Sep16 recovery277 archive ledger count differs';
  END IF;
  UPDATE public.lab_arena_submissions
  SET source_ref = v_authority.recovery_source_ref,
      source_size_bytes = v_authority.recovery_source_size_bytes,
      submission_doc = submission_doc || pg_catalog.jsonb_build_object(
        'source_ref', v_authority.recovery_source_ref,
        'source_size_bytes', v_authority.recovery_source_size_bytes,
        'source_sha256', v_authority.recovery_source_sha256,
        'source_commit', v_authority.recovery_source_commit
      )
  WHERE submission_id = v_baseline_id AND round_id = v_round_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep16 recovery277 baseline source switch failed';
  END IF;
  v_new_configuration := pg_catalog.jsonb_set(
    pg_catalog.jsonb_set(
      v_round.configuration_doc, '{schedule}', p_forward_schedule, FALSE
    ),
    '{call_quotas,openrouter}', '200'::JSONB, FALSE
  );
  IF (v_new_configuration - 'schedule' - 'call_quotas') IS DISTINCT FROM
       (v_round.configuration_doc - 'schedule' - 'call_quotas')
     OR ((v_new_configuration -> 'call_quotas') - 'openrouter') IS DISTINCT FROM
        ((v_round.configuration_doc -> 'call_quotas') - 'openrouter')
     OR v_round.configuration_doc #>> '{call_quotas,openrouter}' <> '60'
     OR v_new_configuration #>> '{call_quotas,openrouter}' <> '200' THEN
    RAISE EXCEPTION 'Sep16 recovery277 changed frozen configuration';
  END IF;
  SELECT pg_catalog.jsonb_agg(
    CASE WHEN item ->> 'submission_id' = v_baseline_id
      THEN item || pg_catalog.jsonb_build_object(
        'source_ref', v_authority.recovery_source_ref,
        'source_size_bytes', v_authority.recovery_source_size_bytes
      )
      ELSE item END
    ORDER BY ordinal
  ) INTO v_new_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants)
    WITH ORDINALITY AS entries(item, ordinal);
  UPDATE public.lab_arena_rounds
  SET status = 'stage1',
      status_generation = v_authority.terminal_status_generation + 1,
      stage_generation = v_authority.terminal_stage_generation + 1,
      configuration_doc = v_new_configuration,
      participants = v_new_participants,
      stage1_scoring_plan_doc = NULL, stage2_scoring_plan_doc = NULL,
      finalists = NULL, publication_doc = NULL, king_hotkey = NULL,
      published_at = NULL, cancel_reason = NULL
  WHERE round_id = v_round_id;
  FOR v_position IN 0 .. 19 LOOP
    v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;
    v_assignment := v_round_id || ':' || v_baseline_id || ':' ||
      v_stage::TEXT || ':' || v_position::TEXT || ':rerun277';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, status, stage_generation
    ) VALUES (
      v_assignment || ':1', v_assignment, v_round_id, v_baseline_id,
      v_baseline.miner_hotkey, v_stage, v_position, 1, 'pending',
      v_authority.terminal_stage_generation + 1
    );
  END LOOP;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;
  IF NOT public.lab_arena_sep16_recovery277_archive_valid_v1()
     OR public.lab_arena_sep16_recovery_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery272_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery273_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery275_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_challenger_seals_valid_v1() IS NOT TRUE
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-16-rerun269archive'
           AND submission_id = 'baseline-2026-09-16-native-rerun269-archive'
           AND entry_kind = 'settlement'
           AND entry_doc ->>
                 'sep16_generic_http_catalog_reconciliation' = 'true') <> 2
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE submission_id = v_baseline_id AND round_id = v_round_id
         AND source_ref = v_authority.recovery_source_ref
         AND source_size_bytes = v_authority.recovery_source_size_bytes
         AND submission_doc ->> 'source_ref' = v_authority.recovery_source_ref
         AND submission_doc ->> 'source_sha256' =
               v_authority.recovery_source_sha256
         AND submission_doc ->> 'source_commit' =
               v_authority.recovery_source_commit
     )
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND status = 'pending'
           AND assignment_id LIKE '%:rerun277') <> 20
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND kind = 'execute'
           AND stage IN (1, 2)) <> 100
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_round_id AND status = 'stage1'
         AND status_generation = v_authority.terminal_status_generation + 1
         AND stage_generation = v_authority.terminal_stage_generation + 1
         AND configuration_doc -> 'schedule' = p_forward_schedule
         AND configuration_doc ->> 'max_attempts_per_assignment' = '2'
         AND configuration_doc #>> '{call_quotas,openrouter}' = '200'
         AND EXISTS (
           SELECT 1 FROM pg_catalog.jsonb_array_elements(participants) AS item
           WHERE item ->> 'submission_id' = v_baseline_id
             AND item ->> 'source_ref' = v_authority.recovery_source_ref
             AND (item ->> 'source_size_bytes')::BIGINT =
                   v_authority.recovery_source_size_bytes
         )
     ) THEN
    RAISE EXCEPTION 'Sep16 recovery277 post-archive verification differs: %',
      pg_catalog.jsonb_build_object(
        'archive_valid', public.lab_arena_sep16_recovery277_archive_valid_v1(),
        'archive_round_valid', EXISTS (SELECT 1 FROM public.lab_arena_rounds
          WHERE round_id = v_authority.archive_round_id AND status = 'cancelled'
            AND rewards_enabled IS FALSE
            AND cancel_reason = 'authorized_sep16_failed_native_rerun_archive'),
        'archive_run_count', (SELECT pg_catalog.count(*)
          FROM public.lab_arena_runs WHERE round_id = v_authority.archive_round_id
            AND submission_id = v_authority.archive_submission_id),
        'archive_ledger_count', (SELECT pg_catalog.count(*)
          FROM public.lab_arena_ledger
          WHERE round_id = v_authority.archive_round_id
            AND submission_id = v_authority.archive_submission_id
            AND entry_id <= v_authority.terminal_baseline_ledger_max_entry_id),
        'archive_submission_equal', (SELECT
          (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')
            IS NOT DISTINCT FROM
          (v_audit.terminal_baseline_submission_doc - 'round_id' - 'submission_id')
          FROM public.lab_arena_submissions AS row_value
          WHERE round_id = v_authority.archive_round_id
            AND submission_id = v_authority.archive_submission_id),
        'challengers_valid', public.lab_arena_sep16_challenger_seals_valid_v1(),
        'prior_archive_valid',
          public.lab_arena_sep16_recovery_archive_valid_v1(),
        'prior_recovery272_archive_valid',
          public.lab_arena_sep16_recovery272_archive_valid_v1(),
        'prior_recovery273_archive_valid',
          public.lab_arena_sep16_recovery273_archive_valid_v1(),
        'prior_recovery275_archive_valid',
          public.lab_arena_sep16_recovery275_archive_valid_v1(),
        'pending_recovery', (SELECT pg_catalog.count(*)
          FROM public.lab_arena_runs WHERE round_id = v_round_id
            AND submission_id = v_baseline_id AND kind = 'execute'
            AND status = 'pending' AND assignment_id LIKE '%:rerun277'),
        'execute_assignments', (SELECT pg_catalog.count(DISTINCT assignment_id)
          FROM public.lab_arena_runs WHERE round_id = v_round_id
            AND kind = 'execute' AND stage IN (1, 2)),
        'round_status', (SELECT status FROM public.lab_arena_rounds
          WHERE round_id = v_round_id),
        'status_generation', (SELECT status_generation
          FROM public.lab_arena_rounds WHERE round_id = v_round_id),
        'stage_generation', (SELECT stage_generation
          FROM public.lab_arena_rounds WHERE round_id = v_round_id)
      );
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'prepared', 'round_id', v_round_id,
    'baseline_execute_assignments', 20,
    'archived_runs', v_authority.terminal_baseline_run_count,
    'archived_ledger_entries', v_authority.terminal_baseline_ledger_count,
    'execute_namespace', v_authority.execute_namespace,
    'openrouter_calls_per_icp', 200
  );
END;
$prepare_sep16_baseline_recovery277$;
ALTER FUNCTION public.lab_arena_prepare_sep16_baseline_recovery277_v1(
  BIGINT, TEXT, TEXT, JSONB
)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep16_baseline_recovery277_v1(
  BIGINT, TEXT, TEXT, JSONB
)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep16_baseline_recovery277_v1(
  BIGINT, TEXT, TEXT, JSONB
)
  TO lab_arena_service;

-- Change only the exact Sep16 one-off namespace. Generic scoring, retries,
-- aggregation, publication, promotion, and weights remain unchanged.
DO $replace_sep16_recovery277_namespaces$
DECLARE
  v_proc REGPROCEDURE;
  v_definition TEXT;
  v_old_count INTEGER;
  v_new_count INTEGER;
  v_expected_count INTEGER;
BEGIN
  FOREACH v_proc IN ARRAY ARRAY[
    'public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb)'::REGPROCEDURE,
    'public.lab_arena_sep16_rerun_score_namespace_guard_v1()'::REGPROCEDURE,
    'public.lab_arena_open_sep16_baseline_scoring_v1(text,smallint,jsonb)'::REGPROCEDURE,
    'public.lab_arena_sep16_rerun_publication_guard_v1()'::REGPROCEDURE
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(v_proc) INTO STRICT v_definition;
    v_expected_count := CASE v_proc
      WHEN 'public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb)'::REGPROCEDURE
        THEN 1
      WHEN 'public.lab_arena_sep16_rerun_score_namespace_guard_v1()'::REGPROCEDURE
        THEN 1
      WHEN 'public.lab_arena_open_sep16_baseline_scoring_v1(text,smallint,jsonb)'::REGPROCEDURE
        THEN 3
      WHEN 'public.lab_arena_sep16_rerun_publication_guard_v1()'::REGPROCEDURE
        THEN 6
      ELSE NULL
    END;
    v_old_count := (
      pg_catalog.length(v_definition) -
      pg_catalog.length(pg_catalog.replace(v_definition, ':rerun275', ''))
    ) / pg_catalog.length(':rerun275');
    v_new_count := (
      pg_catalog.length(v_definition) -
      pg_catalog.length(pg_catalog.replace(v_definition, ':rerun277', ''))
    ) / pg_catalog.length(':rerun277');
    IF v_old_count = v_expected_count AND v_new_count = 0 THEN
      EXECUTE pg_catalog.replace(v_definition, ':rerun275', ':rerun277');
    ELSIF NOT (
      v_old_count = 0 AND v_new_count = v_expected_count
    ) THEN
      RAISE EXCEPTION 'Sep16 recovery namespace function shape differs: %', v_proc
        USING ERRCODE = '55000';
    END IF;
  END LOOP;
END;
$replace_sep16_recovery277_namespaces$;

DO $bind_sep16_recovery277_archive_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()
       OR NOT public.lab_arena_sep16_recovery272_archive_valid_v1()
       OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()
       OR NOT public.lab_arena_sep16_recovery275_archive_valid_v1()$old$;
  v_new TEXT := $new$OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()
       OR NOT public.lab_arena_sep16_recovery272_archive_valid_v1()
       OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()
       OR NOT public.lab_arena_sep16_recovery275_archive_valid_v1()
       OR public.lab_arena_sep16_recovery277_archive_valid_v1() IS NOT TRUE$new$;
  v_old_count INTEGER;
  v_new_count INTEGER;
  v_base_function_count INTEGER;
  v_older_function_count INTEGER;
  v_prior_function_count INTEGER;
  v_latest_prior_function_count INTEGER;
  v_new_function_count INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_sep16_rerun_publication_guard_v1()'::REGPROCEDURE
  ) INTO STRICT v_definition;
  v_old_count := (
    pg_catalog.length(v_definition) -
    pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))
  ) / pg_catalog.length(v_old);
  v_new_count := (
    pg_catalog.length(v_definition) -
    pg_catalog.length(pg_catalog.replace(v_definition, v_new, ''))
  ) / pg_catalog.length(v_new);
  v_base_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery_archive_valid_v1()');
  v_older_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery272_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery272_archive_valid_v1()');
  v_prior_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery273_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery273_archive_valid_v1()');
  v_latest_prior_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery275_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery275_archive_valid_v1()');
  v_new_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery277_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery277_archive_valid_v1()');
  IF v_old_count <> 1 OR v_base_function_count <> 1
     OR v_older_function_count <> 1
     OR v_prior_function_count <> 1
     OR v_latest_prior_function_count <> 1 THEN
    RAISE EXCEPTION 'Sep16 publication archive guard shape differs'
      USING ERRCODE = '55000';
  END IF;
  IF v_new_count = 0 AND v_new_function_count = 0 THEN
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  ELSIF NOT (v_new_count = 1 AND v_new_function_count = 1) THEN
    RAISE EXCEPTION 'Sep16 publication recovery277 guard shape differs'
      USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_sep16_rerun_publication_guard_v1()'::REGPROCEDURE
  ) INTO STRICT v_definition;
  v_old_count := (
    pg_catalog.length(v_definition) -
    pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))
  ) / pg_catalog.length(v_old);
  v_new_count := (
    pg_catalog.length(v_definition) -
    pg_catalog.length(pg_catalog.replace(v_definition, v_new, ''))
  ) / pg_catalog.length(v_new);
  v_base_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery_archive_valid_v1()');
  v_older_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery272_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery272_archive_valid_v1()');
  v_prior_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery273_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery273_archive_valid_v1()');
  v_latest_prior_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery275_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery275_archive_valid_v1()');
  v_new_function_count := (
    pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(
      v_definition, 'lab_arena_sep16_recovery277_archive_valid_v1()', ''
    ))
  ) / pg_catalog.length('lab_arena_sep16_recovery277_archive_valid_v1()');
  IF v_old_count <> 1 OR v_new_count <> 1
     OR v_base_function_count <> 1 OR v_older_function_count <> 1
     OR v_prior_function_count <> 1
     OR v_latest_prior_function_count <> 1
     OR v_new_function_count <> 1 THEN
    RAISE EXCEPTION 'Sep16 publication recovery277 guard binding failed'
      USING ERRCODE = '55000';
  END IF;
END;
$bind_sep16_recovery277_archive_guard$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
