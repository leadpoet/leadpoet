-- Exact reseal for the latest tested champion_model before the Sep16 rerun.
-- This migration is rendered only after the same commit is published on main
-- and lab and its public lab archive is staged and read back at the new,
-- immutable source key. It refuses any prepared or changed competition state.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $owner268_reseal$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-16';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-16';
  v_old_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-16/sources/baseline-2026-09-16-native-rerun265.tar.gz';
  v_new_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-16/sources/baseline-2026-09-16-native-rerun268.tar.gz';
  v_old_source_size CONSTANT BIGINT := 523337;
  v_old_source_sha CONSTANT TEXT :=
    '9170a9c6551b927ac311a415da632babd91e9e6ba049cdcfd72c3f753a496576';
  v_old_source_commit CONSTANT TEXT :=
    'b718c5d2f4d02b804a148dae37d511a84ba38ffd';
  v_new_source_size_text CONSTANT TEXT := '524266';
  v_new_source_sha CONSTANT TEXT := 'c8d188c4766008ac949c5ff794536fc905589f46657d4b1ed27f3a22385f11b0';
  v_new_source_commit CONSTANT TEXT := '3afe71508636254eb31a172d94f5a63f4f84a5c2';
  v_new_source_size BIGINT;
  v_bank CONSTANT TEXT :=
    '42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390';
  v_old_image CONSTANT TEXT :=
    'sha256:ee84f274ba24b07fa204c03535b21aac3c030c72aff0fcf7918d88d3c2c8ddee';
  v_new_image_ref CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385';
  v_new_image CONSTANT TEXT :=
    'sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385';
  v_scoring_tree CONSTANT TEXT :=
    'f8452314cc8b1529fbfa8b7fc9345143c1b7d731';
  v_runtime CONSTANT TEXT :=
    '9d97e2ae295f715209b8fdfef2b6ddebdc46622e';
  v_old_round_hash CONSTANT TEXT := 'sha256:ddcbc34c51d1e5d4ee673c0c4529170b5e835cf2259508a9e175a03b99a4a7a0';
  v_old_baseline_submission_hash CONSTANT TEXT :=
    'sha256:09055b51ec3a540e6c8da64056c307dda08af67d8f4d21892c60cf3483fb29ff';
  v_old_baseline_runs_hash CONSTANT TEXT := 'sha256:b4a70ca43df2abfe823e0a481915bb38dbf6bcaab9294e4fe166838c2dacca2c';
  v_old_baseline_ledger_hash CONSTANT TEXT := 'sha256:84bf8e11bed7d8bb41e53b1ee8b9bd493f4885172c2c572f5d46c185ec286573';
  v_old_challenger_runs_hash CONSTANT TEXT := 'sha256:518533eface43776db8b7a15871a684f976e379c106466cea97c8030561f3768';
  v_old_challenger_submissions_hash CONSTANT TEXT :=
    'sha256:dd73d5d02a6d50b7790172f83149e89cadefb2b527dac2a8f6fbde57f482d89b';
  v_old_challenger_ledger_hash CONSTANT TEXT :=
    'sha256:3c6aeabb33649104ac2b59621f5ec29324e10cbea987d89e972bc91d51444367';
  v_old_challenger_ledger_max CONSTANT BIGINT :=
    434377;
  v_old_baseline_actual CONSTANT BIGINT :=
    4531542;
  v_rounds_table_hash CONSTANT TEXT := 'sha256:049d4e683a0486ebf4dfe74c726e39a25c0a6b8d23058b7b1a896f3d4389b5d9';
  v_submissions_table_hash CONSTANT TEXT := 'sha256:7893c8aaeda41e64b5fdc41896a589b887903b436846e4cbc79a2218de9e2410';
  v_runs_table_hash CONSTANT TEXT := 'sha256:b0c9c5883f19dd30c7528e812d55c78a3c752b6f73daf6a785b84938bd6ce017';
  v_ledger_table_hash CONSTANT TEXT := 'sha256:1c282c344c1a2d8e8089cde8ebee672e20fcbbfd77b0a37b0ff68e8bb5122c5e';
  v_old_schedule CONSTANT JSONB :=
    $old_schedule${"benchmark_deadline":"2026-09-16T16:58:00Z","final_scoring_close":"2026-09-17T01:13:00Z","publication_deadline":"2026-09-17T01:43:00Z","stage_1_close":"2026-09-16T21:58:00Z","stage_1_scoring_close":"2026-09-16T23:28:00Z","stage_1_start":"2026-09-16T16:58:01Z","stage_2_close":"2026-09-16T23:43:00Z","stage_2_start":"2026-09-16T23:33:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"}$old_schedule$::JSONB;
  v_new_schedule CONSTANT JSONB :=
    $new_schedule${"benchmark_deadline":"2026-09-16T17:45:00Z","final_scoring_close":"2026-09-17T02:15:00Z","publication_deadline":"2026-09-17T02:45:00Z","stage_1_close":"2026-09-16T23:00:00Z","stage_1_scoring_close":"2026-09-17T00:30:00Z","stage_1_start":"2026-09-16T17:45:01Z","stage_2_close":"2026-09-17T00:45:00Z","stage_2_start":"2026-09-17T00:35:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"}$new_schedule$::JSONB;
  v_old_function_hash CONSTANT TEXT :=
    '156a3bf3ef35848825dddcda9aa90c983b74ead218984f0705d6dd7688ef9d12';
  v_new_function_hash CONSTANT TEXT :=
    '0855a9f974ff2a9dfa40540694e59b34336014831d50afef70cafcf733dd6e73';
  v_authority public.lab_arena_sep16_rerun_release_authority%ROWTYPE;
  v_round_hash TEXT;
  v_baseline_submission_hash TEXT;
  v_baseline_runs_hash TEXT;
  v_baseline_ledger_hash TEXT;
  v_challenger_runs_hash TEXT;
  v_challenger_submissions_hash TEXT;
  v_challenger_ledger_hash TEXT;
  v_challenger_ledger_max BIGINT;
  v_baseline_actual BIGINT;
  v_rounds_hash TEXT;
  v_submissions_hash TEXT;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_constraint_definition TEXT;
  v_function_definition TEXT;
  v_function_hash TEXT;
  v_function_owner OID;
  v_function_acl ACLITEM[];
  v_after_owner OID;
  v_after_acl ACLITEM[];
  v_is_old BOOLEAN;
  v_is_new BOOLEAN;
BEGIN
  IF pg_catalog.to_regclass(
       'public.lab_arena_sep16_rerun_release_authority'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_rerun_audit'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_prepare_sep16_baseline_rerun_v1(bigint,text,text,text,jsonb)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply Sep16 migrations265-267 before owner268';
  END IF;
  IF v_new_source_size_text !~ '^[1-9][0-9]{0,7}$'
     OR v_new_source_sha !~ '^[0-9a-f]{64}$'
     OR v_new_source_commit !~ '^[0-9a-f]{40}$'
     OR pg_catalog.jsonb_typeof(v_old_schedule) IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(v_new_schedule) IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(
          v_new_schedule
        )) <> 10
     OR NOT v_new_schedule ?& ARRAY[
       'submission_open', 'submission_cutoff', 'benchmark_deadline',
       'stage_1_start', 'stage_1_close', 'stage_1_scoring_close',
       'stage_2_start', 'stage_2_close', 'final_scoring_close',
       'publication_deadline'
     ]
     OR v_new_schedule ->> 'submission_open' IS DISTINCT FROM
          v_old_schedule ->> 'submission_open'
     OR v_new_schedule ->> 'submission_cutoff' IS DISTINCT FROM
          v_old_schedule ->> 'submission_cutoff' THEN
    RAISE EXCEPTION 'owner268 seal is incomplete or invalid'
      USING ERRCODE = '22023';
  END IF;
  v_new_source_size := v_new_source_size_text::BIGINT;
  IF v_new_source_size NOT BETWEEN 1 AND 10485760
     OR v_new_source_commit = v_old_source_commit
     OR v_new_source_sha = v_old_source_sha THEN
    RAISE EXCEPTION 'owner268 source must be a new bounded release'
      USING ERRCODE = '22023';
  END IF;

  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-16-native-baseline-v1', 0)
  );
  LOCK TABLE public.lab_arena_sep16_rerun_release_authority IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_sep16_baseline_rerun_audit IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;

  IF EXISTS (
    SELECT 1 FROM public.lab_arena_sep16_baseline_rerun_audit
    WHERE round_id = v_round_id
  ) THEN
    RAISE EXCEPTION 'Sep16 rerun is already prepared; reseal refused'
      USING ERRCODE = '55000';
  END IF;
  SELECT * INTO STRICT v_authority
  FROM public.lab_arena_sep16_rerun_release_authority
  WHERE round_id = v_round_id FOR UPDATE;
  IF v_authority.bank_sha256 IS DISTINCT FROM v_bank
     OR v_authority.old_round_hash IS DISTINCT FROM v_old_round_hash
     OR v_authority.old_baseline_submission_hash IS DISTINCT FROM
          v_old_baseline_submission_hash
     OR v_authority.old_baseline_runs_hash IS DISTINCT FROM
          v_old_baseline_runs_hash
     OR v_authority.old_baseline_ledger_hash IS DISTINCT FROM
          v_old_baseline_ledger_hash
     OR v_authority.old_challenger_runs_hash IS DISTINCT FROM
          v_old_challenger_runs_hash
     OR v_authority.old_challenger_submissions_hash IS DISTINCT FROM
          v_old_challenger_submissions_hash
     OR v_authority.old_challenger_ledger_hash IS DISTINCT FROM
          v_old_challenger_ledger_hash
     OR v_authority.old_challenger_ledger_max_entry_id IS DISTINCT FROM
          v_old_challenger_ledger_max
     OR v_authority.old_baseline_actual_microusd IS DISTINCT FROM
          v_old_baseline_actual
     OR v_authority.old_scorer_image_digest IS DISTINCT FROM v_old_image
     OR v_authority.new_scorer_image_reference IS DISTINCT FROM v_new_image_ref
     OR v_authority.new_scorer_image_digest IS DISTINCT FROM v_new_image
     OR v_authority.scoring_tree_hash IS DISTINCT FROM v_scoring_tree
     OR v_authority.native_runtime_commit IS DISTINCT FROM v_runtime
     OR v_authority.verified_parallel_runner_slots IS DISTINCT FROM 10 THEN
    RAISE EXCEPTION 'historical owner266 authority differs before reseal'
      USING ERRCODE = '55000';
  END IF;

  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_round_hash FROM public.lab_arena_rounds AS row_value
  WHERE round_id = v_round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_baseline_submission_hash
  FROM public.lab_arena_submissions AS row_value
  WHERE submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), 'sha256'), 'hex')
  INTO v_baseline_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), 'sha256'), 'hex')
  INTO v_baseline_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY run_id), 'sha256'), 'hex')
  INTO v_challenger_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY submission_id), 'sha256'), 'hex')
  INTO v_challenger_submissions_hash
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_challenger_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT COALESCE(pg_catalog.max(entry_id), 0)
  INTO v_challenger_ledger_max FROM public.lab_arena_ledger
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT COALESCE(pg_catalog.sum(amount_microusd), 0)
  INTO v_baseline_actual FROM public.lab_arena_ledger
  WHERE round_id = v_round_id AND submission_id = v_baseline_id
    AND entry_kind IN ('settlement', 'uncertain');

  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY round_id), ''), 'sha256'), 'hex')
  INTO v_rounds_hash FROM public.lab_arena_rounds AS row_value
  WHERE round_id = v_round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY submission_id), ''), 'sha256'), 'hex')
  INTO v_submissions_hash FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'),
      '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id;
  IF v_round_hash IS DISTINCT FROM v_old_round_hash
     OR v_baseline_submission_hash IS DISTINCT FROM
          v_old_baseline_submission_hash
     OR v_baseline_runs_hash IS DISTINCT FROM v_old_baseline_runs_hash
     OR v_baseline_ledger_hash IS DISTINCT FROM v_old_baseline_ledger_hash
     OR v_challenger_runs_hash IS DISTINCT FROM v_old_challenger_runs_hash
     OR v_challenger_submissions_hash IS DISTINCT FROM
          v_old_challenger_submissions_hash
     OR v_challenger_ledger_hash IS DISTINCT FROM
          v_old_challenger_ledger_hash
     OR v_challenger_ledger_max IS DISTINCT FROM v_old_challenger_ledger_max
     OR v_baseline_actual IS DISTINCT FROM v_old_baseline_actual
     OR v_rounds_hash IS DISTINCT FROM v_rounds_table_hash
     OR v_submissions_hash IS DISTINCT FROM v_submissions_table_hash
     OR v_runs_hash IS DISTINCT FROM v_runs_table_hash
     OR v_ledger_hash IS DISTINCT FROM v_ledger_table_hash THEN
    RAISE EXCEPTION 'Sep16 protected rows differ before reseal'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.pg_get_constraintdef(oid)
  INTO STRICT v_constraint_definition FROM pg_catalog.pg_constraint
  WHERE conrelid =
      'public.lab_arena_sep16_rerun_release_authority'::pg_catalog.regclass
    AND conname =
      'lab_arena_sep16_rerun_release_authority_source_ref_check';
  SELECT pg_catalog.pg_get_functiondef(proc.oid), proc.proowner, proc.proacl
  INTO STRICT v_function_definition, v_function_owner, v_function_acl
  FROM pg_catalog.pg_proc AS proc
  WHERE proc.oid =
    'public.lab_arena_prepare_sep16_baseline_rerun_v1(bigint,text,text,text,jsonb)'
      ::pg_catalog.regprocedure;
  v_function_hash := pg_catalog.encode(extensions.digest(
    v_function_definition, 'sha256'), 'hex');
  v_is_old :=
    v_authority.source_ref = v_old_source_ref
    AND v_authority.source_size_bytes = v_old_source_size
    AND v_authority.source_sha256 = v_old_source_sha
    AND v_authority.source_commit = v_old_source_commit
    AND v_authority.champion_model_main_commit = v_old_source_commit
    AND v_authority.champion_model_lab_commit = v_old_source_commit
    AND v_authority.forward_schedule = v_old_schedule
    AND v_constraint_definition = pg_catalog.format(
      'CHECK ((source_ref = %L::text))', v_old_source_ref
    )
    AND v_function_hash = v_old_function_hash;
  v_is_new :=
    v_authority.source_ref = v_new_source_ref
    AND v_authority.source_size_bytes = v_new_source_size
    AND v_authority.source_sha256 = v_new_source_sha
    AND v_authority.source_commit = v_new_source_commit
    AND v_authority.champion_model_main_commit = v_new_source_commit
    AND v_authority.champion_model_lab_commit = v_new_source_commit
    AND v_authority.forward_schedule = v_new_schedule
    AND v_constraint_definition = pg_catalog.format(
      'CHECK ((source_ref = %L::text))', v_new_source_ref
    )
    AND v_function_hash = v_new_function_hash;
  IF v_is_new THEN
    RETURN;
  END IF;
  IF NOT v_is_old THEN
    RAISE EXCEPTION 'Sep16 reseal source/function/constraint state is mixed'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.length(v_function_definition) - pg_catalog.length(
       pg_catalog.replace(v_function_definition, v_old_source_ref, '')
     ) <> pg_catalog.length(v_old_source_ref)
     OR pg_catalog.strpos(v_function_definition, v_new_source_ref) <> 0 THEN
    RAISE EXCEPTION 'Sep16 prepare source literal differs'
      USING ERRCODE = '55000';
  END IF;

  ALTER TABLE public.lab_arena_sep16_rerun_release_authority
    DROP CONSTRAINT lab_arena_sep16_rerun_release_authority_source_ref_check;
  UPDATE public.lab_arena_sep16_rerun_release_authority SET
    source_ref = v_new_source_ref,
    source_size_bytes = v_new_source_size,
    source_sha256 = v_new_source_sha,
    source_commit = v_new_source_commit,
    champion_model_main_commit = v_new_source_commit,
    champion_model_lab_commit = v_new_source_commit,
    forward_schedule = v_new_schedule,
    authorized_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round_id;
  ALTER TABLE public.lab_arena_sep16_rerun_release_authority
    ADD CONSTRAINT lab_arena_sep16_rerun_release_authority_source_ref_check
    CHECK (source_ref =
      'arena/arena-2026-09-16/sources/baseline-2026-09-16-native-rerun268.tar.gz');

  v_function_definition := pg_catalog.replace(
    v_function_definition, v_old_source_ref, v_new_source_ref
  );
  EXECUTE v_function_definition;
  SELECT proc.proowner, proc.proacl,
         pg_catalog.encode(extensions.digest(
           pg_catalog.pg_get_functiondef(proc.oid), 'sha256'), 'hex')
  INTO STRICT v_after_owner, v_after_acl, v_function_hash
  FROM pg_catalog.pg_proc AS proc
  WHERE proc.oid =
    'public.lab_arena_prepare_sep16_baseline_rerun_v1(bigint,text,text,text,jsonb)'
      ::pg_catalog.regprocedure;
  IF v_after_owner IS DISTINCT FROM v_function_owner
     OR v_after_acl IS DISTINCT FROM v_function_acl
     OR v_function_hash IS DISTINCT FROM v_new_function_hash THEN
    RAISE EXCEPTION 'Sep16 prepare reseal identity differs'
      USING ERRCODE = '55000';
  END IF;
END;
$owner268_reseal$;

COMMIT;
