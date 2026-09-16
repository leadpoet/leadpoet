-- Exact owner seal for the explicitly authorized Sep16 production rerun.
-- Public champion_model main/lab: b718c5d2f4d02b804a148dae37d511a84ba38ffd.
-- Original twenty ICPs and current production rows were verified read-only.
-- Apply only after stage-source verifies exact object readback on the deployed
-- helper. Local model and runtime checks passed; real positive scoring remains
-- the acceptance condition of the authorized production rerun.
-- This migration grants authority only. It does not prepare, run, score,
-- publish, promote, or change any Git reference.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $owner266_prerequisites$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_sep16_rerun_release_authority') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_sep16_baseline_rerun_audit') IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_prepare_sep16_baseline_rerun_v1(bigint,text,text,text,jsonb)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply inert Sep16 migration265 before owner266';
  END IF;
END;
$owner266_prerequisites$;

DO $owner266_seal$
DECLARE
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_existing public.lab_arena_sep16_rerun_release_authority%ROWTYPE;
  v_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-16/sources/baseline-2026-09-16-native-rerun265.tar.gz';
  v_bank CONSTANT TEXT :=
    '42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390';
  v_old_image CONSTANT TEXT :=
    'sha256:ee84f274ba24b07fa204c03535b21aac3c030c72aff0fcf7918d88d3c2c8ddee';
  v_scoring_tree CONSTANT TEXT :=
    'f8452314cc8b1529fbfa8b7fc9345143c1b7d731';
  v_runtime CONSTANT TEXT :=
    '9d97e2ae295f715209b8fdfef2b6ddebdc46622e';
  v_source_size CONSTANT BIGINT := 523337;
  v_source_sha CONSTANT TEXT := '9170a9c6551b927ac311a415da632babd91e9e6ba049cdcfd72c3f753a496576';
  v_source_commit CONSTANT TEXT := 'b718c5d2f4d02b804a148dae37d511a84ba38ffd';
  v_new_image_ref CONSTANT TEXT := '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385';
  v_new_image_digest CONSTANT TEXT := 'sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385';
  v_old_round_hash CONSTANT TEXT := 'sha256:ddcbc34c51d1e5d4ee673c0c4529170b5e835cf2259508a9e175a03b99a4a7a0';
  v_old_baseline_submission_hash CONSTANT TEXT :=
    'sha256:09055b51ec3a540e6c8da64056c307dda08af67d8f4d21892c60cf3483fb29ff';
  v_old_baseline_runs_hash CONSTANT TEXT := 'sha256:b4a70ca43df2abfe823e0a481915bb38dbf6bcaab9294e4fe166838c2dacca2c';
  v_old_baseline_ledger_hash CONSTANT TEXT := 'sha256:84bf8e11bed7d8bb41e53b1ee8b9bd493f4885172c2c572f5d46c185ec286573';
  v_old_challenger_runs_hash CONSTANT TEXT :=
    'sha256:518533eface43776db8b7a15871a684f976e379c106466cea97c8030561f3768';
  v_old_challenger_submissions_hash CONSTANT TEXT :=
    'sha256:dd73d5d02a6d50b7790172f83149e89cadefb2b527dac2a8f6fbde57f482d89b';
  v_old_challenger_ledger_hash CONSTANT TEXT :=
    'sha256:3c6aeabb33649104ac2b59621f5ec29324e10cbea987d89e972bc91d51444367';
  v_old_challenger_ledger_max CONSTANT BIGINT :=
    434377;
  v_old_baseline_actual CONSTANT BIGINT := 4531542;
  v_schedule CONSTANT JSONB :=
    $sealed_schedule${"benchmark_deadline":"2026-09-16T16:58:00Z","final_scoring_close":"2026-09-17T01:13:00Z","publication_deadline":"2026-09-17T01:43:00Z","stage_1_close":"2026-09-16T21:58:00Z","stage_1_scoring_close":"2026-09-16T23:28:00Z","stage_1_start":"2026-09-16T16:58:01Z","stage_2_close":"2026-09-16T23:43:00Z","stage_2_start":"2026-09-16T23:33:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"}$sealed_schedule$::JSONB;
BEGIN
  IF v_source_size NOT BETWEEN 1 AND 10485760
     OR v_source_sha !~ '^[0-9a-f]{64}$'
     OR v_source_commit !~ '^[0-9a-f]{40}$'
     OR v_new_image_digest !~ '^sha256:[0-9a-f]{64}$'
     OR v_new_image_ref NOT LIKE '%@' || v_new_image_digest
     OR v_old_round_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_old_baseline_submission_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_old_baseline_runs_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_old_baseline_ledger_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_old_challenger_runs_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_old_challenger_submissions_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_old_challenger_ledger_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_old_challenger_ledger_max < 0
     OR v_old_baseline_actual < 0
     OR pg_catalog.jsonb_typeof(v_schedule) IS DISTINCT FROM 'object' THEN
    RAISE EXCEPTION 'owner266 seal is incomplete or invalid' USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_existing
  FROM public.lab_arena_sep16_rerun_release_authority
  WHERE round_id = 'arena-2026-09-16' FOR UPDATE;
  IF FOUND THEN
    IF v_existing.source_ref IS DISTINCT FROM v_source_ref
       OR v_existing.source_size_bytes IS DISTINCT FROM v_source_size
       OR v_existing.source_sha256 IS DISTINCT FROM v_source_sha
       OR v_existing.source_commit IS DISTINCT FROM v_source_commit
       OR v_existing.champion_model_main_commit IS DISTINCT FROM v_source_commit
       OR v_existing.champion_model_lab_commit IS DISTINCT FROM v_source_commit
       OR v_existing.bank_sha256 IS DISTINCT FROM v_bank
       OR v_existing.old_baseline_submission_hash IS DISTINCT FROM
            v_old_baseline_submission_hash
       OR v_existing.old_baseline_runs_hash IS DISTINCT FROM v_old_baseline_runs_hash
       OR v_existing.old_baseline_ledger_hash IS DISTINCT FROM
            v_old_baseline_ledger_hash
       OR v_existing.old_challenger_runs_hash IS DISTINCT FROM
            v_old_challenger_runs_hash
       OR v_existing.old_challenger_submissions_hash IS DISTINCT FROM
            v_old_challenger_submissions_hash
       OR v_existing.old_challenger_ledger_hash IS DISTINCT FROM
            v_old_challenger_ledger_hash
       OR v_existing.old_challenger_ledger_max_entry_id IS DISTINCT FROM
            v_old_challenger_ledger_max
       OR v_existing.old_baseline_actual_microusd IS DISTINCT FROM
            v_old_baseline_actual
       OR v_existing.old_scorer_image_digest IS DISTINCT FROM v_old_image
       OR v_existing.new_scorer_image_reference IS DISTINCT FROM v_new_image_ref
       OR v_existing.new_scorer_image_digest IS DISTINCT FROM v_new_image_digest
       OR v_existing.scoring_tree_hash IS DISTINCT FROM v_scoring_tree
       OR v_existing.native_runtime_commit IS DISTINCT FROM v_runtime
       OR v_existing.verified_parallel_runner_slots IS DISTINCT FROM 10
       OR v_existing.old_round_hash IS DISTINCT FROM v_old_round_hash
       OR v_existing.forward_schedule IS DISTINCT FROM v_schedule THEN
      RAISE EXCEPTION 'existing Sep16 owner seal differs' USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-16' FOR UPDATE;
  SELECT * INTO v_baseline FROM public.lab_arena_submissions
  WHERE submission_id = 'baseline-2026-09-16' FOR UPDATE;
  IF v_round.round_id IS NULL OR v_baseline.submission_id IS NULL
     OR v_round.status IS DISTINCT FROM 'published'
     OR v_round.status_generation IS DISTINCT FROM 12
     OR v_round.stage_generation IS DISTINCT FROM 8
     OR v_round.reward_basis_hash IS DISTINCT FROM
       'sha256:b19a147f1c8cc8365e3cf7dcc96b827ffffdd6c1e8fad65ad9fc25c8895e493f'
     OR v_round.reward_activated_at IS NULL
     OR v_round.king_outcome IS DISTINCT FROM 'no_king'
     OR v_round.configuration_doc ->> 'baseline_source_url' IS DISTINCT FROM
       'https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz'
     OR v_round.configuration_doc ->> 'scorer_image_digest' IS DISTINCT FROM v_old_image
     OR v_baseline.source_ref IS DISTINCT FROM
       'arena/arena-2026-09-16/sources/baseline-2026-09-16.tar.gz' THEN
    RAISE EXCEPTION 'original Sep16 publication differs before owner seal'
      USING ERRCODE = '55000';
  END IF;

  INSERT INTO public.lab_arena_sep16_rerun_release_authority (
    round_id, source_ref, source_size_bytes, source_sha256, source_commit,
    champion_model_main_commit, champion_model_lab_commit, bank_sha256,
    old_round_hash, old_baseline_submission_hash, old_baseline_runs_hash,
    old_baseline_ledger_hash, old_challenger_runs_hash,
    old_challenger_submissions_hash, old_challenger_ledger_hash,
    old_challenger_ledger_max_entry_id, old_baseline_actual_microusd,
    old_scorer_image_digest, new_scorer_image_reference,
    new_scorer_image_digest, scoring_tree_hash, native_runtime_commit,
    verified_parallel_runner_slots, forward_schedule
  ) VALUES (
    v_round.round_id, v_source_ref, v_source_size, v_source_sha, v_source_commit,
    v_source_commit, v_source_commit, v_bank,
    v_old_round_hash, v_old_baseline_submission_hash, v_old_baseline_runs_hash,
    v_old_baseline_ledger_hash, v_old_challenger_runs_hash,
    v_old_challenger_submissions_hash, v_old_challenger_ledger_hash,
    v_old_challenger_ledger_max, v_old_baseline_actual,
    v_old_image, v_new_image_ref, v_new_image_digest, v_scoring_tree, v_runtime,
    10, v_schedule
  );
END;
$owner266_seal$;

COMMIT;
