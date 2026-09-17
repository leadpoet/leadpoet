-- Update only the archive identity for the still-unused September 17 recovery.
-- The public repository was renamed without changing its Git history or lab
-- commit, so the tar root changed the archive hash and size only.
-- No round, run, submission, ledger, schedule, quota, or scoring row changes.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';
DO $sep17_renamed_source_archive280$
DECLARE
  v_before JSONB;
  v_after JSONB;
  v_source_constraint_count INTEGER;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-17-baseline-recovery278', 0)
  );
  LOCK TABLE public.lab_arena_sep17_baseline_recovery278_authority
    IN ACCESS EXCLUSIVE MODE;
  SELECT pg_catalog.to_jsonb(a) INTO STRICT v_before
  FROM public.lab_arena_sep17_baseline_recovery278_authority a
  WHERE round_id = 'arena-2026-09-17' FOR UPDATE;

  IF v_before ->> 'recovery_source_commit' =
       '4ceae936b902433432a195f77af3f94559d80378'
     AND v_before ->> 'recovery_source_sha256' =
       'f6d8ca05a33907489138b37b2089151d78a84a05e5c73020147a282a29383361'
     AND (v_before ->> 'recovery_source_size_bytes')::BIGINT = 563105 THEN
    SELECT pg_catalog.count(*) INTO v_source_constraint_count
    FROM pg_catalog.pg_constraint c
    WHERE c.conrelid =
        'public.lab_arena_sep17_baseline_recovery278_authority'::REGCLASS
      AND c.contype = 'c'
      AND c.conkey && ARRAY[
        (SELECT a.attnum FROM pg_catalog.pg_attribute a
         WHERE a.attrelid = c.conrelid AND a.attname = 'recovery_source_commit'),
        (SELECT a.attnum FROM pg_catalog.pg_attribute a
         WHERE a.attrelid = c.conrelid AND a.attname = 'recovery_source_sha256'),
        (SELECT a.attnum FROM pg_catalog.pg_attribute a
         WHERE a.attrelid = c.conrelid AND a.attname = 'recovery_source_size_bytes')
      ]::SMALLINT[];
    IF v_source_constraint_count <> 1 OR NOT EXISTS (
      SELECT 1 FROM pg_catalog.pg_constraint c
      WHERE c.conrelid =
          'public.lab_arena_sep17_baseline_recovery278_authority'::REGCLASS
        AND c.contype = 'c'
        AND c.conname = 'recovery280_exact_renamed_archive'
        AND c.convalidated
        AND pg_catalog.pg_get_expr(c.conbin, c.conrelid, TRUE) =
          'recovery_source_commit = ''4ceae936b902433432a195f77af3f94559d80378''::text AND recovery_source_sha256 = ''f6d8ca05a33907489138b37b2089151d78a84a05e5c73020147a282a29383361''::text AND recovery_source_size_bytes = 563105'
    ) THEN
      RAISE EXCEPTION 'Sep17 recovery280 source constraint differs'
        USING ERRCODE = '55000';
    END IF;
    RETURN; -- Exact replay is read-only even after recovery activation.
  END IF;

  -- Follow recovery278's lock order before validating the unused terminal seal.
  LOCK TABLE public.lab_arena_sep17_baseline_recovery278_audit
    IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
  IF v_before ->> 'recovery_source_commit' IS DISTINCT FROM
       '4ceae936b902433432a195f77af3f94559d80378'
     OR v_before ->> 'recovery_source_sha256' IS DISTINCT FROM
       '6c9eeaac41386204a7817b00038e5a3f3545e1c205a2dc32de6889927ffe7245'
     OR (v_before ->> 'recovery_source_size_bytes')::BIGINT IS DISTINCT FROM 563030
     OR EXISTS (SELECT 1 FROM public.lab_arena_sep17_baseline_recovery278_audit)
     OR public.lab_arena_sep17_recovery278_terminal_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep17 archive update requires the exact unused recovery278'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.count(*) INTO v_source_constraint_count
  FROM pg_catalog.pg_constraint c
  WHERE c.conrelid =
      'public.lab_arena_sep17_baseline_recovery278_authority'::REGCLASS
    AND c.contype = 'c'
    AND c.conkey && ARRAY[
      (SELECT a.attnum FROM pg_catalog.pg_attribute a
       WHERE a.attrelid = c.conrelid AND a.attname = 'recovery_source_commit'),
      (SELECT a.attnum FROM pg_catalog.pg_attribute a
       WHERE a.attrelid = c.conrelid AND a.attname = 'recovery_source_sha256'),
      (SELECT a.attnum FROM pg_catalog.pg_attribute a
       WHERE a.attrelid = c.conrelid AND a.attname = 'recovery_source_size_bytes')
    ]::SMALLINT[];
  IF v_source_constraint_count <> 1 OR NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_constraint c
    WHERE c.conrelid =
        'public.lab_arena_sep17_baseline_recovery278_authority'::REGCLASS
      AND c.contype = 'c'
      AND c.conname = 'recovery279_exact_reviewed_source'
      AND c.convalidated
      AND pg_catalog.pg_get_expr(c.conbin, c.conrelid, TRUE) =
        'recovery_source_commit = ''4ceae936b902433432a195f77af3f94559d80378''::text AND recovery_source_sha256 = ''6c9eeaac41386204a7817b00038e5a3f3545e1c205a2dc32de6889927ffe7245''::text AND recovery_source_size_bytes = 563030'
  ) THEN
    RAISE EXCEPTION 'Sep17 recovery279 source constraint differs'
      USING ERRCODE = '55000';
  END IF;

  ALTER TABLE public.lab_arena_sep17_baseline_recovery278_authority
    DROP CONSTRAINT recovery279_exact_reviewed_source;
  UPDATE public.lab_arena_sep17_baseline_recovery278_authority
  SET recovery_source_sha256 =
        'f6d8ca05a33907489138b37b2089151d78a84a05e5c73020147a282a29383361',
      recovery_source_size_bytes = 563105
  WHERE round_id = 'arena-2026-09-17';
  ALTER TABLE public.lab_arena_sep17_baseline_recovery278_authority
    ADD CONSTRAINT recovery280_exact_renamed_archive CHECK (
      recovery_source_commit = '4ceae936b902433432a195f77af3f94559d80378'
      AND recovery_source_sha256 =
        'f6d8ca05a33907489138b37b2089151d78a84a05e5c73020147a282a29383361'
      AND recovery_source_size_bytes = 563105
    );

  SELECT pg_catalog.to_jsonb(a) INTO STRICT v_after
  FROM public.lab_arena_sep17_baseline_recovery278_authority a
  WHERE round_id = 'arena-2026-09-17';
  IF (v_before - 'recovery_source_sha256' - 'recovery_source_size_bytes')
       IS DISTINCT FROM
     (v_after - 'recovery_source_sha256' - 'recovery_source_size_bytes')
     OR public.lab_arena_sep17_recovery278_terminal_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep17 archive update changed another recovery field';
  END IF;
END;
$sep17_renamed_source_archive280$;
COMMIT;
