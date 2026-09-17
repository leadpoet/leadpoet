-- Reseal only the unused recovery275 source for arena-2026-09-16.
-- Migration 275 remains immutable. This migration is valid only before its
-- recovery RPC has created an audit row or any rerun275 work.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $reseal_unused_recovery275_source$
DECLARE
  v_authority public.lab_arena_sep16_baseline_recovery275_authority%ROWTYPE;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_expected_non_source JSONB := $expected${
    "archive_round_id": "arena-2026-09-16-rerun273archive",
    "archive_submission_id": "baseline-2026-09-16-native-rerun273-archive",
    "bank_sha256": "42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390",
    "challenger_ledger_hash": "sha256:3c6aeabb33649104ac2b59621f5ec29324e10cbea987d89e972bc91d51444367",
    "challenger_ledger_max_entry_id": 434377,
    "challenger_runs_hash": "sha256:518533eface43776db8b7a15871a684f976e379c106466cea97c8030561f3768",
    "challenger_submissions_hash": "sha256:dd73d5d02a6d50b7790172f83149e89cadefb2b527dac2a8f6fbde57f482d89b",
    "execute_namespace": "rerun275",
    "forward_schedule": {"benchmark_deadline":"2026-09-16T23:15:00Z","final_scoring_close":"2026-09-17T08:15:00Z","publication_deadline":"2026-09-17T08:45:00Z","stage_1_close":"2026-09-17T05:00:00Z","stage_1_scoring_close":"2026-09-17T06:30:00Z","stage_1_start":"2026-09-16T23:15:01Z","stage_2_close":"2026-09-17T06:45:00Z","stage_2_start":"2026-09-17T06:35:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"},
    "old_schedule": {"benchmark_deadline":"2026-09-16T23:15:00Z","final_scoring_close":"2026-09-17T08:00:00Z","publication_deadline":"2026-09-17T08:30:00Z","stage_1_close":"2026-09-17T04:45:00Z","stage_1_scoring_close":"2026-09-17T06:15:00Z","stage_1_start":"2026-09-16T23:15:01Z","stage_2_close":"2026-09-17T06:30:00Z","stage_2_start":"2026-09-17T06:20:00Z","submission_cutoff":"2026-09-16T00:00:00Z","submission_open":"2026-09-15T00:00:00Z"},
    "prior_recovery_audit_hash": "sha256:1ad718f9bdd3681366ddddc23893507cbaabc4c8fab5abcc9a1c6416a3012d17",
    "prior_recovery_authority_hash": "sha256:88889eabfb751c8c1383c4f08c891b1e6884d5e80d7be49c22b3eb8bfc948664",
    "prior_rerun_audit_hash": "sha256:ed1412ab5cb0be2aa60a9903952fb67173f6038528f2384cec43b48ffa562c33",
    "recovery_source_ref": "arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery275.tar.gz",
    "release_authority_hash": "sha256:4edca10bfc6ad93d120688efdfa15f50e30bc02e51c416e78d5505e9a0766a7f",
    "round_id": "arena-2026-09-16",
    "scorer_image_digest": "sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385",
    "terminal_baseline_ledger_count": 4428,
    "terminal_baseline_ledger_hash": "sha256:d35756d2e525da021fc9bc1729836456d4756b3880646e7245d009e8b4b01eba",
    "terminal_baseline_ledger_max_entry_id": 457405,
    "terminal_baseline_run_count": 27,
    "terminal_baseline_runs_hash": "sha256:22aaa564cc00297f03576dadcd47f69582a77fc443800d78739cf770b9a82b05",
    "terminal_baseline_settled_microusd": 11431169,
    "terminal_baseline_submission_hash": "sha256:25cd58d4e9111d97fe9f8bfa3ec8a19544d7d57fcf907436de1bf0ef86a72889",
    "terminal_baseline_uncertain_microusd": 12127632,
    "terminal_cancel_reason": "operator",
    "terminal_round_hash": "sha256:96f220bdd5c3e7f9c4b4fa1feb427a0a1b9eac147fe45dd0a3f0d955aa8db285",
    "terminal_source_commit": "ec4887f728b77002de570c4023415f8963995b79",
    "terminal_source_ref": "arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery273.tar.gz",
    "terminal_source_sha256": "858ad5b2e68e3c6c354c0e4e358264186cec206b83220a68719a6e217273d057",
    "terminal_source_size_bytes": 547308,
    "terminal_stage_generation": 16,
    "terminal_status_generation": 20
  }$expected$::JSONB;
  v_before_non_source JSONB;
  v_after_non_source JSONB;
  v_round_hash TEXT;
  v_submission_hash TEXT;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_challenger_runs_hash TEXT;
  v_challenger_submissions_hash TEXT;
  v_challenger_ledger_hash TEXT;
  v_prior_rerun_audit_hash TEXT;
  v_release_authority_hash TEXT;
  v_prior_recovery_authority_hash TEXT;
  v_prior_recovery_audit_hash TEXT;
  v_run_count BIGINT;
  v_ledger_count BIGINT;
  v_ledger_max BIGINT;
  v_settled BIGINT;
  v_uncertain BIGINT;
  v_challenger_ledger_max BIGINT;
  v_execute_cost JSONB;
  v_score_cost JSONB;
  v_size_constraint TEXT;
  v_sha_constraint TEXT;
  v_commit_constraint TEXT;
  v_size_definition TEXT;
  v_sha_definition TEXT;
  v_commit_definition TEXT;
  v_relowner OID;
  v_relrowsecurity BOOLEAN;
  v_relacl ACLITEM[];
  v_old BOOLEAN;
  v_new BOOLEAN;
BEGIN
  IF pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_recovery275_authority'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_baseline_recovery275_audit'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_prepare_sep16_baseline_recovery275_v1(bigint,text,text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply exact recovery275 before source reseal';
  END IF;

  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-16-native-baseline-v1', 0)
  );

  SELECT c.relowner, c.relrowsecurity, c.relacl
  INTO STRICT v_relowner, v_relrowsecurity, v_relacl
  FROM pg_catalog.pg_class AS c
  WHERE c.oid =
    'public.lab_arena_sep16_baseline_recovery275_authority'::REGCLASS;

  SELECT * INTO STRICT v_authority
  FROM public.lab_arena_sep16_baseline_recovery275_authority
  WHERE round_id = 'arena-2026-09-16'
  FOR UPDATE;

  IF (SELECT pg_catalog.count(*)
      FROM public.lab_arena_sep16_baseline_recovery275_authority) <> 1 THEN
    RAISE EXCEPTION 'recovery275 authority cardinality differs'
      USING ERRCODE = '55000';
  END IF;

  v_before_non_source := pg_catalog.to_jsonb(v_authority)
    - 'authorized_at'
    - 'recovery_source_size_bytes'
    - 'recovery_source_sha256'
    - 'recovery_source_commit';
  IF v_before_non_source IS DISTINCT FROM v_expected_non_source THEN
    RAISE EXCEPTION 'recovery275 non-source authority differs'
      USING ERRCODE = '55000';
  END IF;

  IF EXISTS (
       SELECT 1 FROM public.lab_arena_sep16_baseline_recovery275_audit
       WHERE round_id = v_authority.round_id
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_authority.round_id
         AND assignment_id LIKE '%:rerun275%'
     ) THEN
    RAISE EXCEPTION 'recovery275 already started; source reseal refused'
      USING ERRCODE = '55000';
  END IF;

  SELECT * INTO STRICT v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_authority.round_id
  FOR SHARE;
  SELECT * INTO STRICT v_baseline
  FROM public.lab_arena_submissions
  WHERE submission_id = 'baseline-2026-09-16'
  FOR SHARE;

  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_round)::TEXT, 'sha256'), 'hex')
  INTO v_round_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_baseline)::TEXT, 'sha256'), 'hex')
  INTO v_submission_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*)
  INTO v_runs_hash, v_run_count
  FROM public.lab_arena_runs AS row_value
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
  INTO v_challenger_runs_hash
  FROM public.lab_arena_runs AS row_value
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
  INTO v_prior_rerun_audit_hash
  FROM public.lab_arena_sep16_baseline_rerun_audit AS row_value
  WHERE round_id = v_authority.round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_release_authority_hash
  FROM public.lab_arena_sep16_rerun_release_authority AS row_value
  WHERE round_id = v_authority.round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_recovery_authority_hash
  FROM public.lab_arena_sep16_baseline_recovery273_authority AS row_value
  WHERE round_id = v_authority.round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex')
  INTO v_prior_recovery_audit_hash
  FROM public.lab_arena_sep16_baseline_recovery273_audit AS row_value
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
     OR v_run_count IS DISTINCT FROM v_authority.terminal_baseline_run_count
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
     OR v_prior_rerun_audit_hash IS DISTINCT FROM
          v_authority.prior_rerun_audit_hash
     OR v_release_authority_hash IS DISTINCT FROM
          v_authority.release_authority_hash
     OR v_prior_recovery_authority_hash IS DISTINCT FROM
          v_authority.prior_recovery_authority_hash
     OR v_prior_recovery_audit_hash IS DISTINCT FROM
          v_authority.prior_recovery_audit_hash
     OR v_round.status <> 'cancelled'
     OR v_round.status_generation <> 20
     OR v_round.stage_generation <> 16
     OR v_round.cancel_reason <> 'operator'
     OR COALESCE((v_execute_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE(
          (v_execute_cost ->> 'success_unresolved_calls')::BIGINT, -1
        ) <> 0
     OR COALESCE((v_score_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE(
          (v_score_cost ->> 'success_unresolved_calls')::BIGINT, -1
        ) <> 0
     OR public.lab_arena_sep16_challenger_seals_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery272_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep16_recovery273_archive_valid_v1() IS NOT TRUE
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_authority.round_id
         AND status IN ('pending', 'leased', 'submitted')
     ) THEN
    RAISE EXCEPTION 'recovery275 sealed terminal state differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT c.conname, pg_catalog.pg_get_constraintdef(c.oid, FALSE)
  INTO STRICT v_size_constraint, v_size_definition
  FROM pg_catalog.pg_constraint AS c
  WHERE c.conrelid =
          'public.lab_arena_sep16_baseline_recovery275_authority'::REGCLASS
    AND c.contype = 'c'
    AND (SELECT a.attnum FROM pg_catalog.pg_attribute AS a
         WHERE a.attrelid = c.conrelid
           AND a.attname = 'recovery_source_size_bytes') = ANY(c.conkey);
  SELECT c.conname, pg_catalog.pg_get_constraintdef(c.oid, FALSE)
  INTO STRICT v_sha_constraint, v_sha_definition
  FROM pg_catalog.pg_constraint AS c
  WHERE c.conrelid =
          'public.lab_arena_sep16_baseline_recovery275_authority'::REGCLASS
    AND c.contype = 'c'
    AND (SELECT a.attnum FROM pg_catalog.pg_attribute AS a
         WHERE a.attrelid = c.conrelid
           AND a.attname = 'recovery_source_sha256') = ANY(c.conkey);
  SELECT c.conname, pg_catalog.pg_get_constraintdef(c.oid, FALSE)
  INTO STRICT v_commit_constraint, v_commit_definition
  FROM pg_catalog.pg_constraint AS c
  WHERE c.conrelid =
          'public.lab_arena_sep16_baseline_recovery275_authority'::REGCLASS
    AND c.contype = 'c'
    AND (SELECT a.attnum FROM pg_catalog.pg_attribute AS a
         WHERE a.attrelid = c.conrelid
           AND a.attname = 'recovery_source_commit') = ANY(c.conkey);

  v_old := v_authority.recovery_source_size_bytes = 556089
    AND v_authority.recovery_source_sha256 =
      'dd877baf3f1210480b8bb1a0fdd62c9b0d75a1478da705a0db043d98978ffe67'
    AND v_authority.recovery_source_commit =
      '396bcb277ce831fd92e31a80f556d2996812bbf4'
    AND v_size_definition =
      'CHECK ((recovery_source_size_bytes = 556089))'
    AND v_sha_definition =
      'CHECK ((recovery_source_sha256 = ''dd877baf3f1210480b8bb1a0fdd62c9b0d75a1478da705a0db043d98978ffe67''::text))'
    AND v_commit_definition =
      'CHECK ((recovery_source_commit = ''396bcb277ce831fd92e31a80f556d2996812bbf4''::text))';
  v_new := v_authority.recovery_source_size_bytes =
      556500
    AND v_authority.recovery_source_sha256 =
      'c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104'
    AND v_authority.recovery_source_commit =
      '2b8386b835de02739ddef6c182bdeba43c6ad467'
    AND v_size_constraint = 'lab_arena_recovery275_source_size_ck'
    AND v_sha_constraint = 'lab_arena_recovery275_source_sha256_ck'
    AND v_commit_constraint = 'lab_arena_recovery275_source_commit_ck'
    AND v_size_definition =
      'CHECK ((recovery_source_size_bytes = 556500))'
    AND v_sha_definition =
      'CHECK ((recovery_source_sha256 = ''c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104''::text))'
    AND v_commit_definition =
      'CHECK ((recovery_source_commit = ''2b8386b835de02739ddef6c182bdeba43c6ad467''::text))';

  IF NOT v_old AND NOT v_new THEN
    RAISE EXCEPTION 'recovery275 source authority or constraints differ'
      USING ERRCODE = '55000';
  END IF;

  IF v_old THEN
    EXECUTE pg_catalog.format(
      'ALTER TABLE public.lab_arena_sep16_baseline_recovery275_authority DROP CONSTRAINT %I',
      v_size_constraint
    );
    EXECUTE pg_catalog.format(
      'ALTER TABLE public.lab_arena_sep16_baseline_recovery275_authority DROP CONSTRAINT %I',
      v_sha_constraint
    );
    EXECUTE pg_catalog.format(
      'ALTER TABLE public.lab_arena_sep16_baseline_recovery275_authority DROP CONSTRAINT %I',
      v_commit_constraint
    );
    UPDATE public.lab_arena_sep16_baseline_recovery275_authority
    SET recovery_source_size_bytes = 556500,
        recovery_source_sha256 = 'c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104',
        recovery_source_commit = '2b8386b835de02739ddef6c182bdeba43c6ad467'
    WHERE round_id = 'arena-2026-09-16';
    ALTER TABLE public.lab_arena_sep16_baseline_recovery275_authority
      ADD CONSTRAINT lab_arena_recovery275_source_size_ck CHECK (
        recovery_source_size_bytes = 556500
      ),
      ADD CONSTRAINT lab_arena_recovery275_source_sha256_ck CHECK (
        recovery_source_sha256 = 'c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104'
      ),
      ADD CONSTRAINT lab_arena_recovery275_source_commit_ck CHECK (
        recovery_source_commit = '2b8386b835de02739ddef6c182bdeba43c6ad467'
      );
  END IF;

  SELECT * INTO STRICT v_authority
  FROM public.lab_arena_sep16_baseline_recovery275_authority
  WHERE round_id = 'arena-2026-09-16';
  v_after_non_source := pg_catalog.to_jsonb(v_authority)
    - 'authorized_at'
    - 'recovery_source_size_bytes'
    - 'recovery_source_sha256'
    - 'recovery_source_commit';
  IF v_after_non_source IS DISTINCT FROM v_before_non_source
     OR v_authority.recovery_source_size_bytes <>
          556500
     OR v_authority.recovery_source_sha256 <>
          'c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104'
     OR v_authority.recovery_source_commit <>
          '2b8386b835de02739ddef6c182bdeba43c6ad467'
     OR (SELECT c.relowner FROM pg_catalog.pg_class AS c
         WHERE c.oid =
           'public.lab_arena_sep16_baseline_recovery275_authority'::REGCLASS)
          IS DISTINCT FROM v_relowner
     OR (SELECT c.relrowsecurity FROM pg_catalog.pg_class AS c
         WHERE c.oid =
           'public.lab_arena_sep16_baseline_recovery275_authority'::REGCLASS)
          IS DISTINCT FROM v_relrowsecurity
     OR (SELECT c.relacl FROM pg_catalog.pg_class AS c
         WHERE c.oid =
           'public.lab_arena_sep16_baseline_recovery275_authority'::REGCLASS)
          IS DISTINCT FROM v_relacl THEN
    RAISE EXCEPTION 'recovery275 source reseal verification differs'
      USING ERRCODE = '55000';
  END IF;
END;
$reseal_unused_recovery275_source$;

COMMIT;
