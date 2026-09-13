-- Bind the corrected scorer image to work created after September 13 stage 1.
-- Stage-1 judgments remain bound to their original image and cache scopes.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_238_bind_future_stage_scorer$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-13';
  v_old_digest CONSTANT TEXT :=
    'sha256:67b3ed3c8691a321007da8cc8353a4bb89b517813c7e017fcb93f55e468dd536';
  v_old_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:67b3ed3c8691a321007da8cc8353a4bb89b517813c7e017fcb93f55e468dd536';
  v_initial_digest CONSTANT TEXT :=
    'sha256:b6e58855c4962faa345bd785e0d9aff3210457f5cf2d1dd3b4ee8e29f0b69757';
  v_initial_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:b6e58855c4962faa345bd785e0d9aff3210457f5cf2d1dd3b4ee8e29f0b69757';
  v_first_recovery_digest CONSTANT TEXT :=
    'sha256:f1812fbedf8700fd491fea94c576917018cc6fc1970a124784e22359eabaee4d';
  v_first_recovery_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:f1812fbedf8700fd491fea94c576917018cc6fc1970a124784e22359eabaee4d';
  v_new_digest CONSTANT TEXT :=
    'sha256:188fe7f79e0233c4213bd6262f0d47e07f83d14b3374ee502f7268433111ba4e';
  v_new_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:188fe7f79e0233c4213bd6262f0d47e07f83d14b3374ee502f7268433111ba4e';
  v_round public.lab_arena_rounds%ROWTYPE;
  v_configuration_stable JSONB;
  v_round_stable JSONB;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_cache_hash TEXT;
  v_submissions_hash TEXT;
  v_credentials_hash TEXT;
  v_unrelated_hash TEXT;
  v_count INTEGER;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN;
  END IF;

  IF v_new_digest !~ '^sha256:[0-9a-f]{64}$'
     OR v_new_reference NOT LIKE '%@' || v_new_digest THEN
    RAISE EXCEPTION 'set the exact scorer image before migration 238';
  END IF;

  IF v_round.configuration_doc ->> 'scorer_image_digest' = v_new_digest THEN
    IF v_round.configuration_doc ->> 'scorer_image_reference'
         IS DISTINCT FROM v_new_reference THEN
      RAISE EXCEPTION 'arena 2026-09-13 scorer image replay differs';
    END IF;
    RETURN;
  END IF;

  IF NOT (
       (v_round.status = 'stage1_judged'
        AND v_round.status_generation = 9)
       OR
       (v_round.status = 'stage1_scored'
        AND v_round.status_generation = 10)
     )
     OR v_round.stage_generation <> 8
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.evaluation_date <> '2026-09-13'
     OR v_round.icp_set_date <> DATE '2026-09-12'
     OR v_round.configuration_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_old_digest
     OR v_round.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_old_reference
     OR pg_catalog.jsonb_typeof(
          v_round.stage1_scoring_plan_doc -> 'work_items'
        ) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'work_items'
        ) <> 110
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND stage > 1
     )
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 110
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted') <> 110
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_runs AS score
       LEFT JOIN public.lab_arena_judgment_cache AS cache
         ON cache.cache_key = score.judgment_cache_key
       LEFT JOIN public.lab_arena_runs AS source
         ON source.run_id = score.judgment_cache_source_run_id
       WHERE score.round_id = v_round_id AND score.stage = 1
         AND score.kind = 'score' AND score.status = 'accepted'
         AND (
           score.judgment_cache_key IS NULL
           OR score.judgment_cache_source_run_id IS NULL
           OR (score.judgment_scope_doc ->> 'scorer_image_digest',
               score.judgment_scope_doc ->> 'scorer_image_reference')
              NOT IN (
                (v_initial_digest, v_initial_reference),
                (v_first_recovery_digest, v_first_recovery_reference),
                (v_old_digest, v_old_reference)
              )
           OR cache.cache_key IS NULL
           OR cache.scope_doc IS DISTINCT FROM score.judgment_scope_doc
           OR cache.scoring_input_hash IS DISTINCT FROM
              score.judgment_input_hash
           OR cache.source_score_run_id IS DISTINCT FROM
              score.judgment_cache_source_run_id
           OR cache.evidence_doc ->> 'source_score_run_id'
              IS DISTINCT FROM score.judgment_cache_source_run_id
           OR source.status IS DISTINCT FROM 'accepted'
           OR source.kind IS DISTINCT FROM 'score'
           OR source.scored_run_id IS DISTINCT FROM
              cache.evidence_doc ->> 'source_scored_run_id'
           OR source.runner_hotkey IS DISTINCT FROM
              cache.source_runner_hotkey
           OR source.runner_hotkey IS DISTINCT FROM
              cache.evidence_doc ->> 'source_runner_hotkey'
         )
     )
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted'
           AND judgment_scope_doc ->> 'scorer_image_digest'
               = v_initial_digest) <> 22
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted'
           AND judgment_scope_doc ->> 'scorer_image_digest'
               = v_first_recovery_digest) <> 4
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted'
           AND judgment_scope_doc ->> 'scorer_image_digest'
               = v_old_digest) <> 84
     OR (
       v_round.status = 'stage1_scored'
       AND (
         pg_catalog.jsonb_typeof(v_round.finalists) IS DISTINCT FROM 'array'
         OR pg_catalog.jsonb_array_length(v_round.finalists) <> 10
         OR (SELECT pg_catalog.count(*)
             FROM public.lab_arena_runs
             WHERE round_id = v_round_id AND stage = 1
               AND kind = 'execute' AND per_icp_score IS NOT NULL) <> 110
       )
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 future scorer state differs';
  END IF;

  v_configuration_stable := v_round.configuration_doc
    - 'scorer_image_digest' - 'scorer_image_reference';
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'configuration_doc' - 'updated_at';
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         ), '')) INTO v_runs_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(ledger)::TEXT, '|' ORDER BY entry_id
         ), '')) INTO v_ledger_hash
  FROM public.lab_arena_ledger AS ledger
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(cache_row)::TEXT, '|' ORDER BY cache_key
         ), '')) INTO v_cache_hash
  FROM public.lab_arena_judgment_cache AS cache_row
  WHERE scope_doc ->> 'round_id' = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(submission)::TEXT, '|'
           ORDER BY submission_id
         ), '')) INTO v_submissions_hash
  FROM public.lab_arena_submissions AS submission
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(credential)::TEXT, '|'
           ORDER BY submission_id, provider
         ), '')) INTO v_credentials_hash
  FROM public.lab_arena_submission_credentials AS credential
  WHERE submission_id IN (
    SELECT participant ->> 'submission_id'
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
  );
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(other_round)::TEXT, '|'
           ORDER BY round_id
         ), '')) INTO v_unrelated_hash
  FROM public.lab_arena_rounds AS other_round
  WHERE round_id <> v_round_id;

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET configuration_doc = pg_catalog.jsonb_set(
    pg_catalog.jsonb_set(
      configuration_doc,
      '{scorer_image_digest}', pg_catalog.to_jsonb(v_new_digest), FALSE
    ),
    '{scorer_image_reference}', pg_catalog.to_jsonb(v_new_reference), FALSE
  )
  WHERE round_id = v_round_id
    AND configuration_doc ->> 'scorer_image_digest' = v_old_digest
    AND configuration_doc ->> 'scorer_image_reference' = v_old_reference;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;

  IF v_count <> 1
     OR (SELECT configuration_doc
         FROM public.lab_arena_rounds WHERE round_id = v_round_id)
          - 'scorer_image_digest' - 'scorer_image_reference'
        IS DISTINCT FROM v_configuration_stable
     OR (SELECT pg_catalog.to_jsonb(round_row)
           - 'configuration_doc' - 'updated_at'
         FROM public.lab_arena_rounds AS round_row
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_round_id
         AND configuration_doc ->> 'scorer_image_digest' = v_new_digest
         AND configuration_doc ->> 'scorer_image_reference' = v_new_reference
     )
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         ), '')) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id) IS DISTINCT FROM v_runs_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(ledger)::TEXT, '|' ORDER BY entry_id
         ), '')) FROM public.lab_arena_ledger AS ledger
         WHERE round_id = v_round_id) IS DISTINCT FROM v_ledger_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(cache_row)::TEXT, '|' ORDER BY cache_key
         ), '')) FROM public.lab_arena_judgment_cache AS cache_row
         WHERE scope_doc ->> 'round_id' = v_round_id)
        IS DISTINCT FROM v_cache_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(submission)::TEXT, '|'
           ORDER BY submission_id
         ), '')) FROM public.lab_arena_submissions AS submission
         WHERE round_id = v_round_id) IS DISTINCT FROM v_submissions_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(credential)::TEXT, '|'
           ORDER BY submission_id, provider
         ), '')) FROM public.lab_arena_submission_credentials AS credential
         WHERE submission_id IN (
           SELECT participant ->> 'submission_id'
           FROM pg_catalog.jsonb_array_elements(v_round.participants)
             AS participant
         )) IS DISTINCT FROM v_credentials_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(other_round)::TEXT, '|'
           ORDER BY round_id
         ), '')) FROM public.lab_arena_rounds AS other_round
         WHERE round_id <> v_round_id) IS DISTINCT FROM v_unrelated_hash
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once' AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 future scorer verification failed';
  END IF;
END;
$lab_arena_238_bind_future_stage_scorer$;

COMMIT;
