-- Resume the cancelled stage-1 scoring for arena-2026-09-13 after the
-- company-level evidence-failure isolation fix. Preserve every accepted
-- execution and score. Requeue only the 88 incomplete score assignments
-- under a fresh provider-call namespace and bind those claims to the fixed
-- scorer image.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE MODE;
LOCK TABLE public.lab_arena_judgment_cache IN SHARE MODE;
LOCK TABLE public.lab_arena_company_judgment_reservations IN SHARE MODE;

DO $lab_arena_234_recover_20260913_scoring$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-13';
  v_suffix CONSTANT TEXT := ':recovery234';
  v_old_digest CONSTANT TEXT :=
    'sha256:b6e58855c4962faa345bd785e0d9aff3210457f5cf2d1dd3b4ee8e29f0b69757';
  v_old_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:b6e58855c4962faa345bd785e0d9aff3210457f5cf2d1dd3b4ee8e29f0b69757';
  v_new_digest CONSTANT TEXT :=
    'sha256:f1812fbedf8700fd491fea94c576917018cc6fc1970a124784e22359eabaee4d';
  v_new_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:f1812fbedf8700fd491fea94c576917018cc6fc1970a124784e22359eabaee4d';
  v_participant_ids CONSTANT TEXT[] := ARRAY[
    'baseline-2026-09-13',
    'sub-00ee4222188195476cdcf63fdb94171f',
    'sub-028c4e5c655e855f1c343fa39274cd2c',
    'sub-1728be05a36000fcdb781a11d35594cd',
    'sub-5c1f20eb379bcd4e11c29912b5251b0d',
    'sub-5d27bd75ca8c5999ae22f28a0a244660',
    'sub-5dffdbaa2b96e8dc78160aea8f80a7b9',
    'sub-5ff557c97e83dcddd7b266226e16239c',
    'sub-6211e8d46819c34df3418ded36f788ef',
    'sub-ca0c9c5da2d2b203f246258213c27263',
    'sub-caf0e1ef30c9712e6385afe24a75375e',
    'sub-cb17cc2f2f351c1e92a59b85e35333c3'
  ]::TEXT[];
  v_round public.lab_arena_rounds%ROWTYPE;
  v_live_participants TEXT[];
  v_incomplete_assignments TEXT[];
  v_latest_run_ids TEXT[];
  v_count INTEGER;
  v_accepted_hash TEXT;
  v_non_score_hash TEXT;
  v_submission_hash TEXT;
  v_credential_hash TEXT;
  v_ledger_hash TEXT;
  v_cache_hash TEXT;
  v_round_stable JSONB;
  v_configuration_stable JSONB;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_close_scoring(text,smallint)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'extensions.digest(bytea,text)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 234';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN;
  END IF;

  -- A completed application stays a no-op while normal scoring advances.
  SELECT pg_catalog.count(DISTINCT assignment_id) INTO v_count
  FROM public.lab_arena_runs
  WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
    AND assignment_id LIKE '%' || v_suffix;
  IF v_count > 0 THEN
    IF v_count <> 88
       OR v_round.status = 'cancelled'
       OR v_round.status NOT IN (
         'stage1_scoring', 'stage1_judged', 'stage1_scored', 'stage2',
         'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored',
         'stage3', 'stage3_closed', 'stage3_scoring', 'stage3_judged',
         'confirmed', 'published'
       )
       OR v_round.configuration_doc ->> 'scorer_image_digest'
          IS DISTINCT FROM v_new_digest
       OR v_round.configuration_doc ->> 'scorer_image_reference'
          IS DISTINCT FROM v_new_reference THEN
      RAISE EXCEPTION 'arena 2026-09-13 recovery replay state differs';
    END IF;
    RETURN;
  END IF;

  IF NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_runs'::pg_catalog.regclass
         AND tgname = 'lab_arena_runs_terminal' AND tgenabled = 'O'
         AND NOT tgisinternal
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once' AND tgenabled = 'O'
         AND NOT tgisinternal
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_ledger'::pg_catalog.regclass
         AND tgname = 'lab_arena_ledger_append_only' AND tgenabled = 'O'
         AND NOT tgisinternal
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 recovery guard state differs';
  END IF;

  IF v_round.status <> 'cancelled'
     OR v_round.cancel_reason <> 'scoring_incomplete'
     OR v_round.status_generation <> 5
     OR v_round.stage_generation <> 4
     OR v_round.evaluation_date <> '2026-09-13'
     OR v_round.icp_set_date <> DATE '2026-09-12'
     OR v_round.publication_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR pg_catalog.jsonb_typeof(v_round.participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 12
     OR pg_catalog.jsonb_typeof(v_round.stage1_scoring_plan_doc)
        IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'work_items'
        ) <> 110
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'zero_rows'
        ) <> 10
     OR v_round.configuration_doc ->> 'max_attempts_per_assignment'
        IS DISTINCT FROM '2'
     OR v_round.configuration_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_old_digest
     OR v_round.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_old_reference
     OR v_round.configuration_doc #>> '{schedule,submission_open}'
        IS DISTINCT FROM '2026-09-12T00:00:00Z'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
        IS DISTINCT FROM '2026-09-13T00:00:00Z'
     OR v_round.configuration_doc #>> '{schedule,stage_1_scoring_close}'
        IS DISTINCT FROM '2026-09-13T11:00:01Z'
     OR v_round.configuration_doc #>> '{schedule,stage_2_start}'
        IS DISTINCT FROM '2026-09-13T11:00:02Z' THEN
    RAISE EXCEPTION 'arena 2026-09-13 frozen round state differs';
  END IF;

  SELECT pg_catalog.array_agg(
           item ->> 'submission_id' ORDER BY item ->> 'submission_id'
         )
  INTO v_live_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants) AS item;
  IF v_live_participants IS DISTINCT FROM v_participant_ids
     OR (
       SELECT pg_catalog.array_agg(submission_id ORDER BY submission_id)
       FROM public.lab_arena_submissions
       WHERE round_id = v_round_id AND status = 'frozen'
     ) IS DISTINCT FROM v_participant_ids
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submission_credentials
         WHERE submission_id = ANY(v_participant_ids)) <> 24 THEN
    RAISE EXCEPTION 'arena 2026-09-13 participant state differs';
  END IF;

  IF EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_company_judgment_reservations AS reservation
       JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
       WHERE run.round_id = v_round_id
     )
     OR EXISTS (
       SELECT 1 FROM (
         SELECT DISTINCT ON (ledger.call_identity)
           ledger.entry_kind, run.kind
         FROM public.lab_arena_ledger AS ledger
         JOIN public.lab_arena_runs AS run ON run.run_id = ledger.run_id
         WHERE ledger.round_id = v_round_id
           AND ledger.call_identity IS NOT NULL
         ORDER BY ledger.call_identity, ledger.entry_id DESC
       ) AS heads
       WHERE heads.kind = 'score'
         AND heads.entry_kind IN ('reservation', 'dispatch')
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 scoring recovery preflight differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
        AND status = 'accepted' AND terminal_cause = 'accepted') <> 110
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND status = 'failed' AND terminal_cause = 'credential_error') <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND status = 'failed' AND terminal_cause = 'lease_expired') <> 2
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 116
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 110
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 22
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'judge_error') <> 7
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'stage_closed') <> 87
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_judgment_cache
         WHERE scope_doc ->> 'round_id' = v_round_id) <> 11
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
         AND company_judgment_refs IS NOT NULL
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 run state differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE assignment_id =
        'arena-2026-09-13:sub-00ee4222188195476cdcf63fdb94171f:1:1:score'
        AND attempt IN (1, 2) AND status = 'failed'
        AND terminal_cause = 'judge_error') <> 2 THEN
    RAISE EXCEPTION 'arena 2026-09-13 triggering failure differs';
  END IF;

  WITH latest AS (
    SELECT DISTINCT ON (assignment_id)
      assignment_id, run_id, status, terminal_cause
    FROM public.lab_arena_runs
    WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
    ORDER BY assignment_id, (status = 'accepted') DESC, attempt DESC
  )
  SELECT pg_catalog.array_agg(assignment_id ORDER BY assignment_id),
         pg_catalog.array_agg(run_id ORDER BY assignment_id)
  INTO v_incomplete_assignments, v_latest_run_ids
  FROM latest
  WHERE status <> 'accepted';
  IF pg_catalog.array_length(v_incomplete_assignments, 1) <> 88
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND assignment_id = ANY(v_incomplete_assignments)) <> 92
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE run_id = ANY(v_latest_run_ids)
           AND terminal_cause = 'judge_error') <> 1
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE run_id = ANY(v_latest_run_ids)
           AND terminal_cause = 'stage_closed') <> 87
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(
         v_round.stage1_scoring_plan_doc -> 'work_items'
       ) AS plan(item)
       LEFT JOIN public.lab_arena_runs AS execution
         ON execution.run_id = plan.item ->> 'scored_run_id'
        AND execution.round_id = v_round_id
        AND execution.kind = 'execute'
        AND execution.status = 'accepted'
        AND execution.submission_id = plan.item ->> 'submission_id'
        AND execution.icp_position =
            (plan.item ->> 'icp_position')::SMALLINT
        AND execution.output_ref = plan.item ->> 'output_ref'
       WHERE execution.run_id IS NULL
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS score
       WHERE score.round_id = v_round_id AND score.stage = 1
         AND score.kind = 'score'
         AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.jsonb_array_elements(
             v_round.stage1_scoring_plan_doc -> 'work_items'
           ) AS plan(item)
           WHERE plan.item ->> 'scored_run_id' = score.scored_run_id
         )
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 incomplete scoring set differs';
  END IF;

  IF EXISTS (
       SELECT 1
       FROM public.lab_arena_runs AS run
       WHERE run.run_id = ANY(v_latest_run_ids)
         AND (
           pg_catalog.jsonb_typeof(run.judgment_scope_doc)
             IS DISTINCT FROM 'object'
           OR ARRAY(
             SELECT field.key
             FROM pg_catalog.jsonb_each(
               CASE
                 WHEN pg_catalog.jsonb_typeof(run.judgment_scope_doc) = 'object'
                 THEN run.judgment_scope_doc
                 ELSE '{}'::JSONB
               END
             ) AS field
             ORDER BY field.key COLLATE "C"
           ) IS DISTINCT FROM ARRAY[
             'cache_key', 'evaluation_date', 'integrity_policy', 'netuid',
             'network_name', 'round_id', 'schema_version',
             'scorer_image_digest', 'scorer_image_reference',
             'scoring_input_hash'
           ]::TEXT[]
           OR run.judgment_scope_doc ->> 'cache_key'
             IS DISTINCT FROM run.judgment_cache_key
           OR run.judgment_scope_doc ->> 'evaluation_date'
             IS DISTINCT FROM '2026-09-13'
           OR run.judgment_scope_doc ->> 'integrity_policy'
             IS DISTINCT FROM 'arena_integrity_v1'
           OR run.judgment_scope_doc ->> 'netuid' IS DISTINCT FROM '71'
           OR run.judgment_scope_doc ->> 'network_name'
             IS DISTINCT FROM 'finney'
           OR run.judgment_scope_doc ->> 'round_id'
             IS DISTINCT FROM v_round_id
           OR run.judgment_scope_doc ->> 'schema_version'
             IS DISTINCT FROM 'leadpoet.lab_arena.judgment_cache_scope.v1'
           OR run.judgment_scope_doc ->> 'scorer_image_digest'
             IS DISTINCT FROM v_old_digest
           OR run.judgment_scope_doc ->> 'scorer_image_reference'
             IS DISTINCT FROM v_old_reference
           OR run.judgment_scope_doc ->> 'scoring_input_hash'
             IS DISTINCT FROM run.judgment_input_hash
           OR run.judgment_cache_key !~ '^sha256:[0-9a-f]{64}$'
           OR run.judgment_input_hash !~ '^sha256:[0-9a-f]{64}$'
         )
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 judgment scope differs';
  END IF;

  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         )) INTO v_accepted_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id AND status = 'accepted';
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         )) INTO v_non_score_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id AND kind <> 'score';
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(submission)::TEXT, '|' ORDER BY submission_id
         )) INTO v_submission_hash
  FROM public.lab_arena_submissions AS submission
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(credential)::TEXT, '|'
           ORDER BY submission_id, provider
         )) INTO v_credential_hash
  FROM public.lab_arena_submission_credentials AS credential
  WHERE submission_id = ANY(v_participant_ids);
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
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'status' - 'status_generation' - 'stage_generation'
    - 'cancel_reason' - 'updated_at' - 'configuration_doc';
  v_configuration_stable := v_round.configuration_doc
    - 'scorer_image_digest' - 'scorer_image_reference';

  -- Changing the assignment namespace makes every recovered provider call
  -- identity fresh. The private pre-recovery snapshot retains the exact
  -- terminal form of the reset rows at
  -- arena/arena-2026-09-13/recovery/company-verification-20260913.json
  -- (SHA-256 842f75c689ec2b28e44fe3dc9e662fa501e38d5fb96906f53fe6116eb7c56819).
  -- Accepted rows and all cost rows remain byte-for-byte unchanged here.
  ALTER TABLE public.lab_arena_runs
    DISABLE TRIGGER lab_arena_runs_terminal;
  UPDATE public.lab_arena_runs
  SET assignment_id = assignment_id || v_suffix
  WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
    AND assignment_id = ANY(v_incomplete_assignments);
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 92 THEN
    RAISE EXCEPTION 'arena 2026-09-13 recovery lineage count differs';
  END IF;

  WITH scope_bodies AS (
    SELECT run_id,
      (judgment_scope_doc - 'cache_key') || pg_catalog.jsonb_build_object(
        'scorer_image_digest', v_new_digest,
        'scorer_image_reference', v_new_reference
      ) AS scope_body
    FROM public.lab_arena_runs
    WHERE run_id = ANY(v_latest_run_ids)
  ), canonical AS (
    SELECT scope.run_id, scope.scope_body,
      '{' || pg_catalog.string_agg(
        pg_catalog.to_jsonb(field.key)::TEXT || ':' || field.value::TEXT,
        ',' ORDER BY field.key COLLATE "C"
      ) || '}' AS canonical_json
    FROM scope_bodies AS scope
    CROSS JOIN LATERAL pg_catalog.jsonb_each(scope.scope_body) AS field
    GROUP BY scope.run_id, scope.scope_body
  ), keyed AS (
    SELECT run_id, scope_body,
      'sha256:' || pg_catalog.encode(
        extensions.digest(
          pg_catalog.convert_to(canonical_json, 'UTF8'),
          'sha256'
        ),
        'hex'
      ) AS cache_key
    FROM canonical
  )
  UPDATE public.lab_arena_runs AS run
  SET status = 'pending',
      runner_hotkey = NULL,
      lease_token_hash = NULL,
      lease_generation = run.lease_generation + 1,
      stage_generation = 5,
      lease_expires_at = NULL,
      claim_request_id = NULL,
      claim_request_hash = NULL,
      claim_response = NULL,
      result_doc = NULL,
      output_ref = NULL,
      terminal_cause = NULL,
      terminal_doc = NULL,
      per_icp_score = NULL,
      judgment_cache_key = keyed.cache_key,
      judgment_scope_doc = keyed.scope_body || pg_catalog.jsonb_build_object(
        'cache_key', keyed.cache_key
      ),
      judgment_cache_source_run_id = NULL
  FROM keyed
  WHERE run.run_id = keyed.run_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 88 THEN
    RAISE EXCEPTION 'arena 2026-09-13 requeue count differs';
  END IF;

  WITH groups AS (
    SELECT judgment_cache_key,
           pg_catalog.min(run_id) AS leader_run_id,
           pg_catalog.array_agg(
             DISTINCT miner_hotkey ORDER BY miner_hotkey
           ) AS miner_hotkeys
    FROM public.lab_arena_runs
    WHERE run_id = ANY(v_latest_run_ids)
    GROUP BY judgment_cache_key
  )
  UPDATE public.lab_arena_runs AS run
  SET judgment_group_leader = run.run_id = groups.leader_run_id,
      judgment_group_miner_hotkeys = groups.miner_hotkeys
  FROM groups
  WHERE run.run_id = ANY(v_latest_run_ids)
    AND run.judgment_cache_key = groups.judgment_cache_key;

  ALTER TABLE public.lab_arena_runs
    ENABLE TRIGGER lab_arena_runs_terminal;

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1_scoring',
      status_generation = 6,
      stage_generation = 5,
      cancel_reason = NULL,
      configuration_doc = pg_catalog.jsonb_set(
        pg_catalog.jsonb_set(
          configuration_doc,
          '{scorer_image_digest}', pg_catalog.to_jsonb(v_new_digest), FALSE
        ),
        '{scorer_image_reference}', pg_catalog.to_jsonb(v_new_reference), FALSE
      )
  WHERE round_id = v_round_id AND status = 'cancelled'
    AND cancel_reason = 'scoring_incomplete'
    AND status_generation = 5 AND stage_generation = 4;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-13 round restart count differs';
  END IF;

  IF (SELECT pg_catalog.count(DISTINCT assignment_id)
      FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
        AND assignment_id LIKE '%' || v_suffix) <> 88
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE run_id = ANY(v_latest_run_ids) AND status = 'pending'
           AND stage_generation = 5
           AND assignment_id LIKE '%' || v_suffix) <> 88
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE run_id = ANY(v_latest_run_ids)
         AND (
           judgment_scope_doc ->> 'scorer_image_digest'
             IS DISTINCT FROM v_new_digest
           OR judgment_scope_doc ->> 'scorer_image_reference'
             IS DISTINCT FROM v_new_reference
           OR judgment_scope_doc ->> 'cache_key'
             IS DISTINCT FROM judgment_cache_key
           OR judgment_cache_source_run_id IS NOT NULL
         )
     )
     OR EXISTS (
       SELECT judgment_cache_key
       FROM public.lab_arena_runs
       WHERE run_id = ANY(v_latest_run_ids)
       GROUP BY judgment_cache_key
       HAVING pg_catalog.count(*) FILTER (
         WHERE COALESCE(judgment_group_leader, FALSE)
       ) <> 1
     )
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id AND status = 'accepted')
        IS DISTINCT FROM v_accepted_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id AND kind <> 'score')
        IS DISTINCT FROM v_non_score_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(submission)::TEXT, '|' ORDER BY submission_id
         )) FROM public.lab_arena_submissions AS submission
         WHERE round_id = v_round_id) IS DISTINCT FROM v_submission_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(credential)::TEXT, '|'
           ORDER BY submission_id, provider
         )) FROM public.lab_arena_submission_credentials AS credential
         WHERE submission_id = ANY(v_participant_ids))
        IS DISTINCT FROM v_credential_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(ledger)::TEXT, '|' ORDER BY entry_id
         ), '')) FROM public.lab_arena_ledger AS ledger
         WHERE round_id = v_round_id) IS DISTINCT FROM v_ledger_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(cache_row)::TEXT, '|' ORDER BY cache_key
         ), '')) FROM public.lab_arena_judgment_cache AS cache_row
         WHERE scope_doc ->> 'round_id' = v_round_id)
        IS DISTINCT FROM v_cache_hash
     OR (SELECT pg_catalog.to_jsonb(round_row)
           - 'status' - 'status_generation' - 'stage_generation'
           - 'cancel_reason' - 'updated_at' - 'configuration_doc'
         FROM public.lab_arena_rounds AS round_row
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR (SELECT configuration_doc
           - 'scorer_image_digest' - 'scorer_image_reference'
         FROM public.lab_arena_rounds
         WHERE round_id = v_round_id) IS DISTINCT FROM v_configuration_stable
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_round_id AND status = 'stage1_scoring'
         AND status_generation = 6 AND stage_generation = 5
         AND cancel_reason IS NULL
         AND configuration_doc ->> 'scorer_image_digest' = v_new_digest
         AND configuration_doc ->> 'scorer_image_reference' = v_new_reference
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_runs'::pg_catalog.regclass
         AND tgname = 'lab_arena_runs_terminal' AND tgenabled = 'O'
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once' AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 recovery verification failed';
  END IF;
END;
$lab_arena_234_recover_20260913_scoring$;

COMMIT;
