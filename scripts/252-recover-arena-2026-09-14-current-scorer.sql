-- Resume only the three September 14 stage-1 scores that remain unfinished.
-- Bind the new assignments to the fixed scorer image while every historical
-- run, receipt, cost row, credential, submission, and cache row stays immutable.

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

DO $lab_arena_252_recover_20260914_current_scorer$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-14';
  v_target_submission CONSTANT TEXT :=
    'sub-dd76cafd44732cb230a476ab031b24ad';
  v_recovery_suffix CONSTANT TEXT := ':recovery252';
  v_prior_suffix CONSTANT TEXT := ':recovery249';
  v_target_assignments CONSTANT TEXT[] := ARRAY[
    'arena-2026-09-14:sub-dd76cafd44732cb230a476ab031b24ad:1:5:score:recovery252',
    'arena-2026-09-14:sub-dd76cafd44732cb230a476ab031b24ad:1:6:score:recovery252',
    'arena-2026-09-14:sub-dd76cafd44732cb230a476ab031b24ad:1:8:score:recovery252'
  ]::TEXT[];
  v_participant_ids CONSTANT TEXT[] := ARRAY[
    'baseline-2026-09-14',
    'sub-09f530ce94b6f63221ce4d69962e3988',
    'sub-0bee4ffefcb6a792ca70d16d722fa1ac',
    'sub-287f56f3e3718776a6aa10e4c130e154',
    'sub-2968d4fba9582f9d8881bfd998290cbe',
    'sub-4c5b60cfe00763f2d022031ecbd40039',
    'sub-4e87858d429552dd34271afa3be03a9b',
    'sub-804245a7d75aa38aa4c74a59e07b51d2',
    'sub-8ec81a864da4e10204d803a3fb7c3b58',
    'sub-946ff9115aec053aeb36a02476175b82',
    'sub-a7a50fba0fe795bc6f87a433fb6682f3',
    'sub-dae52a6a6cca8db5914a259116bc1b56',
    'sub-dd76cafd44732cb230a476ab031b24ad'
  ]::TEXT[];
  v_old_digest CONSTANT TEXT :=
    'sha256:412beed7799ecfcceb9c07c44f36644f8214c33513a117d603dede98c0b918e9';
  v_old_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'
    || v_old_digest;
  v_new_digest CONSTANT TEXT :=
    'sha256:748e08acbbdf55bbc46f8d510fe9404d4ed31e575bb78f6cd56f32d4caf849bf';
  v_new_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'
    || v_new_digest;
  v_expected_schedule CONSTANT JSONB := pg_catalog.jsonb_build_object(
    'submission_open', '2026-09-13T00:00:00Z',
    'submission_cutoff', '2026-09-14T00:00:00Z',
    'benchmark_deadline', '2026-09-14T00:30:00Z',
    'stage_1_start', '2026-09-14T00:30:01Z',
    'stage_1_close', '2026-09-14T04:30:01Z',
    'stage_1_scoring_close', '2026-09-14T22:20:07.492848Z',
    'stage_2_start', '2026-09-14T22:20:08.492848Z',
    'stage_2_close', '2026-09-15T01:20:08.492848Z',
    'final_scoring_close', '2026-09-15T07:50:08.492848Z',
    'stage_3_start', '2026-09-15T07:50:09.492848Z',
    'stage_3_close', '2026-09-15T08:50:09.492848Z',
    'stage_3_scoring_close', '2026-09-15T10:40:09.492848Z',
    'publication_deadline', '2026-09-15T10:40:10.492848Z'
  );
  v_expected_configuration_hash CONSTANT TEXT :=
    '6c210f6fdf20d410eaec559955fe0507';
  v_expected_runs_hash CONSTANT TEXT :=
    'ae0ead0f07863cf4caaa39d69f5dd612';
  v_expected_accepted_hash CONSTANT TEXT :=
    '8b7d9b0c7da399fbb4bc7b5893391a1a';
  v_expected_ledger_hash CONSTANT TEXT :=
    'c81e427f8afdb076aeff6b3bdf7b82a4';
  v_expected_submission_hash CONSTANT TEXT :=
    '939df0f1d86a931feb384115b81e3bed';
  v_expected_credential_hash CONSTANT TEXT :=
    'e1834999a30b26c36f7759e9d8d8a628';
  v_expected_cache_hash CONSTANT TEXT :=
    '9488bb4df1ac2e0e4b2e66848dfeca16';

  v_round public.lab_arena_rounds%ROWTYPE;
  v_runs_hash TEXT;
  v_accepted_hash TEXT;
  v_ledger_hash TEXT;
  v_submission_hash TEXT;
  v_credential_hash TEXT;
  v_cache_hash TEXT;
  v_reservation_hash TEXT;
  v_unrelated_hash TEXT;
  v_round_stable JSONB;
  v_configuration_stable JSONB;
  v_schema JSONB;
  v_definition TEXT;
  v_stage1_scoring_close TIMESTAMPTZ;
  v_missing_positions INTEGER[];
  v_missing_submissions TEXT[];
  v_count INTEGER;
BEGIN
  IF v_new_digest !~ '^sha256:[0-9a-f]{64}$'
     OR v_new_digest = v_old_digest
     OR v_new_reference !~ '@sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 scorer digest is not frozen';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN;
  END IF;

  -- A replay is a no-op only for the complete new namespace with the exact
  -- logical bindings. Later retries and terminal progress remain valid.
  SELECT pg_catalog.count(*) INTO v_count
  FROM public.lab_arena_runs
  WHERE round_id = v_round_id
    AND assignment_id LIKE '%' || v_recovery_suffix;
  IF v_count > 0 THEN
    IF (SELECT pg_catalog.count(*)
        FROM public.lab_arena_runs
        WHERE round_id = v_round_id
          AND assignment_id LIKE '%' || v_recovery_suffix
          AND attempt = 1) <> 3
       OR (SELECT pg_catalog.array_agg(DISTINCT assignment_id ORDER BY assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id
             AND assignment_id LIKE '%' || v_recovery_suffix)
          IS DISTINCT FROM v_target_assignments
       OR v_round.configuration_doc ->> 'scorer_image_digest'
          IS DISTINCT FROM v_new_digest
       OR v_round.configuration_doc ->> 'scorer_image_reference'
          IS DISTINCT FROM v_new_reference
       OR EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS recovered
         WHERE recovered.round_id = v_round_id
           AND recovered.assignment_id LIKE '%' || v_recovery_suffix
           AND (
             recovered.submission_id <> v_target_submission
             OR recovered.stage <> 1
             OR recovered.kind <> 'score'
             OR recovered.icp_position NOT IN (5, 6, 8)
             OR recovered.assignment_id <> v_round_id || ':'
                || recovered.submission_id || ':1:'
                || recovered.icp_position::TEXT || ':score' || v_recovery_suffix
             OR recovered.scored_run_id <> v_round_id || ':'
                || recovered.submission_id || ':1:'
                || recovered.icp_position::TEXT || ':1'
             OR recovered.stage_generation <> 10
             OR recovered.judgment_scope_doc ->> 'scorer_image_digest'
                IS DISTINCT FROM v_new_digest
             OR recovered.judgment_scope_doc ->> 'scorer_image_reference'
                IS DISTINCT FROM v_new_reference
             OR recovered.judgment_scope_doc ->> 'cache_key'
                IS DISTINCT FROM recovered.judgment_cache_key
           )
       )
       OR EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS recovered
         LEFT JOIN LATERAL (
           SELECT prior.*
           FROM public.lab_arena_runs AS prior
           WHERE prior.round_id = v_round_id
             AND prior.submission_id = recovered.submission_id
             AND prior.icp_position = recovered.icp_position
             AND prior.scored_run_id = recovered.scored_run_id
             AND prior.assignment_id LIKE '%' || v_prior_suffix
           ORDER BY prior.stage_generation DESC, prior.attempt DESC,
             prior.created_at DESC, prior.run_id DESC
           LIMIT 1
         ) AS prior ON TRUE
         WHERE recovered.round_id = v_round_id
           AND recovered.assignment_id LIKE '%' || v_recovery_suffix
           AND recovered.attempt = 1
           AND (
             prior.run_id IS NULL
             OR recovered.judgment_input_hash
                IS DISTINCT FROM prior.judgment_input_hash
             OR recovered.company_judgment_refs
                IS DISTINCT FROM prior.company_judgment_refs
             OR recovered.judgment_scope_doc
                  - 'cache_key' - 'scorer_image_digest' - 'scorer_image_reference'
                IS DISTINCT FROM prior.judgment_scope_doc
                  - 'cache_key' - 'scorer_image_digest' - 'scorer_image_reference'
           )
       ) THEN
      RAISE EXCEPTION 'arena 2026-09-14 recovery252 replay state differs';
    END IF;
    RETURN;
  END IF;

  IF pg_catalog.to_regprocedure(
       'public.lab_arena_deepline_cost_reconciliation_schema_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_twenty_icp_promotion_schema_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)'
     ) IS NULL
     OR pg_catalog.to_regprocedure('extensions.digest(bytea,text)') IS NULL THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 prerequisite missing';
  END IF;
  SELECT public.lab_arena_deepline_cost_reconciliation_schema_v1()
  INTO v_schema;
  IF (v_schema ->> 'version')::INTEGER < 248 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 prerequisite version differs';
  END IF;
  SELECT public.lab_arena_twenty_icp_promotion_schema_v1()
  INTO v_schema;
  IF v_schema ->> 'schema_version' IS DISTINCT FROM
       'leadpoet.lab_arena.twenty_icp_promotion_schema.v1'
     OR (v_schema ->> 'version')::INTEGER < 251 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 promotion prerequisite differs';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
      ::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_closed_scoring_reservation_admission'
     ) = 0 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 reserve prerequisite missing';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
      ::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_closed_scoring_reservation_claim'
     ) = 0 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 claim prerequisite missing';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'
      ::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_closed_scoring_billing_reconciliation'
     ) = 0 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 billing prerequisite missing';
  END IF;

  IF v_round.status <> 'cancelled'
     OR v_round.cancel_reason <> 'scoring_incomplete'
     OR v_round.status_generation <> 10
     OR v_round.stage_generation <> 9
     OR v_round.evaluation_date <> '2026-09-14'
     OR v_round.icp_set_date <> '2026-09-13'
     OR v_round.published_at IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR pg_catalog.md5(v_round.configuration_doc::TEXT)
        IS DISTINCT FROM v_expected_configuration_hash
     OR v_round.configuration_doc -> 'schedule' IS DISTINCT FROM v_expected_schedule
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_old_digest
     OR v_round.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_old_reference
     OR pg_catalog.jsonb_typeof(v_round.participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 13
     OR (
       SELECT pg_catalog.array_agg(
         item ->> 'submission_id' ORDER BY item ->> 'submission_id'
       )
       FROM pg_catalog.jsonb_array_elements(v_round.participants)
         AS participant(item)
     ) IS DISTINCT FROM v_participant_ids
     OR pg_catalog.jsonb_typeof(v_round.stage1_scoring_plan_doc)
        IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'work_items'
        ) <> 130
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'zero_rows'
        ) <> 0
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submissions
         WHERE round_id = v_round_id AND status = 'frozen') <> 13
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submission_credentials
         WHERE submission_id = ANY(v_participant_ids)) <> 28 THEN
    RAISE EXCEPTION 'arena 2026-09-14 frozen round state differs for recovery252';
  END IF;

  -- Bind the exact terminal post-recovery249 state.
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) INTO v_runs_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) INTO v_accepted_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id AND status = 'accepted';
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(ledger)::TEXT), '|' ORDER BY entry_id
         ), '')) INTO v_ledger_hash
  FROM public.lab_arena_ledger AS ledger
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(submission)::TEXT),
           '|' ORDER BY submission_id
         )) INTO v_submission_hash
  FROM public.lab_arena_submissions AS submission
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(credential)::TEXT),
           '|' ORDER BY submission_id, provider
         )) INTO v_credential_hash
  FROM public.lab_arena_submission_credentials AS credential
  WHERE submission_id = ANY(v_participant_ids);
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(cache_row)::TEXT),
           '|' ORDER BY cache_key
         ), '')) INTO v_cache_hash
  FROM public.lab_arena_judgment_cache AS cache_row
  WHERE scope_doc ->> 'round_id' = v_round_id;

  IF v_runs_hash IS DISTINCT FROM v_expected_runs_hash
     OR v_accepted_hash IS DISTINCT FROM v_expected_accepted_hash
     OR v_ledger_hash IS DISTINCT FROM v_expected_ledger_hash
     OR v_submission_hash IS DISTINCT FROM v_expected_submission_hash
     OR v_credential_hash IS DISTINCT FROM v_expected_credential_hash
     OR v_cache_hash IS DISTINCT FROM v_expected_cache_hash
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id) <> 362
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 130
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 232
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 226
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND attempt = 2) <> 6
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 127
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'credential_error') <> 2
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'judge_error') <> 8
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'stage_closed') <> 95
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id
           AND assignment_id LIKE '%' || v_prior_suffix) <> 4
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id
           AND assignment_id LIKE '%' || v_prior_suffix) <> 3
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = v_round_id) <> 12315
     OR (SELECT pg_catalog.max(entry_id) FROM public.lab_arena_ledger
         WHERE round_id = v_round_id) <> 350351
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_judgment_cache
         WHERE scope_doc ->> 'round_id' = v_round_id) <> 56
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_company_judgment_reservations AS reservation
       JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
       WHERE run.round_id = v_round_id
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-14 terminal snapshot differs for recovery252';
  END IF;

  IF (WITH latest_missing AS (
        SELECT DISTINCT ON (
          run.submission_id, run.icp_position, run.scored_run_id
        ) run.*
        FROM public.lab_arena_runs AS run
        WHERE run.round_id = v_round_id
          AND run.stage = 1
          AND run.kind = 'score'
          AND run.submission_id = v_target_submission
          AND run.icp_position IN (5, 6, 8)
        ORDER BY run.submission_id, run.icp_position, run.scored_run_id,
          run.stage_generation DESC, run.attempt DESC,
          run.created_at DESC, run.run_id DESC
      ), expected(position, attempt, terminal_cause, input_hash, cache_key) AS (
        VALUES
          (5, 2, 'judge_error',
           'sha256:ba498c4b7b9ee0a5d88d6a1db36b391ae157407dd91f5b0ed1f694ac4c291f77',
           'sha256:1cb539d3dc9a882b80d37b12f3668bf7832cb6ac527b2e3ef2a8162a4422b0e0'),
          (6, 1, 'stage_closed',
           'sha256:1b62c23c84742552d518183fcb54a71b1cd475ed78b056189926d76d710b5498',
           'sha256:a0c4667c4694b8e616ef90abed01c35e970a4a76918003a8bf473bf6ebcdf66e'),
          (8, 1, 'stage_closed',
           'sha256:1e85fe814989df84bf1ec29ef820288d093187d926ac35c01e3782f3db83cdc6',
           'sha256:d15d000e5f5c738c520789573718742a1fd2c36aa613251789f153f4367f2f51')
      )
      SELECT pg_catalog.count(*)
      FROM latest_missing AS latest
      JOIN expected
        ON expected.position = latest.icp_position
       AND expected.attempt = latest.attempt
       AND expected.terminal_cause = latest.terminal_cause
       AND expected.input_hash = latest.judgment_input_hash
       AND expected.cache_key = latest.judgment_cache_key
      WHERE latest.status = 'failed'
        AND latest.assignment_id LIKE '%' || v_prior_suffix
        AND latest.stage_generation = 8
        AND latest.judgment_scope_doc ->> 'scorer_image_digest' = v_old_digest
        AND latest.judgment_scope_doc ->> 'scorer_image_reference' = v_old_reference
        AND latest.company_judgment_refs IS NULL) <> 3 THEN
    RAISE EXCEPTION 'arena 2026-09-14 prior recovery binding differs for recovery252';
  END IF;

  IF EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(
         v_round.stage1_scoring_plan_doc -> 'work_items'
       ) AS plan(item)
       WHERE NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS execution
         WHERE execution.round_id = v_round_id
           AND execution.kind = 'execute'
           AND execution.status = 'accepted'
           AND execution.run_id = plan.item ->> 'scored_run_id'
           AND execution.submission_id = plan.item ->> 'submission_id'
           AND execution.icp_position =
             (plan.item ->> 'icp_position')::SMALLINT
           AND execution.output_ref = plan.item ->> 'output_ref'
       )
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-14 execution plan binding differs for recovery252';
  END IF;
  WITH missing AS (
    SELECT plan.item
    FROM pg_catalog.jsonb_array_elements(
      v_round.stage1_scoring_plan_doc -> 'work_items'
    ) AS plan(item)
    WHERE NOT EXISTS (
      SELECT 1
      FROM public.lab_arena_runs AS accepted
      WHERE accepted.round_id = v_round_id
        AND accepted.kind = 'score'
        AND accepted.status = 'accepted'
        AND accepted.submission_id = plan.item ->> 'submission_id'
        AND accepted.icp_position =
          (plan.item ->> 'icp_position')::SMALLINT
        AND accepted.scored_run_id = plan.item ->> 'scored_run_id'
    )
  )
  SELECT pg_catalog.count(*),
    pg_catalog.array_agg(
      (item ->> 'icp_position')::INTEGER
      ORDER BY (item ->> 'icp_position')::INTEGER
    ),
    pg_catalog.array_agg(
      item ->> 'submission_id'
      ORDER BY (item ->> 'icp_position')::INTEGER
    )
  INTO v_count, v_missing_positions, v_missing_submissions
  FROM missing;
  IF v_count <> 3
     OR v_missing_positions IS DISTINCT FROM ARRAY[5, 6, 8]::INTEGER[]
     OR v_missing_submissions IS DISTINCT FROM ARRAY[
       v_target_submission, v_target_submission, v_target_submission
     ]::TEXT[] THEN
    RAISE EXCEPTION 'arena 2026-09-14 unfinished logical set differs for recovery252';
  END IF;

  -- Snapshot rows that the transaction must preserve.
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(reservation)::TEXT),
           '|' ORDER BY reservation.cache_key, reservation.authority_slot
         ), '')) INTO v_reservation_hash
  FROM public.lab_arena_company_judgment_reservations AS reservation
  JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
  WHERE run.round_id = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(other_round)::TEXT),
           '|' ORDER BY round_id
         ), '')) INTO v_unrelated_hash
  FROM public.lab_arena_rounds AS other_round
  WHERE round_id <> v_round_id;
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'status' - 'status_generation' - 'stage_generation'
    - 'cancel_reason' - 'configuration_doc' - 'updated_at';
  v_configuration_stable := v_round.configuration_doc
    - 'schedule' - 'scorer_image_digest' - 'scorer_image_reference';

  WITH latest_missing AS (
    SELECT DISTINCT ON (
      run.submission_id, run.icp_position, run.scored_run_id
    ) run.*
    FROM public.lab_arena_runs AS run
    WHERE run.round_id = v_round_id
      AND run.stage = 1
      AND run.kind = 'score'
      AND run.submission_id = v_target_submission
      AND run.icp_position IN (5, 6, 8)
    ORDER BY run.submission_id, run.icp_position, run.scored_run_id,
      run.stage_generation DESC, run.attempt DESC,
      run.created_at DESC, run.run_id DESC
  ), scope_bodies AS (
    SELECT latest_missing.*,
      (latest_missing.judgment_scope_doc - 'cache_key')
        || pg_catalog.jsonb_build_object(
          'scorer_image_digest', v_new_digest,
          'scorer_image_reference', v_new_reference
        ) AS scope_body
    FROM latest_missing
  ), keyed AS (
    SELECT scope.*,
      'sha256:' || pg_catalog.encode(
        extensions.digest(
          pg_catalog.convert_to(canonical.canonical_json, 'UTF8'), 'sha256'
        ), 'hex'
      ) AS new_cache_key
    FROM scope_bodies AS scope
    CROSS JOIN LATERAL (
      SELECT '{' || pg_catalog.string_agg(
        pg_catalog.to_jsonb(field.key)::TEXT || ':' || field.value::TEXT,
        ',' ORDER BY field.key COLLATE "C"
      ) || '}' AS canonical_json
      FROM pg_catalog.jsonb_each(scope.scope_body) AS field
    ) AS canonical
  )
  INSERT INTO public.lab_arena_runs (
    run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
    icp_position, attempt, status, lease_generation, stage_generation, kind,
    scored_run_id, previous_runner_hotkey, judgment_cache_key,
    judgment_input_hash, judgment_scope_doc, judgment_group_leader,
    judgment_group_miner_hotkeys, judgment_cache_source_run_id,
    company_judgment_refs
  )
  SELECT new_assignment || ':1', new_assignment, round_id, submission_id,
    miner_hotkey, stage, icp_position, 1, 'pending', 0, 10, kind,
    scored_run_id, NULL, new_cache_key, judgment_input_hash,
    scope_body || pg_catalog.jsonb_build_object('cache_key', new_cache_key),
    FALSE, judgment_group_miner_hotkeys, NULL, company_judgment_refs
  FROM keyed
  CROSS JOIN LATERAL (
    SELECT v_round_id || ':' || keyed.submission_id || ':1:'
      || keyed.icp_position::TEXT || ':score' || v_recovery_suffix
      AS new_assignment
  ) AS recovered;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 3 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 insert count differs';
  END IF;

  WITH groups AS (
    SELECT judgment_cache_key, pg_catalog.min(run_id) AS leader_run_id,
      pg_catalog.array_agg(DISTINCT miner_hotkey ORDER BY miner_hotkey)
        AS miner_hotkeys
    FROM public.lab_arena_runs
    WHERE round_id = v_round_id
      AND assignment_id LIKE '%' || v_recovery_suffix
    GROUP BY judgment_cache_key
  )
  UPDATE public.lab_arena_runs AS run
  SET judgment_group_leader = run.run_id = groups.leader_run_id,
      judgment_group_miner_hotkeys = groups.miner_hotkeys
  FROM groups
  WHERE run.round_id = v_round_id
    AND run.assignment_id LIKE '%' || v_recovery_suffix
    AND run.judgment_cache_key = groups.judgment_cache_key;

  v_stage1_scoring_close := pg_catalog.transaction_timestamp()
    + INTERVAL '6 hours 30 minutes';
  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1_scoring',
      status_generation = 11,
      stage_generation = 10,
      cancel_reason = NULL,
      configuration_doc = pg_catalog.jsonb_set(
        pg_catalog.jsonb_set(
          pg_catalog.jsonb_set(
            configuration_doc,
            '{schedule}',
            ((configuration_doc -> 'schedule')
              - 'stage_3_start' - 'stage_3_close' - 'stage_3_scoring_close')
              || pg_catalog.jsonb_build_object(
              'stage_1_scoring_close', pg_catalog.to_char(
                v_stage1_scoring_close AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
              ),
              'stage_2_start', pg_catalog.to_char(
                (v_stage1_scoring_close + INTERVAL '1 second') AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
              ),
              'stage_2_close', pg_catalog.to_char(
                (v_stage1_scoring_close + INTERVAL '3 hours 1 second')
                  AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
              ),
              'final_scoring_close', pg_catalog.to_char(
                (v_stage1_scoring_close + INTERVAL '9 hours 30 minutes 1 second')
                  AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
              ),
              'publication_deadline', pg_catalog.to_char(
                (v_stage1_scoring_close + INTERVAL '9 hours 30 minutes 2 seconds')
                  AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
              )
            ), FALSE
          ),
          '{scorer_image_digest}', pg_catalog.to_jsonb(v_new_digest), FALSE
        ),
        '{scorer_image_reference}', pg_catalog.to_jsonb(v_new_reference), FALSE
      )
  WHERE round_id = v_round_id
    AND status = 'cancelled'
    AND cancel_reason = 'scoring_incomplete'
    AND status_generation = 10
    AND stage_generation = 9;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 transition failed';
  END IF;

  IF (SELECT pg_catalog.array_agg(assignment_id ORDER BY assignment_id)
      FROM public.lab_arena_runs
      WHERE round_id = v_round_id
        AND assignment_id LIKE '%' || v_recovery_suffix
        AND attempt = 1 AND status = 'pending'
        AND run_id = assignment_id || ':1'
        AND stage_generation = 10) IS DISTINCT FROM v_target_assignments
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_runs AS recovered
       JOIN LATERAL (
         SELECT prior.*
         FROM public.lab_arena_runs AS prior
         WHERE prior.round_id = v_round_id
           AND prior.submission_id = recovered.submission_id
           AND prior.icp_position = recovered.icp_position
           AND prior.scored_run_id = recovered.scored_run_id
           AND prior.assignment_id LIKE '%' || v_prior_suffix
         ORDER BY prior.stage_generation DESC, prior.attempt DESC,
           prior.created_at DESC, prior.run_id DESC
         LIMIT 1
       ) AS prior ON TRUE
       WHERE recovered.round_id = v_round_id
         AND recovered.assignment_id LIKE '%' || v_recovery_suffix
         AND (
           recovered.judgment_input_hash IS DISTINCT FROM prior.judgment_input_hash
           OR recovered.company_judgment_refs IS DISTINCT FROM prior.company_judgment_refs
           OR recovered.judgment_cache_key = prior.judgment_cache_key
           OR recovered.judgment_scope_doc ->> 'scorer_image_digest'
              IS DISTINCT FROM v_new_digest
           OR recovered.judgment_scope_doc ->> 'scorer_image_reference'
              IS DISTINCT FROM v_new_reference
           OR recovered.judgment_scope_doc ->> 'cache_key'
              IS DISTINCT FROM recovered.judgment_cache_key
           OR recovered.judgment_cache_source_run_id IS NOT NULL
           OR recovered.judgment_scope_doc
                - 'cache_key' - 'scorer_image_digest' - 'scorer_image_reference'
              IS DISTINCT FROM prior.judgment_scope_doc
                - 'cache_key' - 'scorer_image_digest' - 'scorer_image_reference'
         )
     )
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id
           AND assignment_id NOT LIKE '%' || v_recovery_suffix)
          IS DISTINCT FROM v_runs_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id AND status = 'accepted')
          IS DISTINCT FROM v_accepted_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(ledger)::TEXT), '|' ORDER BY entry_id
         ), '')) FROM public.lab_arena_ledger AS ledger
         WHERE round_id = v_round_id) IS DISTINCT FROM v_ledger_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(submission)::TEXT),
           '|' ORDER BY submission_id
         )) FROM public.lab_arena_submissions AS submission
         WHERE round_id = v_round_id) IS DISTINCT FROM v_submission_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(credential)::TEXT),
           '|' ORDER BY submission_id, provider
         )) FROM public.lab_arena_submission_credentials AS credential
         WHERE submission_id = ANY(v_participant_ids))
          IS DISTINCT FROM v_credential_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(cache_row)::TEXT),
           '|' ORDER BY cache_key
         ), '')) FROM public.lab_arena_judgment_cache AS cache_row
         WHERE scope_doc ->> 'round_id' = v_round_id)
          IS DISTINCT FROM v_cache_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(reservation)::TEXT),
           '|' ORDER BY reservation.cache_key, reservation.authority_slot
         ), ''))
         FROM public.lab_arena_company_judgment_reservations AS reservation
         JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
         WHERE run.round_id = v_round_id) IS DISTINCT FROM v_reservation_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(other_round)::TEXT),
           '|' ORDER BY round_id
         ), '')) FROM public.lab_arena_rounds AS other_round
         WHERE round_id <> v_round_id) IS DISTINCT FROM v_unrelated_hash
     OR (SELECT pg_catalog.to_jsonb(current_round)
           - 'status' - 'status_generation' - 'stage_generation'
           - 'cancel_reason' - 'configuration_doc' - 'updated_at'
         FROM public.lab_arena_rounds AS current_round
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR (SELECT configuration_doc
           - 'schedule' - 'scorer_image_digest' - 'scorer_image_reference'
         FROM public.lab_arena_rounds
         WHERE round_id = v_round_id) IS DISTINCT FROM v_configuration_stable
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_round_id
         AND status = 'stage1_scoring'
         AND status_generation = 11
         AND stage_generation = 10
         AND cancel_reason IS NULL
         AND configuration_doc ->> 'scorer_image_digest' = v_new_digest
         AND configuration_doc ->> 'scorer_image_reference' = v_new_reference
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery252 verification failed';
  END IF;
END;
$lab_arena_252_recover_20260914_current_scorer$;

COMMIT;
