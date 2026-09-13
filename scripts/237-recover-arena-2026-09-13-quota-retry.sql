-- Add the unused normal retry for one September 13 score whose first attempt
-- was mislabeled credential_error after its retained provider-call history hit
-- the per-ICP Deepline quota. Preserve that failed attempt and every cost row.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_237_recover_quota_retry$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-13';
  v_submission_id CONSTANT TEXT :=
    'sub-5c1f20eb379bcd4e11c29912b5251b0d';
  v_miner_hotkey CONSTANT TEXT :=
    '5H956XR9zjVxgzgPAbamfNrWboXTMp9iLDu7LKrBW4VYKF8r';
  v_assignment_id CONSTANT TEXT :=
    'arena-2026-09-13:sub-5c1f20eb379bcd4e11c29912b5251b0d:1:0:score:recovery234:recovery235';
  v_first_run_id CONSTANT TEXT :=
    'arena-2026-09-13:sub-5c1f20eb379bcd4e11c29912b5251b0d:1:0:score:1';
  v_retry_run_id CONSTANT TEXT :=
    'arena-2026-09-13:sub-5c1f20eb379bcd4e11c29912b5251b0d:1:0:score:recovery234:recovery235:2';
  v_scored_run_id CONSTANT TEXT :=
    'arena-2026-09-13:sub-5c1f20eb379bcd4e11c29912b5251b0d:1:0:1';
  v_digest CONSTANT TEXT :=
    'sha256:67b3ed3c8691a321007da8cc8353a4bb89b517813c7e017fcb93f55e468dd536';
  v_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:67b3ed3c8691a321007da8cc8353a4bb89b517813c7e017fcb93f55e468dd536';
  v_round public.lab_arena_rounds%ROWTYPE;
  v_first public.lab_arena_runs%ROWTYPE;
  v_retry public.lab_arena_runs%ROWTYPE;
  v_first_hash TEXT;
  v_ledger_hash TEXT;
  v_count INTEGER;
  v_deepline_settled INTEGER;
  v_openrouter_settled INTEGER;
  v_quota_refusals INTEGER;
  v_open_calls INTEGER;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
     ) IS NULL
     OR pg_catalog.strpos(
       pg_catalog.pg_get_functiondef(
         'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
       ),
       'lab_arena_score_submission_serialization'
     ) = 0 THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 237';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR SHARE;
  IF NOT FOUND THEN
    RETURN;
  END IF;

  SELECT * INTO v_first
  FROM public.lab_arena_runs
  WHERE run_id = v_first_run_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'arena 2026-09-13 quota retry source is missing';
  END IF;

  SELECT * INTO v_retry
  FROM public.lab_arena_runs
  WHERE run_id = v_retry_run_id;
  IF FOUND THEN
    IF v_first.assignment_id IS DISTINCT FROM v_assignment_id
       OR v_first.status <> 'failed'
       OR v_first.terminal_cause <> 'credential_error'
       OR v_retry.assignment_id IS DISTINCT FROM v_assignment_id
       OR v_retry.round_id IS DISTINCT FROM v_round_id
       OR v_retry.submission_id IS DISTINCT FROM v_submission_id
       OR v_retry.miner_hotkey IS DISTINCT FROM v_miner_hotkey
       OR v_retry.stage <> 1 OR v_retry.icp_position <> 0
       OR v_retry.attempt <> 2 OR v_retry.kind <> 'score'
       OR v_retry.scored_run_id IS DISTINCT FROM v_scored_run_id
       OR v_retry.stage_generation <> 7
       OR v_retry.previous_runner_hotkey IS DISTINCT FROM
          v_first.runner_hotkey
       OR v_retry.judgment_cache_key IS DISTINCT FROM
          v_first.judgment_cache_key
       OR v_retry.judgment_input_hash IS DISTINCT FROM
          v_first.judgment_input_hash
       OR v_retry.judgment_scope_doc IS DISTINCT FROM
          v_first.judgment_scope_doc
       OR v_retry.judgment_group_leader IS DISTINCT FROM
          v_first.judgment_group_leader
       OR v_retry.judgment_group_miner_hotkeys IS DISTINCT FROM
          v_first.judgment_group_miner_hotkeys
       OR v_retry.company_judgment_refs IS DISTINCT FROM
          v_first.company_judgment_refs THEN
      RAISE EXCEPTION 'arena 2026-09-13 quota retry replay differs';
    END IF;
    RETURN;
  END IF;

  IF v_round.status <> 'stage1_scoring'
     OR v_round.status_generation <> 8
     OR v_round.stage_generation <> 7
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.evaluation_date <> '2026-09-13'
     OR v_round.icp_set_date <> DATE '2026-09-12'
     OR v_round.configuration_doc ->> 'max_attempts_per_assignment'
        IS DISTINCT FROM '2'
     OR v_round.configuration_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_digest
     OR v_round.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_reference
     OR v_first.assignment_id IS DISTINCT FROM v_assignment_id
     OR v_first.round_id IS DISTINCT FROM v_round_id
     OR v_first.submission_id IS DISTINCT FROM v_submission_id
     OR v_first.miner_hotkey IS DISTINCT FROM v_miner_hotkey
     OR v_first.stage <> 1 OR v_first.icp_position <> 0
     OR v_first.attempt <> 1 OR v_first.kind <> 'score'
     OR v_first.scored_run_id IS DISTINCT FROM v_scored_run_id
     OR v_first.status <> 'failed'
     OR v_first.terminal_cause <> 'credential_error'
     OR v_first.stage_generation <> 7
     OR v_first.lease_generation <> 5
     OR v_first.runner_hotkey IS NULL
     OR v_first.output_ref IS NOT NULL
     OR v_first.terminal_doc IS NOT NULL
     OR v_first.per_icp_score IS NOT NULL
     OR v_first.judgment_cache_source_run_id IS NOT NULL
     OR v_first.company_judgment_refs IS NOT NULL
     OR v_first.result_doc ->> 'schema_version'
        IS DISTINCT FROM 'leadpoet.lab_arena.run_result.v1'
     OR v_first.result_doc ->> 'terminal_status'
        IS DISTINCT FROM 'credential_error'
     OR (v_first.result_doc #>> '{resource_summary,provider_call_count}')::INTEGER
        IS DISTINCT FROM 36
     OR pg_catalog.jsonb_typeof(v_first.judgment_scope_doc)
        IS DISTINCT FROM 'object'
     OR ARRAY(
       SELECT field.key
       FROM pg_catalog.jsonb_each(
         CASE
           WHEN pg_catalog.jsonb_typeof(v_first.judgment_scope_doc) = 'object'
             THEN v_first.judgment_scope_doc
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
     OR v_first.judgment_scope_doc ->> 'cache_key'
        IS DISTINCT FROM v_first.judgment_cache_key
     OR v_first.judgment_scope_doc ->> 'scoring_input_hash'
        IS DISTINCT FROM v_first.judgment_input_hash
     OR v_first.judgment_scope_doc ->> 'round_id'
        IS DISTINCT FROM v_round_id
     OR v_first.judgment_scope_doc ->> 'evaluation_date'
        IS DISTINCT FROM '2026-09-13'
     OR v_first.judgment_scope_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR v_first.judgment_scope_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_digest
     OR v_first.judgment_scope_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_reference
     OR COALESCE(v_first.judgment_group_leader, FALSE) IS NOT TRUE
     OR v_first.judgment_group_miner_hotkeys
        IS DISTINCT FROM ARRAY[v_miner_hotkey]::TEXT[]
     OR pg_catalog.jsonb_typeof(
          v_first.claim_response -> 'runner_authority_exclusions'
        ) IS DISTINCT FROM 'array'
     OR NOT (
       v_first.claim_response -> 'runner_authority_exclusions'
       @> pg_catalog.jsonb_build_array(v_first.runner_hotkey)
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_judgment_cache
       WHERE cache_key = v_first.judgment_cache_key
     )
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS execution
       WHERE execution.run_id = v_scored_run_id
         AND execution.round_id = v_round_id
         AND execution.submission_id = v_submission_id
         AND execution.miner_hotkey = v_miner_hotkey
         AND execution.stage = 1 AND execution.icp_position = 0
         AND execution.kind = 'execute' AND execution.status = 'accepted'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 quota retry state differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE assignment_id = v_assignment_id) <> 1
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_ledger
         WHERE run_id = v_first_run_id AND entry_kind = 'refusal'
           AND provider = 'deepline' AND funding_source = 'miner_key'
           AND entry_doc ->> 'reason' = 'per_icp_quota'
           AND COALESCE(
             (entry_doc ->> 'prior_miner_credential_refusal')::BOOLEAN,
             FALSE
           ) IS FALSE
           AND operation_id IN ('exa.contents', 'scrapingdog.scrape')) <> 2 THEN
    RAISE EXCEPTION 'arena 2026-09-13 quota retry cause differs';
  END IF;

  WITH heads AS (
    SELECT DISTINCT ON (call_identity)
      entry_kind, provider, funding_source
    FROM public.lab_arena_ledger
    WHERE run_id = v_first_run_id AND call_identity IS NOT NULL
    ORDER BY call_identity, entry_id DESC
  )
  SELECT
    pg_catalog.count(*) FILTER (
      WHERE entry_kind = 'settlement' AND provider = 'deepline'
        AND funding_source = 'miner_key'
    ),
    pg_catalog.count(*) FILTER (
      WHERE entry_kind = 'settlement' AND provider = 'openrouter'
        AND funding_source = 'miner_key'
    ),
    pg_catalog.count(*) FILTER (
      WHERE entry_kind = 'refusal' AND provider = 'deepline'
        AND funding_source = 'miner_key'
    ),
    pg_catalog.count(*) FILTER (
      WHERE entry_kind IN ('reservation', 'dispatch')
    )
  INTO v_deepline_settled, v_openrouter_settled,
       v_quota_refusals, v_open_calls
  FROM heads;
  IF v_deepline_settled <> 40
     OR v_openrouter_settled <> 41
     OR v_quota_refusals <> 2
     OR v_open_calls <> 0 THEN
    RAISE EXCEPTION 'arena 2026-09-13 quota retry ledger differs';
  END IF;

  SELECT pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT)
  INTO v_first_hash
  FROM public.lab_arena_runs AS run
  WHERE run_id = v_first_run_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(ledger)::TEXT, '|' ORDER BY entry_id
         ), ''))
  INTO v_ledger_hash
  FROM public.lab_arena_ledger AS ledger
  WHERE run_id = v_first_run_id;

  INSERT INTO public.lab_arena_runs (
    run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
    icp_position, attempt, status, lease_generation, stage_generation,
    kind, scored_run_id, previous_runner_hotkey, judgment_cache_key,
    judgment_input_hash, judgment_scope_doc, judgment_group_leader,
    judgment_group_miner_hotkeys, company_judgment_refs
  ) VALUES (
    v_retry_run_id, v_assignment_id, v_round_id, v_submission_id,
    v_miner_hotkey, 1, 0, 2, 'pending', v_first.lease_generation,
    v_round.stage_generation, 'score', v_scored_run_id,
    v_first.runner_hotkey, v_first.judgment_cache_key,
    v_first.judgment_input_hash, v_first.judgment_scope_doc,
    v_first.judgment_group_leader, v_first.judgment_group_miner_hotkeys,
    v_first.company_judgment_refs
  );
  GET DIAGNOSTICS v_count = ROW_COUNT;

  IF v_count <> 1
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE assignment_id = v_assignment_id) <> 2
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE run_id = v_retry_run_id AND assignment_id = v_assignment_id
         AND attempt = 2 AND status = 'pending'
         AND stage_generation = 7 AND lease_generation = 5
         AND previous_runner_hotkey = v_first.runner_hotkey
         AND judgment_cache_key = v_first.judgment_cache_key
         AND judgment_input_hash = v_first.judgment_input_hash
         AND judgment_scope_doc = v_first.judgment_scope_doc
         AND judgment_group_leader IS TRUE
         AND judgment_group_miner_hotkeys = ARRAY[v_miner_hotkey]::TEXT[]
         AND company_judgment_refs IS NULL
     )
     OR (SELECT pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT)
         FROM public.lab_arena_runs AS run
         WHERE run_id = v_first_run_id) IS DISTINCT FROM v_first_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(ledger)::TEXT, '|' ORDER BY entry_id
         ), '')) FROM public.lab_arena_ledger AS ledger
         WHERE run_id = v_first_run_id) IS DISTINCT FROM v_ledger_hash THEN
    RAISE EXCEPTION 'arena 2026-09-13 quota retry verification failed';
  END IF;
END;
$lab_arena_237_recover_quota_retry$;

COMMIT;
