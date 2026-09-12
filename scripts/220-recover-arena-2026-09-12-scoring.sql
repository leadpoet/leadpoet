-- Recover the cancelled scoring stage of arena-2026-09-12 after the gateway
-- response-size fix. Two Deepline calls lost their provider cost when the
-- gateway replaced valid oversized responses with synthetic 502 responses.
-- Authenticated provider history bounds those two unresolved calls to 4,000
-- microusd in aggregate. Keep each call uncertain at that full upper bound,
-- then requeue only the 45 incomplete scoring assignments under a fresh
-- assignment namespace so their old call identities cannot replay.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;

DO $lab_arena_220_recover_20260912_scoring$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-12';
  v_target_submission CONSTANT TEXT :=
    'sub-ddd33c5c142f317e0279265f179b5bc1';
  v_suffix CONSTANT TEXT := ':recovery220';
  v_participant_ids CONSTANT TEXT[] := ARRAY[
    'baseline-2026-09-12',
    'sub-1460c4e33f19f923b969ef2fd941072f',
    'sub-64750217950f089a3bec21ee6fe72d6e',
    'sub-67fdb5d43bc8f9ea4cb6e74df4797037',
    'sub-c2ab2d55f187807c2793433701485825',
    'sub-d39d12f74f5f6fb56837d22890c7dd94',
    'sub-ddd33c5c142f317e0279265f179b5bc1',
    'sub-eb8a764bc501efb7d47f9c6c74d49756'
  ]::TEXT[];
  v_finalists CONSTANT JSONB := '[
    "sub-c2ab2d55f187807c2793433701485825",
    "sub-ddd33c5c142f317e0279265f179b5bc1",
    "sub-67fdb5d43bc8f9ea4cb6e74df4797037",
    "sub-1460c4e33f19f923b969ef2fd941072f",
    "sub-eb8a764bc501efb7d47f9c6c74d49756",
    "sub-64750217950f089a3bec21ee6fe72d6e"
  ]'::JSONB;
  v_uncertain_ids CONSTANT BIGINT[] := ARRAY[301934, 301941]::BIGINT[];
  v_uncertain_calls CONSTANT TEXT[] := ARRAY[
    'sha256:6acff63503cff507fd39853485d8bf58a8b8bca6c50ae0a64be1afa774149b74',
    'sha256:ba4e76b407cf02a5786b35a5ef03c040f3810b60f1c81070c73c6f654a213a92'
  ]::TEXT[];
  v_group_request_ids CONSTANT JSONB := '[
    "iad1::5rqc9-1789181073615-e3533caafcc0",
    "iad1::2mrvb-1789181066118-d439739c5df7",
    "iad1::nrll6-1789181035861-44985ee12c9a"
  ]'::JSONB;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_live_participants TEXT[];
  v_incomplete_assignments TEXT[];
  v_latest_run_ids TEXT[];
  v_recovery_summary JSONB;
  v_count INTEGER;
  v_accepted_hash TEXT;
  v_submission_hash TEXT;
  v_credential_hash TEXT;
  v_other_ledger_hash TEXT;
  v_round_stable JSONB;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_close_scoring(text,smallint)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 220';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
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
    RAISE EXCEPTION 'arena 2026-09-12 recovery guard state differs';
  END IF;

  IF pg_catalog.jsonb_typeof(v_round.participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 8
     OR v_round.finalists IS DISTINCT FROM v_finalists
     OR v_round.configuration_doc ->> 'integrity_policy' IS NOT NULL
     OR v_round.configuration_doc ->> 'max_attempts_per_assignment'
        IS DISTINCT FROM '2'
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM
        'sha256:1e50937e35c4fe7f099a241c20f0f12f3cb70453d35669d990a968d0734b5631'
     OR pg_catalog.jsonb_typeof(v_round.stage2_scoring_plan_doc)
        IS DISTINCT FROM 'object'
     OR pg_catalog.md5(v_round.stage2_scoring_plan_doc::TEXT)
        IS DISTINCT FROM
        'b2c9fb27a2045cd02c45faa528304037'
     OR pg_catalog.jsonb_typeof(
          v_round.stage2_scoring_plan_doc -> 'work_items'
        ) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(
          v_round.stage2_scoring_plan_doc -> 'work_items'
        ) <> 68
     OR pg_catalog.jsonb_typeof(
          v_round.stage2_scoring_plan_doc -> 'zero_rows'
        ) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(
          v_round.stage2_scoring_plan_doc -> 'zero_rows'
        ) <> 12 THEN
    RAISE EXCEPTION 'arena 2026-09-12 frozen round state differs';
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
     ) IS DISTINCT FROM v_participant_ids THEN
    RAISE EXCEPTION 'arena 2026-09-12 participant identities differ';
  END IF;

  -- A complete prior application is a no-op even after normal scoring and
  -- publication create or update later rows.
  SELECT pg_catalog.count(*) INTO v_count
  FROM public.lab_arena_runs
  WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
    AND assignment_id LIKE '%' || v_suffix;
  IF v_count > 0 THEN
    IF (SELECT pg_catalog.count(DISTINCT assignment_id)
        FROM public.lab_arena_runs
        WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
          AND assignment_id LIKE '%' || v_suffix) <> 45
       OR (SELECT pg_catalog.count(*)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
             AND assignment_id LIKE '%' || v_suffix
             AND run_id NOT LIKE '%' || v_suffix || ':%') < 47
       OR (SELECT pg_catalog.count(*)
           FROM public.lab_arena_ledger
           WHERE entry_id = ANY(v_uncertain_ids)
             AND entry_kind = 'uncertain'
             AND amount_microusd = 4000
             AND entry_doc #>> '{recovery220,schema_version}' =
               'leadpoet.lab_arena.cost_upper_bound_recovery.v1') <> 2
       OR v_round.status = 'cancelled'
       OR v_round.status NOT IN (
         'stage2_scoring', 'stage2_judged', 'scored', 'published'
       ) THEN
      RAISE EXCEPTION 'arena 2026-09-12 recovery replay state differs';
    END IF;
    RETURN;
  END IF;

  IF v_round.status <> 'cancelled'
     OR v_round.cancel_reason <> 'scoring_incomplete'
     OR v_round.status_generation <> 11
     OR v_round.stage_generation <> 9
     OR v_round.publication_doc IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1 FROM (
         SELECT DISTINCT ON (call_identity) entry_kind
         FROM public.lab_arena_ledger
         WHERE round_id = v_round_id AND call_identity IS NOT NULL
         ORDER BY call_identity, entry_id DESC
       ) AS heads
       WHERE entry_kind IN ('reservation', 'dispatch')
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_company_judgment_reservations AS r
       JOIN public.lab_arena_runs AS run ON run.run_id = r.run_id
       WHERE run.round_id = v_round_id
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 scoring recovery preflight differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
        AND status = 'accepted' AND terminal_cause = 'accepted') <> 73
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 70
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'execute') <> 83
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'execute') <> 80
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'execute'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 68
     OR (SELECT ROW(
           pg_catalog.count(*) FILTER (WHERE terminal_cause = 'credential_error'),
           pg_catalog.count(*) FILTER (WHERE terminal_cause = 'model_error'),
           pg_catalog.count(*) FILTER (WHERE terminal_cause = 'provider_error'),
           pg_catalog.count(*) FILTER (WHERE terminal_cause = 'model_timeout')
         ) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'execute'
           AND status = 'failed') IS DISTINCT FROM ROW(10::BIGINT, 3::BIGINT, 1::BIGINT, 1::BIGINT)
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score') <> 70
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score') <> 68
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 23
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'judge_error') <> 3
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'stage_closed') <> 44 THEN
    RAISE EXCEPTION 'arena 2026-09-12 run state differs';
  END IF;

  WITH latest AS (
    SELECT DISTINCT ON (assignment_id)
      assignment_id, run_id, submission_id, icp_position, attempt,
      status, terminal_cause, result_doc, terminal_doc,
      stage_generation, lease_generation
    FROM public.lab_arena_runs
    WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
    ORDER BY assignment_id, (status = 'accepted') DESC, attempt DESC
  )
  SELECT pg_catalog.array_agg(assignment_id ORDER BY assignment_id),
         pg_catalog.array_agg(run_id ORDER BY assignment_id),
         pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
           'run_id', run_id,
           'original_assignment_id', assignment_id,
           'prior_status', status,
           'prior_terminal_cause', terminal_cause,
           'prior_result_hash', pg_catalog.md5(COALESCE(result_doc::TEXT, '')),
           'prior_terminal_hash', pg_catalog.md5(COALESCE(terminal_doc::TEXT, '')),
           'prior_stage_generation', stage_generation,
           'prior_lease_generation', lease_generation
         ) ORDER BY assignment_id)
  INTO v_incomplete_assignments, v_latest_run_ids
       , v_recovery_summary
  FROM latest
  WHERE status <> 'accepted';
  IF pg_catalog.array_length(v_incomplete_assignments, 1) <> 45
     OR EXISTS (
       WITH latest AS (
         SELECT DISTINCT ON (assignment_id)
           assignment_id, run_id, submission_id, icp_position, attempt,
           status, terminal_cause
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
         ORDER BY assignment_id, (status = 'accepted') DESC, attempt DESC
       )
       SELECT 1 FROM latest
       WHERE status <> 'accepted'
         AND NOT (
           terminal_cause = 'stage_closed'
           OR (submission_id = v_target_submission
               AND icp_position = 12 AND attempt = 2
               AND terminal_cause = 'judge_error')
         )
     )
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND submission_id = v_target_submission
           AND ((icp_position = 12 AND attempt IN (1, 2)
                 AND terminal_cause = 'judge_error')
             OR (icp_position = 13 AND attempt = 1
                 AND terminal_cause = 'judge_error')
             OR (icp_position = 13 AND attempt = 2
                 AND terminal_cause = 'stage_closed')
             OR (icp_position BETWEEN 14 AND 19 AND attempt = 1
                 AND terminal_cause = 'stage_closed'))) <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND submission_id = v_target_submission
           AND icp_position IN (10, 11) AND attempt = 1
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 2 THEN
    RAISE EXCEPTION 'arena 2026-09-12 incomplete scoring set differs';
  END IF;

  IF NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE entry_id = 301934
         AND call_identity = v_uncertain_calls[1]
        AND run_id = v_round_id || ':' || v_target_submission || ':2:12:score:1'
        AND entry_kind = 'uncertain' AND provider = 'deepline'
        AND operation_id = 'scrapingdog.scrape'
        AND funding_source = 'miner_key'
        AND amount_microusd = 49329183
        AND entry_doc ->> 'reason' = 'worker_reported'
        AND entry_doc #>> '{call,reason}' = 'missing_provider_cost'
        AND entry_doc #>> '{call,provider_status}' = '502'
        AND entry_doc #>> '{call,response_provenance}' = 'response_too_large'
     )
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE entry_id = 301941
         AND call_identity = v_uncertain_calls[2]
         AND run_id = v_round_id || ':' || v_target_submission || ':2:12:score:1'
         AND entry_kind = 'uncertain' AND provider = 'deepline'
         AND operation_id = 'scrapingdog.scrape'
         AND funding_source = 'miner_key'
         AND amount_microusd = 4217
         AND entry_doc ->> 'reason' = 'worker_reported'
         AND entry_doc #>> '{call,reason}' = 'missing_provider_cost'
         AND entry_doc #>> '{call,provider_status}' = '502'
         AND entry_doc #>> '{call,response_provenance}' = 'response_too_large'
     )
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE entry_id = 301884
         AND call_identity =
           'sha256:21d55103563b01712ff2033ddcc10c55e23e8712e99c485e94b912760a66d880'
         AND entry_kind = 'settlement' AND provider = 'deepline'
         AND operation_id = 'scrapingdog.scrape'
         AND funding_source = 'miner_key' AND amount_microusd = 2000
         AND round_id = v_round_id
         AND submission_id = v_target_submission
         AND run_id = v_round_id || ':' || v_target_submission || ':2:11:score:1'
         AND terminal_response #>> '{provider_cost,request_id}' =
           'iad1::nrll6-1789181035861-44985ee12c9a'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 provider billing evidence differs';
  END IF;

  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         )) INTO v_accepted_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id AND status = 'accepted';
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
         ), '')) INTO v_other_ledger_hash
  FROM public.lab_arena_ledger AS ledger
  WHERE round_id = v_round_id AND entry_id <> ALL(v_uncertain_ids);
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'status' - 'status_generation' - 'stage_generation'
    - 'cancel_reason' - 'updated_at';

  ALTER TABLE public.lab_arena_ledger
    DISABLE TRIGGER lab_arena_ledger_append_only;
  UPDATE public.lab_arena_ledger
  SET amount_microusd = 4000,
      entry_doc = entry_doc || pg_catalog.jsonb_build_object(
        'recovery220', pg_catalog.jsonb_build_object(
          'schema_version',
            'leadpoet.lab_arena.cost_upper_bound_recovery.v1',
          'original_amount_microusd', amount_microusd,
          'retained_uncertain_amount_microusd', 4000,
          'provider_proof_sha256',
            '3e39de630ae4f1882489733834c8be50f2b71a1e6c72fa73f2e1fd455629b23f',
          'provider_group_status', 'completed',
          'provider_charge_state', 'posted',
          'provider_group_total_microusd', 6000,
          'known_group_settlement_microusd', 2000,
          'unresolved_group_upper_bound_microusd', 4000,
          'known_settlement_entry_id', 301884,
          'group_request_ids', v_group_request_ids,
          'matched_request_id', CASE entry_id
            WHEN 301934 THEN 'iad1::2mrvb-1789181066118-d439739c5df7'
            WHEN 301941 THEN 'iad1::5rqc9-1789181073615-e3533caafcc0'
          END,
          'match_basis', 'same_account_operation_and_dispatch_window',
          'requeued_runs', CASE WHEN entry_id = 301934
            THEN v_recovery_summary ELSE NULL END
        )
      )
  WHERE entry_id = ANY(v_uncertain_ids);
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_ledger
    ENABLE TRIGGER lab_arena_ledger_append_only;
  IF v_count <> 2 THEN
    RAISE EXCEPTION 'arena 2026-09-12 uncertainty update count differs';
  END IF;

  ALTER TABLE public.lab_arena_runs
    DISABLE TRIGGER lab_arena_runs_terminal;
  UPDATE public.lab_arena_runs
  SET assignment_id = assignment_id || v_suffix
  WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
    AND assignment_id = ANY(v_incomplete_assignments);
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 47 THEN
    RAISE EXCEPTION 'arena 2026-09-12 recovery lineage count differs';
  END IF;
  UPDATE public.lab_arena_runs
  SET status = 'pending',
      runner_hotkey = NULL,
      lease_token_hash = NULL,
      lease_generation = lease_generation + 1,
      stage_generation = 10,
      lease_expires_at = NULL,
      claim_request_id = NULL,
      claim_request_hash = NULL,
      claim_response = NULL,
      result_doc = NULL,
      output_ref = NULL,
      terminal_cause = NULL,
      terminal_doc = NULL,
      per_icp_score = NULL
  WHERE run_id = ANY(v_latest_run_ids)
    AND assignment_id LIKE '%' || v_suffix;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_runs
    ENABLE TRIGGER lab_arena_runs_terminal;
  IF v_count <> 45 THEN
    RAISE EXCEPTION 'arena 2026-09-12 requeue count differs';
  END IF;

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage2_scoring',
      status_generation = 12,
      stage_generation = 10,
      cancel_reason = NULL
  WHERE round_id = v_round_id AND status = 'cancelled'
    AND cancel_reason = 'scoring_incomplete'
    AND status_generation = 11 AND stage_generation = 9;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-12 round restart count differs';
  END IF;

  IF (SELECT pg_catalog.count(DISTINCT assignment_id)
      FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND stage = 2 AND kind = 'score') <> 68
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND assignment_id LIKE '%' || v_suffix
           AND run_id NOT LIKE '%' || v_suffix || ':%') <> 47
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'pending' AND stage_generation = 10
           AND assignment_id LIKE '%' || v_suffix) <> 45
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'accepted') <> 23
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id AND status = 'accepted')
        IS DISTINCT FROM v_accepted_hash
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
         WHERE round_id = v_round_id AND entry_id <> ALL(v_uncertain_ids))
        IS DISTINCT FROM v_other_ledger_hash
     OR (SELECT pg_catalog.to_jsonb(round_row)
           - 'status' - 'status_generation' - 'stage_generation'
           - 'cancel_reason' - 'updated_at'
         FROM public.lab_arena_rounds AS round_row
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_round_id AND status = 'stage2_scoring'
         AND status_generation = 12 AND stage_generation = 10
         AND cancel_reason IS NULL
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
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_ledger'::pg_catalog.regclass
         AND tgname = 'lab_arena_ledger_append_only' AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 scoring recovery postcondition differs';
  END IF;
END;
$lab_arena_220_recover_20260912_scoring$;

COMMIT;
