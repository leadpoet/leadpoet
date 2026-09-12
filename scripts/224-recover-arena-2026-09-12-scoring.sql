-- Recover the third cancelled scoring stage of arena-2026-09-12. One
-- Deepline call was cancelled after provider dispatch. Authenticated billing
-- history proves that its seven-call group totals 14,000 microusd and six
-- stored settlements total 12,000 microusd. Keep that call uncertain at the
-- remaining 2,000-microusd upper bound, then requeue only the 35 incomplete
-- scoring assignments under a fresh namespace.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;

DO $lab_arena_224_recover_20260912_scoring$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-12';
  v_bounded_submission CONSTANT TEXT :=
    'sub-1460c4e33f19f923b969ef2fd941072f';
  v_suffix CONSTANT TEXT := ':recovery224';
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
  v_uncertain_id CONSTANT BIGINT := 301984;
  v_uncertain_call CONSTANT TEXT :=
    'sha256:d1c0ba74a77451bf0a65c950df16186cd537fd146389c0291adbe543699fe742';
  v_group_request_ids CONSTANT JSONB := '[
    "iad1::z5xhz-1789181126691-b7553928b1f2",
    "iad1::ngv9f-1789181111082-cc2be8cfc3eb",
    "iad1::7bfxj-1789181054267-d3bcb4315d95",
    "iad1::9dnkl-1789181047530-d15b4d25be2e",
    "iad1::8kb7l-1789181028173-1d3c04d66e18",
    "iad1::f65sc-1789181027940-6ca64c78ab6d",
    "iad1::rxx97-1789180998144-d8d38ab2f371"
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
     ) IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_index
       WHERE indexrelid =
         'public.lab_arena_ledger_settlement_uq'::pg_catalog.regclass
         AND indisunique AND indisvalid AND indisready
     ) THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 224';
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
     OR v_round.configuration_doc ->> 'scoring_cap_microusd'
        IS DISTINCT FROM '50000000'
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
          AND assignment_id LIKE '%' || v_suffix) <> 35
       OR (SELECT pg_catalog.count(*)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
             AND assignment_id LIKE '%' || v_suffix
             AND run_id NOT LIKE '%' || v_suffix || ':%') <> 37
       OR (SELECT pg_catalog.count(*)
           FROM public.lab_arena_ledger
           WHERE entry_id = v_uncertain_id
             AND entry_kind = 'uncertain'
             AND amount_microusd = 2000
             AND entry_doc #>> '{recovery224,schema_version}' =
               'leadpoet.lab_arena.cost_upper_bound_recovery.v1') <> 1
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
     OR v_round.status_generation <> 13
     OR v_round.stage_generation <> 11
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
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score') <> 72
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score') <> 68
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 33
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'judge_error') <> 5
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'stage_closed') <> 34
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND status = 'accepted') <> 244 THEN
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
  IF pg_catalog.array_length(v_incomplete_assignments, 1) <> 35
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
           OR (submission_id = v_bounded_submission
               AND icp_position = 13 AND attempt = 2
               AND terminal_cause = 'judge_error')
         )
     )
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND assignment_id = v_round_id || ':' || v_bounded_submission
             || ':2:13:score:recovery220'
           AND attempt IN (1, 2) AND status = 'failed'
           AND terminal_cause = 'judge_error') <> 2
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND assignment_id = ANY(v_incomplete_assignments)) <> 37
     OR EXISTS (
       SELECT 1 FROM pg_catalog.unnest(v_incomplete_assignments) AS item
       WHERE item NOT LIKE '%:recovery220'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 incomplete scoring set differs';
  END IF;

  IF NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE entry_id = v_uncertain_id
         AND call_identity = v_uncertain_call
         AND run_id = v_round_id || ':' || v_bounded_submission
           || ':2:13:score:1'
         AND entry_kind = 'uncertain' AND provider = 'deepline'
         AND operation_id = 'scrapingdog.scrape'
         AND funding_source = 'miner_key'
         AND amount_microusd = 49294310
         AND entry_doc = '{"reason":"round_cancelled"}'::JSONB
         AND terminal_response IS NULL
     )
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_ledger AS ledger
         JOIN (VALUES
           (301857::BIGINT, 'iad1::rxx97-1789180998144-d8d38ab2f371'),
           (301865::BIGINT, 'iad1::f65sc-1789181027940-6ca64c78ab6d'),
           (301868::BIGINT, 'iad1::8kb7l-1789181028173-1d3c04d66e18'),
           (301908::BIGINT, 'iad1::9dnkl-1789181047530-d15b4d25be2e'),
           (301914::BIGINT, 'iad1::7bfxj-1789181054267-d3bcb4315d95'),
           (301963::BIGINT, 'iad1::ngv9f-1789181111082-cc2be8cfc3eb')
         ) AS proof(entry_id, request_id)
           ON proof.entry_id = ledger.entry_id
          AND proof.request_id =
            ledger.terminal_response #>> '{provider_cost,request_id}'
         WHERE ledger.round_id = v_round_id
           AND ledger.entry_kind = 'settlement'
           AND ledger.provider = 'deepline'
           AND ledger.operation_id = 'scrapingdog.scrape'
           AND ledger.funding_source = 'miner_key'
           AND ledger.amount_microusd = 2000) <> 6
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE entry_id IN (301934, 301941)
           AND entry_kind = 'uncertain' AND amount_microusd = 4000
           AND entry_doc #>> '{recovery220,schema_version}' =
             'leadpoet.lab_arena.cost_upper_bound_recovery.v1'
           AND terminal_response IS NULL) <> 2
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE entry_id = 302118
         AND call_identity =
           'sha256:6a2453c7617a47c4454838996928101fbd6e91ae0f5fde25ba2a304cb63564be'
         AND entry_kind = 'uncertain' AND provider = 'openrouter'
         AND operation_id = 'openrouter.chat'
         AND funding_source = 'miner_key' AND amount_microusd = 20254
         AND entry_doc = '{"reason":"round_cancelled"}'::JSONB
         AND terminal_response IS NULL
     )
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE entry_id = 302119
         AND call_identity =
           'sha256:a2d13165186dadb801e4f57486468d12af682bd372a66cb742870b7c709eb916'
         AND entry_kind = 'uncertain' AND provider = 'openrouter'
         AND operation_id = 'openrouter.chat'
         AND funding_source = 'miner_key' AND amount_microusd = 139629
         AND entry_doc = '{"reason":"round_cancelled"}'::JSONB
         AND terminal_response IS NULL
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 provider billing evidence differs';
  END IF;

  -- These are all scoring uncertainties. Entry 289258 is the proven d39
  -- account failure and has no stage-2 score assignment. Every participant
  -- with incomplete work retains room for the largest OpenRouter reservation
  -- observed in this frozen round after entry 301984 is bounded.
  IF (SELECT pg_catalog.count(*)
      FROM (
        SELECT DISTINCT ON (ledger.call_identity)
          ledger.entry_id, ledger.entry_kind
        FROM public.lab_arena_ledger AS ledger
        JOIN public.lab_arena_runs AS run ON run.run_id = ledger.run_id
        WHERE ledger.round_id = v_round_id AND run.kind = 'score'
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      ) AS head
      WHERE head.entry_kind = 'uncertain'
        AND head.entry_id IN (289258, 301934, 301941, 301984, 302118, 302119)
     ) <> 6
     OR EXISTS (
       SELECT 1
       FROM (
         SELECT DISTINCT ON (ledger.call_identity)
           ledger.entry_id, ledger.entry_kind
         FROM public.lab_arena_ledger AS ledger
         JOIN public.lab_arena_runs AS run ON run.run_id = ledger.run_id
         WHERE ledger.round_id = v_round_id AND run.kind = 'score'
           AND ledger.call_identity IS NOT NULL
         ORDER BY ledger.call_identity, ledger.entry_id DESC
       ) AS head
       WHERE head.entry_kind = 'uncertain'
         AND head.entry_id NOT IN (289258, 301934, 301941, 301984, 302118, 302119)
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
         AND submission_id = 'sub-d39d12f74f5f6fb56837d22890c7dd94'
     )
     OR (SELECT pg_catalog.max(amount_microusd)
         FROM public.lab_arena_ledger AS ledger
         JOIN public.lab_arena_runs AS run ON run.run_id = ledger.run_id
         WHERE ledger.round_id = v_round_id AND run.kind = 'score'
           AND ledger.entry_kind = 'reservation'
           AND ledger.provider = 'openrouter') <> 156957 THEN
    RAISE EXCEPTION 'arena 2026-09-12 scoring cost guard differs';
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
  WHERE round_id = v_round_id AND entry_id <> v_uncertain_id;
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'status' - 'status_generation' - 'stage_generation'
    - 'cancel_reason' - 'updated_at';

  ALTER TABLE public.lab_arena_ledger
    DISABLE TRIGGER lab_arena_ledger_append_only;
  UPDATE public.lab_arena_ledger
  SET amount_microusd = 2000,
      entry_doc = entry_doc || pg_catalog.jsonb_build_object(
        'recovery224', pg_catalog.jsonb_build_object(
          'schema_version',
            'leadpoet.lab_arena.cost_upper_bound_recovery.v1',
          'original_amount_microusd', amount_microusd,
          'retained_uncertain_amount_microusd', 2000,
          'provider_proof_sha256',
            '52cc5f4cd6531cc00e379708b00ffd70153430d18ff1ba0936915cdb8bafe51b',
          'provider_group_status', 'completed',
          'provider_charge_state', 'posted',
          'provider_group_total_microusd', 14000,
          'known_group_settlement_microusd', 12000,
          'unresolved_group_upper_bound_microusd', 2000,
          'known_settlement_entry_ids',
            '[301857,301865,301868,301908,301914,301963]'::JSONB,
          'group_request_ids', v_group_request_ids,
          'matched_request_id',
            'iad1::z5xhz-1789181126691-b7553928b1f2',
          'match_basis',
            'same_account_operation_dispatch_window_and_group_remainder',
          'requeued_runs', v_recovery_summary
        )
      )
  WHERE entry_id = v_uncertain_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_ledger
    ENABLE TRIGGER lab_arena_ledger_append_only;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-12 uncertainty update count differs';
  END IF;

  IF EXISTS (
    SELECT locked.submission_id
    FROM (
      SELECT head.submission_id,
             pg_catalog.sum(head.amount_microusd)::BIGINT AS amount_microusd
      FROM (
        SELECT DISTINCT ON (ledger.call_identity)
          ledger.submission_id, ledger.entry_kind, ledger.amount_microusd
        FROM public.lab_arena_ledger AS ledger
        JOIN public.lab_arena_runs AS run ON run.run_id = ledger.run_id
        WHERE ledger.round_id = v_round_id AND run.kind = 'score'
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      ) AS head
      WHERE head.entry_kind IN ('settlement', 'uncertain')
        AND head.submission_id <>
          'sub-d39d12f74f5f6fb56837d22890c7dd94'
      GROUP BY head.submission_id
    ) AS locked
    WHERE locked.amount_microusd > 50000000 - 156957
  ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 scoring budget room differs';
  END IF;

  ALTER TABLE public.lab_arena_runs
    DISABLE TRIGGER lab_arena_runs_terminal;
  UPDATE public.lab_arena_runs
  SET assignment_id = assignment_id || v_suffix
  WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
    AND assignment_id = ANY(v_incomplete_assignments);
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 37 THEN
    RAISE EXCEPTION 'arena 2026-09-12 recovery lineage count differs';
  END IF;
  UPDATE public.lab_arena_runs
  SET status = 'pending',
      runner_hotkey = NULL,
      lease_token_hash = NULL,
      lease_generation = lease_generation + 1,
      stage_generation = 12,
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
  IF v_count <> 35 THEN
    RAISE EXCEPTION 'arena 2026-09-12 requeue count differs';
  END IF;

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage2_scoring',
      status_generation = 14,
      stage_generation = 12,
      cancel_reason = NULL
  WHERE round_id = v_round_id AND status = 'cancelled'
    AND cancel_reason = 'scoring_incomplete'
    AND status_generation = 13 AND stage_generation = 11;
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
           AND run_id NOT LIKE '%' || v_suffix || ':%') <> 37
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'pending' AND stage_generation = 12
           AND assignment_id LIKE '%' || v_suffix) <> 35
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 2 AND kind = 'score'
           AND status = 'accepted') <> 33
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
         WHERE round_id = v_round_id AND entry_id <> v_uncertain_id)
        IS DISTINCT FROM v_other_ledger_hash
     OR (SELECT pg_catalog.to_jsonb(round_row)
           - 'status' - 'status_generation' - 'stage_generation'
           - 'cancel_reason' - 'updated_at'
         FROM public.lab_arena_rounds AS round_row
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_round_id AND status = 'stage2_scoring'
         AND status_generation = 14 AND stage_generation = 12
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
$lab_arena_224_recover_20260912_scoring$;

COMMIT;
