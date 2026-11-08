-- Prospective admission alternate: immutable first-primary marker; legacy Sol retained.
-- Eight-argument v2 explicitly opts in; existing seven-argument callers bind Sol.
-- Both entrypoints enforce saved authority. No existing rows or frozen policies change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE NOWAIT;

DO $begin669$
DECLARE
  v_signature TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_v2_signature TEXT := 'public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)';
  v_definition TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_schema_acl TEXT;
  v_has_create BOOLEAN;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity FROM pg_proc p WHERE p.oid=to_regprocedure(v_signature);
  IF v_identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"]]'::JSONB THEN
    RAISE EXCEPTION 'Arena alternate669 security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')='2ade4eb49ebeddead5cf0356f6f94e273a0a4d3d5de47de294c1576ca3c8cbb8' THEN
    IF encode(extensions.digest(pg_get_functiondef(to_regprocedure(v_v2_signature)),'sha256'),'hex') IS DISTINCT FROM 'e7c425663c97dc4a0bf615081c60c7d6e3ac2f18fc0d90456737d027b37ea8df'
       OR (SELECT jsonb_build_array(pg_get_userbyid(proowner),proacl::TEXT,prosecdef,provolatile,proconfig)
           FROM pg_proc WHERE oid=to_regprocedure(v_v2_signature)) IS DISTINCT FROM v_identity THEN
      RAISE EXCEPTION 'Arena alternate669 v2 readback differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM '03009634a9dcde9342a7428d73eaf4285bf341811241a1d51e5d54bc8b8b04b9'
     OR to_regprocedure(v_v2_signature) IS NOT NULL THEN
    RAISE EXCEPTION 'Arena alternate669 preimage differs' USING ERRCODE='55000';
  END IF;
  SELECT nspacl::TEXT,has_schema_privilege('lab_arena_owner',oid,'CREATE') INTO v_schema_acl,v_has_create
  FROM pg_namespace WHERE nspname='public';
  IF NOT v_has_create THEN GRANT CREATE ON SCHEMA public TO lab_arena_owner; END IF;
  EXECUTE $v2_669$CREATE OR REPLACE FUNCTION public.lab_arena_begin_submission_review_v2(p_submission_id text, p_miner_hotkey text, p_claim_token_hash text, p_reservation_microusd bigint, p_review_model text, p_file_count integer, p_source_bytes bigint, p_review_alternate_model text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE
  v_review_guard_present BOOLEAN;
  v_output_token_recommendation INTEGER := 32768;
  v_invalid_start INTEGER;
  v_invalid_count INTEGER;
  v_invalid_doc JSONB := '{}'::JSONB;
  v_episode_start INTEGER;
  v_primary_count INTEGER;
  v_alternate_count INTEGER;
  v_refusal_recovery BOOLEAN := FALSE;
  v_legacy_refusal BOOLEAN := FALSE;
  v_required_model TEXT;
  v_primary_model TEXT;
  v_alternate_model TEXT;
  v_first_primary public.lab_arena_ledger;
  v_nov03_recovery BOOLEAN;
  v_nov06_deadline TIMESTAMPTZ;
  v_round_id TEXT;
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_now TIMESTAMPTZ;
  v_freeze_at TIMESTAMPTZ;
  v_deadline TIMESTAMPTZ;
  v_retry_at TIMESTAMPTZ;
  v_expires TIMESTAMPTZ;
  v_attempt INTEGER;
  v_attempt_limit INTEGER;
  v_backoff_seconds INTEGER;
  v_retryable BOOLEAN;
  v_call_identity TEXT;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_review_cost BIGINT;
BEGIN
  IF COALESCE(p_review_alternate_model,'') NOT IN ('openai/gpt-6.1-sol','moonshotai/kimi-k3.1-pro')
     OR COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
     OR COALESCE(p_miner_hotkey, '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
     OR COALESCE(p_claim_token_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_reservation_microusd, -1) NOT BETWEEN 0 AND 1000000000
     OR COALESCE(p_review_model, '') !~ '^[A-Za-z0-9._:/-]{1,200}$'
     OR COALESCE(p_file_count, -1) NOT BETWEEN 0 AND 1000
     OR COALESCE(p_source_bytes, -1) NOT BETWEEN 0 AND 52428800 THEN
    RAISE EXCEPTION 'lab_arena_code_review_input_invalid'
      USING ERRCODE = '22023';
  END IF;

  SELECT submission.round_id INTO v_round_id
  FROM public.lab_arena_submissions AS submission
  WHERE submission.submission_id = p_submission_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR SHARE;
  SELECT * INTO v_submission
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id
  FOR UPDATE;
  IF NOT FOUND
     OR v_submission.miner_hotkey IS DISTINCT FROM p_miner_hotkey
     OR v_submission.status NOT IN ('accepted', 'frozen') THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  v_nov03_recovery := public.lab_arena__nov03_review605_v1(v_round,v_submission);
  -- Follow the existing round/submission lock order before control SHARE.
  -- Hold the singleton through reservation commit so guard updates must wait.
  SELECT guard_commitment <> '' INTO STRICT v_review_guard_present
  FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;

  v_nov06_deadline := public.lab_arena__nov06_late_review635_v1(v_round,v_submission);
  v_now := pg_catalog.clock_timestamp();
  v_freeze_at :=
    (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      - pg_catalog.make_interval(secs => COALESCE(v_round.replacement_freeze_seconds,3600));
  v_deadline := COALESCE(
    (v_round.configuration_doc #>> '{schedule,benchmark_deadline}')::TIMESTAMPTZ,
    (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      + INTERVAL '30 minutes'
  );
  IF v_freeze_at IS NULL OR v_deadline IS NULL OR v_freeze_at >= v_deadline THEN
    RAISE EXCEPTION 'lab_arena_code_review_schedule_invalid'
      USING ERRCODE = '22023';
  END IF;

  IF v_nov03_recovery THEN
    v_deadline := LEAST((v_round.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ, (v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ);
  END IF;
  IF v_nov06_deadline IS NOT NULL THEN v_deadline:=v_nov06_deadline; END IF;
  v_is_baseline :=
    v_submission.is_king
    AND v_submission.submission_id =
      'baseline-' || pg_catalog.regexp_replace(v_submission.round_id, '^arena-', '')
    AND v_submission.miner_hotkey =
      v_round.configuration_doc ->> 'baseline_hotkey';
  IF COALESCE(v_is_baseline, FALSE) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'baseline',
      'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts
    );
  END IF;
  IF v_submission.code_review_status IN ('passed', 'rejected') THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing',
      'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts
    );
  END IF;

  -- Completed decisions and exact replay above/below remain read-only.
  IF v_review_guard_present AND NOT (
      v_submission.code_review_status = 'reviewing'
      AND v_submission.code_review_claim IS NOT DISTINCT FROM p_claim_token_hash
  ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'paused');
  END IF;

  IF v_submission.code_review_attempts>0 THEN
    v_call_identity := 'sha256:' || pg_catalog.encode(extensions.digest(
      pg_catalog.convert_to('lab_arena.code_review:' || p_submission_id || ':' ||
        v_submission.code_review_attempts::TEXT,'UTF8'),'sha256'),'hex');
    v_head := public.lab_arena__ledger_head(v_call_identity);
    SELECT * INTO v_reservation FROM public.lab_arena_ledger
      WHERE call_identity=v_call_identity AND entry_kind='reservation';
    v_legacy_refusal := COALESCE(
      v_submission.code_review_status='error'
      AND v_submission.code_review_doc ->> 'error_code'='code_review_provider_request_rejected'
      AND v_submission.code_review_doc -> 'provider_http_status'='200'::JSONB
      AND v_head.entry_kind IN ('settlement','uncertain')
      AND v_head.entry_doc ->> 'error_code'='code_review_provider_request_rejected'
      AND v_head.entry_doc -> 'provider_http_status'='200'::JSONB
      AND v_reservation.entry_doc -> 'attempt'=pg_catalog.to_jsonb(v_submission.code_review_attempts)
      AND v_reservation.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'
      AND TRUE,
      FALSE);
    v_refusal_recovery := COALESCE(v_reservation.entry_doc -> 'refusal_recovery'='true'::JSONB,FALSE)
      OR v_legacy_refusal
      OR (v_submission.code_review_status='error'
        AND v_submission.code_review_doc ->> 'error_code'='code_review_provider_refused'
        AND v_head.entry_kind IN ('settlement','uncertain')
        AND v_head.entry_doc ->> 'error_code'='code_review_provider_refused');
  END IF;
  IF v_refusal_recovery THEN
    v_episode_start := (v_reservation.entry_doc ->> 'refusal_episode_start')::INTEGER;
    IF v_episode_start IS NULL THEN
      SELECT min((reservation.entry_doc ->> 'attempt')::INTEGER) INTO v_episode_start
      FROM public.lab_arena_ledger reservation JOIN public.lab_arena_ledger terminal
        ON terminal.call_identity=reservation.call_identity
      WHERE reservation.submission_id=p_submission_id
        AND reservation.operation_id='openrouter.code_review' AND reservation.entry_kind='reservation'
        AND reservation.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')
        AND terminal.entry_kind IN ('settlement','uncertain')
        AND (terminal.entry_doc ->> 'error_code'='code_review_provider_refused'
          OR (terminal.entry_doc ->> 'error_code'='code_review_provider_request_rejected'
              AND terminal.entry_doc -> 'provider_http_status'='200'::JSONB));
    END IF;
    v_episode_start := COALESCE(v_episode_start,v_submission.code_review_attempts);
  END IF;
  -- The first actual primary reservation is immutable submission authority.
  -- This lookup deliberately ignores refusal episode counter resets. Old rows
  -- without the marker always retain Sol; no historical row is rewritten.
  SELECT * INTO v_first_primary
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND ledger.entry_doc ->> 'review_model' IN
      ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')
  ORDER BY ledger.entry_id LIMIT 1;
  IF v_first_primary.entry_id IS NOT NULL THEN
    IF v_first_primary.entry_doc ? 'review_alternate_model' AND (
        pg_catalog.jsonb_typeof(v_first_primary.entry_doc -> 'review_alternate_model') IS DISTINCT FROM 'string'
        OR v_first_primary.entry_doc ->> 'review_alternate_model' NOT IN
          ('openai/gpt-6.1-sol','moonshotai/kimi-k3.1-pro')
        OR (v_first_primary.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'
            AND v_first_primary.entry_doc ->> 'review_alternate_model'<>'openai/gpt-6.1-sol')) THEN
      RAISE EXCEPTION 'lab_arena_code_review_alternate_binding_invalid' USING ERRCODE='55000';
    END IF;
    v_alternate_model:=COALESCE(v_first_primary.entry_doc ->> 'review_alternate_model','openai/gpt-6.1-sol');
  ELSE
    v_alternate_model:=CASE WHEN p_review_model='deepseek/deepseek-v4.1-flash-1017'
      THEN p_review_alternate_model ELSE 'openai/gpt-6.1-sol' END;
  END IF;
  SELECT count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')),
         count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' IN ('openai/gpt-6.1-sol','moonshotai/kimi-k3.1-pro'))
  INTO v_primary_count,v_alternate_count
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);
  -- Freeze this episode to its first actual primary reservation. Historical
  -- Sonnet episodes keep Sonnet; model IDs and paid counters remain truthful.
  SELECT ledger.entry_doc ->> 'review_model' INTO v_primary_model
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND ledger.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start)
  ORDER BY ledger.entry_id LIMIT 1;
  v_primary_model := COALESCE(v_primary_model,'anthropic/claude-sonnet-5.5');
  v_output_token_recommendation := COALESCE(
    (v_reservation.entry_doc ->> 'review_output_token_recommendation')::INTEGER,32768);
  IF v_head.entry_kind IN ('settlement','uncertain')
     AND v_head.entry_doc ->> 'error_code'='review_response_invalid'
     AND v_head.entry_doc ->> 'error_reason' IN ('truncated','truncated_at_limit') THEN
    v_output_token_recommendation := 65536;
  END IF;
  -- A new finite episode starts at the latest bound unusable review, not at
  -- the global attempt count or at historical unrelated infrastructure errors.
  v_invalid_start := (v_reservation.entry_doc ->> 'invalid_episode_start')::INTEGER;
  IF v_invalid_start IS NULL AND v_submission.code_review_status='error'
     AND v_reservation.entry_id IS NOT NULL
     AND v_reservation.entry_doc -> 'attempt'=pg_catalog.to_jsonb(v_submission.code_review_attempts)
     AND v_head.entry_kind IN ('settlement','uncertain')
     AND v_head.entry_doc ->> 'error_code' IN
       ('review_response_invalid','code_review_research_unverified')
     AND v_head.entry_doc ->> 'error_code' IS NOT DISTINCT FROM
       v_submission.code_review_doc ->> 'error_code' THEN
    v_invalid_start := v_submission.code_review_attempts;
  END IF;
  IF v_invalid_start IS NOT NULL THEN
    SELECT count(*) INTO v_invalid_count FROM public.lab_arena_ledger ledger
    WHERE ledger.submission_id=p_submission_id
      AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
      AND (ledger.entry_doc ->> 'attempt')::INTEGER>=v_invalid_start;
    v_invalid_doc := pg_catalog.jsonb_build_object(
      'invalid_episode_start',v_invalid_start,'review_invalid_attempts',v_invalid_count);
  END IF;
  IF v_submission.code_review_status = 'reviewing' THEN
    IF v_submission.code_review_claim = p_claim_token_hash THEN
      v_call_identity := 'sha256:' || pg_catalog.encode(
        extensions.digest(
          pg_catalog.convert_to(
            'lab_arena.code_review:' || p_submission_id || ':' ||
              v_submission.code_review_attempts::TEXT,
            'UTF8'
          ),
          'sha256'
        ),
        'hex'
      );
      RETURN pg_catalog.jsonb_build_object(
        'status', 'claimed', 'idempotent', TRUE,
        'code_review_status', 'reviewing',
        'attempt', v_submission.code_review_attempts,
        'claim_expires_at', v_submission.code_review_expires_at,
        'call_identity', v_call_identity,
        'reserved_microusd', p_reservation_microusd,
        'review_alternate_model',v_alternate_model
      );
    END IF;
    IF v_submission.code_review_expires_at > v_now THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'busy', 'code_review_status', 'reviewing',
        'attempt', v_submission.code_review_attempts,
        'claim_expires_at', v_submission.code_review_expires_at
      );
    END IF;

    v_call_identity := 'sha256:' || pg_catalog.encode(
      extensions.digest(
        pg_catalog.convert_to(
          'lab_arena.code_review:' || p_submission_id || ':' ||
            v_submission.code_review_attempts::TEXT,
          'UTF8'
        ),
        'sha256'
      ),
      'hex'
    );
    v_head := public.lab_arena__ledger_head(v_call_identity);
    SELECT * INTO v_reservation
    FROM public.lab_arena_ledger
    WHERE call_identity = v_call_identity AND entry_kind = 'reservation';
    IF v_head.entry_id IS NULL OR v_reservation.entry_id IS NULL THEN
      RAISE EXCEPTION 'lab_arena_code_review_ledger_missing'
        USING ERRCODE = '55000';
    END IF;
    IF v_head.entry_kind IN ('reservation', 'dispatch') THEN
      INSERT INTO public.lab_arena_ledger (
        entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
        call_identity, provider, operation_id, funding_source,
        amount_microusd, entry_doc
      ) VALUES (
        'uncertain', v_submission.miner_hotkey, v_submission.round_id,
        v_submission.submission_id, NULL, NULL, v_call_identity,
        'openrouter', 'openrouter.code_review', 'miner_key',
        v_reservation.amount_microusd,
        pg_catalog.jsonb_build_object(
          'reason', 'code_review_claim_expired',
          'error_code', 'code_review_transport_error',
          'retryable', (NOT v_refusal_recovery OR v_alternate_count<2)
            AND (v_invalid_start IS NULL OR v_invalid_count<3),
          'reserved_microusd', v_reservation.amount_microusd
        )
      );
    ELSIF v_head.entry_kind NOT IN ('settlement', 'uncertain') THEN
      RAISE EXCEPTION 'lab_arena_code_review_ledger_invalid'
        USING ERRCODE = '55000';
    END IF;
    SELECT COALESCE(pg_catalog.sum(head.amount_microusd), 0)::BIGINT
    INTO v_review_cost
    FROM (
      SELECT DISTINCT ON (ledger.call_identity)
        ledger.call_identity, ledger.entry_kind, ledger.amount_microusd
      FROM public.lab_arena_ledger AS ledger
      WHERE ledger.submission_id = p_submission_id
        AND ledger.operation_id = 'openrouter.code_review'
        AND ledger.call_identity IS NOT NULL
      ORDER BY ledger.call_identity, ledger.entry_id DESC
    ) AS head
    WHERE head.entry_kind IN
      ('reservation', 'dispatch', 'settlement', 'uncertain');
    v_now := pg_catalog.clock_timestamp();
    UPDATE public.lab_arena_submissions
    SET code_review_status = 'error',
        code_review_doc = pg_catalog.jsonb_build_object(
          'schema_version', 'leadpoet.lab_arena.code_review.result.v1',
          'verdict', 'error',
          'error_code', CASE
            WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
            WHEN (v_round.status <> 'open' AND NOT v_nov03_recovery AND v_nov06_deadline IS NULL) THEN 'code_review_round_closed'
            ELSE 'code_review_transport_error'
          END,
          'retryable', v_now < v_deadline AND (v_round.status = 'open' OR v_nov03_recovery)
            AND (NOT v_refusal_recovery OR v_alternate_count<2)
            AND (v_invalid_start IS NULL OR v_invalid_count<3),
          'model', v_reservation.entry_doc ->> 'review_model',
          'file_count', v_reservation.entry_doc -> 'file_count',
          'source_bytes', v_reservation.entry_doc -> 'source_bytes',
          'cost_microusd', v_reservation.amount_microusd,
          'cost_status', CASE
            WHEN v_head.entry_kind IN ('settlement', 'uncertain')
              THEN v_head.entry_kind
            ELSE 'uncertain'
          END,
          'review_cost_microusd', v_review_cost
        ) || CASE WHEN v_refusal_recovery THEN pg_catalog.jsonb_build_object(
      'refusal_recovery', TRUE, 'refusal_episode_start', v_episode_start, 'review_primary_attempts', v_primary_count,
      'review_alternate_attempts', v_alternate_count, 'review_primary_model', v_primary_model) ELSE '{}'::JSONB END || v_invalid_doc || pg_catalog.jsonb_build_object(
          'review_output_token_recommendation',v_output_token_recommendation,
          'review_alternate_model',v_alternate_model),
        code_review_started_at = v_now,
        code_review_expires_at = NULL
    WHERE submission_id = p_submission_id;
    RETURN pg_catalog.jsonb_build_object(
      'status', CASE
        WHEN v_now >= v_deadline THEN 'deadline'
        WHEN (v_round.status <> 'open' AND NOT v_nov03_recovery AND v_nov06_deadline IS NULL) THEN 'round_closed'
        ELSE 'recovered'
      END,
      'code_review_status', 'error',
      'attempt', v_submission.code_review_attempts
    );
  END IF;

  IF (v_round.status <> 'open' AND NOT v_nov03_recovery AND v_nov06_deadline IS NULL) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'round_closed',
      'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts
    );
  END IF;
  v_now := pg_catalog.clock_timestamp();
  IF v_now >= v_deadline THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'deadline',
      'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts
    );
  END IF;
  -- lab_arena_queued_replacement_review_deferral
  IF v_now < v_freeze_at THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'deferred',
      'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts,
      'retry_after', v_freeze_at
    );
  END IF;

  -- Always recover/settle an expired lease above before enforcing the cap.
  -- Refusal does not reset an existing invalid-review episode.
  IF (v_invalid_start IS NOT NULL AND v_invalid_count>=3
      AND NOT (v_nov06_deadline IS NOT NULL AND v_submission.code_review_attempts=3 AND v_invalid_count=3))
     OR (v_submission.code_review_status='error'
       AND v_head.entry_kind IN ('settlement','uncertain')
       AND v_head.entry_doc ->> 'error_code'='review_response_invalid'
       AND v_head.entry_doc ->> 'error_reason'='truncated_at_limit') THEN
    RETURN pg_catalog.jsonb_build_object('status','exhausted','code_review_status',v_submission.code_review_status,
      'attempt',v_submission.code_review_attempts);
  END IF;

  IF v_refusal_recovery AND (v_submission.code_review_doc ->> 'error_code' IN (
      'code_review_provider_refused','code_review_preparation_unavailable',
      'code_review_provider_rate_limited','code_review_provider_timeout',
      'code_review_provider_unavailable','code_review_transport_error',
      'review_response_invalid','code_review_research_unverified')
      OR v_legacy_refusal) THEN
    IF v_alternate_count>=2 THEN
      RETURN pg_catalog.jsonb_build_object('status','exhausted','code_review_status',v_submission.code_review_status,
        'attempt',v_submission.code_review_attempts);
    END IF;
    v_required_model := CASE WHEN v_primary_count>=2 OR v_alternate_count>0
      THEN v_alternate_model ELSE v_primary_model END;
    IF p_review_model IS DISTINCT FROM v_required_model THEN
      RETURN pg_catalog.jsonb_build_object('status','route_required','review_model',v_required_model);
    END IF;
  ELSIF v_nov06_deadline IS NOT NULL AND v_submission.code_review_attempts=3 THEN
    IF p_review_model IS DISTINCT FROM 'openai/gpt-6.1-sol' THEN
      RETURN jsonb_build_object('status','route_required','review_model','openai/gpt-6.1-sol');
    END IF;
  ELSIF NOT v_refusal_recovery
      AND v_reservation.entry_doc ->> 'review_model'='deepseek/deepseek-v4.1-flash-1017'
      AND v_head.entry_kind IN ('settlement','uncertain')
      AND v_head.entry_doc ->> 'error_code'='review_response_invalid'
      AND v_head.entry_doc ->> 'error_reason'='truncated' THEN
    -- Use the next existing invalid-review slot, never a 65536-token DS call.
    IF p_review_model IS DISTINCT FROM v_alternate_model THEN
      RETURN pg_catalog.jsonb_build_object('status','route_required','review_model',v_alternate_model);
    END IF;
  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL
        AND v_invalid_count>=2 AND v_invalid_count<3 THEN
    -- Spend only the last existing invalid-review reservation on the alternate.
    IF p_review_model IS DISTINCT FROM v_alternate_model THEN
      RETURN pg_catalog.jsonb_build_object('status','route_required','review_model',v_alternate_model);
    END IF;
  ELSIF p_review_model IN ('openai/gpt-6.1-sol','moonshotai/kimi-k3.1-pro') THEN
    RETURN pg_catalog.jsonb_build_object('status','fallback_not_authorized');
  ELSIF v_primary_count>0 AND p_review_model IS DISTINCT FROM v_primary_model THEN
    RETURN pg_catalog.jsonb_build_object('status','route_required','review_model',v_primary_model);
  END IF;

  IF v_submission.code_review_status = 'error' THEN
    v_retryable := CASE
      WHEN v_refusal_recovery AND (v_submission.code_review_doc ->> 'error_code' IN (
      'code_review_provider_refused','code_review_preparation_unavailable',
      'code_review_provider_rate_limited','code_review_provider_timeout',
      'code_review_provider_unavailable','code_review_transport_error',
      'review_response_invalid','code_review_research_unverified')
      OR v_legacy_refusal) THEN TRUE
      WHEN v_submission.code_review_doc ->> 'error_code' IN (
        'code_review_preparation_unavailable', 'code_review_provider_rate_limited',
        'code_review_provider_timeout', 'code_review_provider_unavailable',
        'code_review_transport_error', 'review_response_invalid',
        'code_review_research_unverified'
      ) THEN TRUE
      WHEN v_submission.code_review_doc -> 'retryable' = 'false'::JSONB THEN FALSE
      ELSE NULL
    END;
    v_attempt_limit := CASE
      WHEN v_retryable IS TRUE THEN NULL
      WHEN v_retryable IS FALSE THEN v_submission.code_review_attempts
      ELSE 3
    END;
    IF v_attempt_limit IS NOT NULL
       AND v_submission.code_review_attempts >= v_attempt_limit THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'exhausted', 'code_review_status', 'error',
        'attempt', v_submission.code_review_attempts
      );
    END IF;
    v_backoff_seconds := CASE
      WHEN v_retryable IS TRUE THEN
        (ARRAY[60, 120, 240, 480, 900])[LEAST(
          v_submission.code_review_attempts, 5
        )]
      ELSE 60
    END;
    v_retry_at := GREATEST(
      v_submission.code_review_started_at
        + pg_catalog.make_interval(secs => v_backoff_seconds),
      v_freeze_at
    );
    IF v_retry_at >= v_deadline THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'exhausted', 'code_review_status', 'error',
        'attempt', v_submission.code_review_attempts
      );
    END IF;
    IF v_retry_at > v_now THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'backoff', 'code_review_status', 'error',
        'attempt', v_submission.code_review_attempts,
        'retry_after', v_retry_at
      );
    END IF;
  ELSIF v_submission.code_review_status <> 'pending'
        AND v_submission.code_review_status <> 'reviewing' THEN
    RAISE EXCEPTION 'lab_arena_code_review_state_invalid'
      USING ERRCODE = '55000';
  END IF;

  v_attempt := v_submission.code_review_attempts + 1;
  v_expires := LEAST(
    v_now + pg_catalog.make_interval(secs => 600),
    v_deadline
  );
  v_call_identity := 'sha256:' || pg_catalog.encode(
    extensions.digest(
      pg_catalog.convert_to(
        'lab_arena.code_review:' || p_submission_id || ':' || v_attempt::TEXT,
        'UTF8'
      ),
      'sha256'
    ),
    'hex'
  );
  IF (public.lab_arena__ledger_head(v_call_identity)).entry_id IS NOT NULL THEN
    RAISE EXCEPTION 'lab_arena_code_review_call_identity_conflict'
      USING ERRCODE = '23505';
  END IF;

  UPDATE public.lab_arena_submissions
  SET code_review_status = 'reviewing',
      code_review_doc = NULL,
      code_review_claim = p_claim_token_hash,
      code_review_started_at = v_now,
      code_review_expires_at = v_expires,
      code_review_attempts = v_attempt
  WHERE submission_id = p_submission_id;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc
  ) VALUES (
    'reservation', v_submission.miner_hotkey, v_submission.round_id,
    v_submission.submission_id, NULL, NULL, v_call_identity,
    'openrouter', 'openrouter.code_review', 'miner_key',
    p_reservation_microusd,
    pg_catalog.jsonb_build_object(
      'attempt', v_attempt,
      'review_alternate_model',v_alternate_model,
      'review_model', p_review_model, 'refusal_recovery', v_refusal_recovery, 'refusal_episode_start', v_episode_start,
      'invalid_episode_start', v_invalid_start,
      'review_output_token_recommendation',v_output_token_recommendation,
      'file_count', p_file_count,
      'source_bytes', p_source_bytes
    )
  );
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc
  ) VALUES (
    'dispatch', v_submission.miner_hotkey, v_submission.round_id,
    v_submission.submission_id, NULL, NULL, v_call_identity,
    'openrouter', 'openrouter.code_review', 'miner_key',
    p_reservation_microusd,
    pg_catalog.jsonb_build_object('attempt', v_attempt)
  );
  RETURN pg_catalog.jsonb_build_object(
    'status', 'claimed', 'idempotent', FALSE,
    'code_review_status', 'reviewing', 'attempt', v_attempt,
    'claim_expires_at', v_expires,
    'call_identity', v_call_identity,
    'reserved_microusd', p_reservation_microusd,
    'review_output_token_recommendation',v_output_token_recommendation,
    'review_alternate_model',v_alternate_model
  );
END;
$function$
$v2_669$;
  ALTER FUNCTION public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text) OWNER TO lab_arena_owner;
  REVOKE ALL ON FUNCTION public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text) FROM PUBLIC,anon,authenticated,service_role;
  GRANT EXECUTE ON FUNCTION public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text) TO lab_arena_service;
  IF NOT v_has_create THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
  EXECUTE $wrapper669$CREATE OR REPLACE FUNCTION public.lab_arena_begin_submission_review(p_submission_id text, p_miner_hotkey text, p_claim_token_hash text, p_reservation_microusd bigint, p_review_model text, p_file_count integer, p_source_bytes bigint)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
BEGIN
  RETURN public.lab_arena_begin_submission_review_v2(
    p_submission_id,p_miner_hotkey,p_claim_token_hash,p_reservation_microusd,
    p_review_model,p_file_count,p_source_bytes,'openai/gpt-6.1-sol');
END;
$function$
$wrapper669$;
  IF (SELECT nspacl::TEXT FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM v_schema_acl
     OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=to_regprocedure(v_signature)) IS DISTINCT FROM v_metadata
     OR encode(extensions.digest(pg_get_functiondef(to_regprocedure(v_signature)),'sha256'),'hex') IS DISTINCT FROM '2ade4eb49ebeddead5cf0356f6f94e273a0a4d3d5de47de294c1576ca3c8cbb8'
     OR encode(extensions.digest(pg_get_functiondef(to_regprocedure(v_v2_signature)),'sha256'),'hex') IS DISTINCT FROM 'e7c425663c97dc4a0bf615081c60c7d6e3ac2f18fc0d90456737d027b37ea8df'
     OR (SELECT jsonb_build_array(pg_get_userbyid(proowner),proacl::TEXT,prosecdef,provolatile,proconfig)
         FROM pg_proc WHERE oid=to_regprocedure(v_v2_signature)) IS DISTINCT FROM v_identity THEN
    RAISE EXCEPTION 'Arena alternate669 readback differs' USING ERRCODE='55000';
  END IF;
END;
$begin669$;

DO $alternate669_1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '91f4d574ab3d1d704228adac40507f6c4fb80a49fd091019ca713af1a0ff8a96';
  v_postimage CONSTANT TEXT := '54414a7f9d07d00111206bbd02884896e2fee89c12f80eabdd35c344b0767530';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_old TEXT;
  v_new TEXT;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena alternate669 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena alternate669 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old669_1_0$  v_primary_model TEXT;$old669_1_0$;
  v_new:=$new669_1_0$  v_primary_model TEXT;
  v_alternate_model TEXT;
  v_first_primary public.lab_arena_ledger;$new669_1_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old669_1_1$  SELECT count(*) FILTER (WHERE ledger.entry_doc$old669_1_1$;
  v_new:=$new669_1_1$  -- The first actual primary reservation is immutable submission authority.
  -- This lookup deliberately ignores refusal episode counter resets. Old rows
  -- without the marker always retain Sol; no historical row is rewritten.
  SELECT * INTO v_first_primary
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND ledger.entry_doc ->> 'review_model' IN
      ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')
  ORDER BY ledger.entry_id LIMIT 1;
  IF v_first_primary.entry_id IS NOT NULL THEN
    IF v_first_primary.entry_doc ? 'review_alternate_model' AND (
        pg_catalog.jsonb_typeof(v_first_primary.entry_doc -> 'review_alternate_model') IS DISTINCT FROM 'string'
        OR v_first_primary.entry_doc ->> 'review_alternate_model' NOT IN
          ('openai/gpt-6.1-sol','moonshotai/kimi-k3.1-pro')
        OR (v_first_primary.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'
            AND v_first_primary.entry_doc ->> 'review_alternate_model'<>'openai/gpt-6.1-sol')) THEN
      RAISE EXCEPTION 'lab_arena_code_review_alternate_binding_invalid' USING ERRCODE='55000';
    END IF;
    v_alternate_model:=COALESCE(v_first_primary.entry_doc ->> 'review_alternate_model','openai/gpt-6.1-sol');
  ELSE
    v_alternate_model:='openai/gpt-6.1-sol';
  END IF;
  SELECT count(*) FILTER (WHERE ledger.entry_doc$new669_1_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old669_1_2$count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' = 'openai/gpt-6.1-sol')$old669_1_2$;
  v_new:=$new669_1_2$count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' IN ('openai/gpt-6.1-sol','moonshotai/kimi-k3.1-pro'))$new669_1_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old669_1_3$'deepseek/deepseek-v4.1-flash-1017','deepseek/deepseek-v4.1-flash-20261017')$old669_1_3$;
  v_new:=$new669_1_3$'deepseek/deepseek-v4.1-flash-1017','deepseek/deepseek-v4.1-flash-20261017',
                'moonshotai/kimi-k3.1-pro','moonshotai/kimi-k3.1-pro-20261023')$new669_1_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old669_1_4$('Anthropic','Amazon Bedrock','Google','Google Vertex','Azure','Nebius')$old669_1_4$;
  v_new:=$new669_1_4$('Anthropic','Amazon Bedrock','Google','Google Vertex','Azure','Nebius','Moonshot AI')$new669_1_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old669_1_5$  IF v_refusal_recovery AND p_status='error' THEN$old669_1_5$;
  v_new:=$new669_1_5$  -- Only an exact Kimi-bound reservation can finish as Kimi. Caller fields
  -- cannot authorize a route or replace the immutable first-primary marker.
  IF v_reservation.entry_doc ->> 'review_model'='moonshotai/kimi-k3.1-pro' AND (
      v_alternate_model IS DISTINCT FROM 'moonshotai/kimi-k3.1-pro'
      OR v_reservation.entry_doc ->> 'review_alternate_model' IS DISTINCT FROM v_alternate_model
      OR (p_status IN ('passed','rejected') AND (
        v_provider_doc #>> '{provider_diagnostic,context_integrity}' IS DISTINCT FROM 'verified'
        OR COALESCE(v_provider_doc #>> '{provider_diagnostic,selected_model}','') NOT IN
          ('moonshotai/kimi-k3.1-pro','moonshotai/kimi-k3.1-pro-20261023')
        OR v_provider_doc #>> '{provider_diagnostic,selected_provider}' IS DISTINCT FROM 'Moonshot AI'
        OR COALESCE(v_provider_doc #>> '{provider_diagnostic,max_output_tokens}','') NOT IN ('32768','65536')
        OR COALESCE(v_provider_doc #>> '{provider_diagnostic,prompt_tokens}','') !~ '^[1-9][0-9]{0,5}$'
        OR (v_provider_doc #>> '{provider_diagnostic,prompt_tokens}')::BIGINT>922000
      ))) THEN
    RAISE EXCEPTION 'lab_arena_code_review_alternate_binding_invalid' USING ERRCODE='22023';
  END IF;
  IF v_refusal_recovery AND p_status='error' THEN$new669_1_5$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.5 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old669_1_6$'review_output_token_recommendation',v_output_token_recommendation);$old669_1_6$;
  v_new:=$new669_1_6$'review_output_token_recommendation',v_output_token_recommendation,
      'review_alternate_model',v_alternate_model);$new669_1_6$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.6 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old669_1_7$'review_output_token_recommendation',v_output_token_recommendation,
    'attempt'$old669_1_7$;
  v_new:=$new669_1_7$'review_output_token_recommendation',v_output_token_recommendation,
    'review_alternate_model',v_alternate_model,
    'attempt'$new669_1_7$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 1.7 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena alternate669 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena alternate669 readback differs' USING ERRCODE='55000';
  END IF;
END;
$alternate669_1$;

DO $alternate669_2$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_code_review_schema_v1()';
  v_preimage CONSTANT TEXT := '55c3a3644ed53787bcff9af034e6be7b2a325474503e1bc5aa761de92a6d4d16';
  v_postimage CONSTANT TEXT := 'bbda94432cfee29aee621964cddcc174a3ce308e680ba2da00ab47099def6d51';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_old TEXT;
  v_new TEXT;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena alternate669 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena alternate669 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old669_2_0$    'refusal_fallback',$old669_2_0$;
  v_new:=$new669_2_0$    'prospective_alternate', pg_catalog.jsonb_build_object(
      'model','moonshotai/kimi-k3.1-pro', 'legacy_model','openai/gpt-6.1-sol',
      'rpc','lab_arena_begin_submission_review_v2',
      'binding','first_primary_reservation_marker_or_legacy_sol',
      'context_window_tokens',1048576, 'max_output_tokens',65536),
    'refusal_fallback',$new669_2_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena alternate669 anchor 2.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena alternate669 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena alternate669 readback differs' USING ERRCODE='55000';
  END IF;
END;
$alternate669_2$;

NOTIFY pgrst, 'reload schema';
COMMIT;
