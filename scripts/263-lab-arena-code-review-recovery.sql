-- Add bounded retry recovery for safely classified temporary code-review failures.
-- The existing RPC signatures remain compatible with a rolling gateway deploy.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_263_requires_current_review$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_submission_replacement_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply Arena migrations through 262 first';
  END IF;
END;
$lab_arena_263_requires_current_review$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_code_review_state_check;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_code_review_state_check
  CHECK (
    (
      code_review_status = 'pending'
      AND code_review_attempts = 0
      AND code_review_doc IS NULL
      AND code_review_claim IS NULL
      AND code_review_started_at IS NULL
      AND code_review_expires_at IS NULL
    )
    OR (
      code_review_status = 'reviewing'
      AND code_review_attempts BETWEEN 1 AND 6
      AND code_review_doc IS NULL
      AND code_review_claim IS NOT NULL
      AND code_review_started_at IS NOT NULL
      AND code_review_expires_at > code_review_started_at
    )
    OR (
      code_review_status IN ('passed', 'rejected', 'error')
      AND code_review_attempts BETWEEN 1 AND 6
      AND code_review_doc IS NOT NULL
      AND code_review_claim IS NOT NULL
      AND code_review_started_at IS NOT NULL
      AND code_review_expires_at IS NULL
    )
  );

CREATE OR REPLACE FUNCTION public.lab_arena_code_review_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_code_review_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.code_review.v1',
    'version', 207,
    'claim_ttl_seconds', 600,
    'retry_backoff_seconds', 60,
    'max_attempts', 3,
    'retry_policy', 'bounded_transient_v1',
    'max_transient_attempts', 6,
    'transient_retry_backoff_seconds',
      pg_catalog.jsonb_build_array(60, 120, 240, 480, 900),
    'legacy_max_attempts', 3,
    'retry_window', 'replacement_freeze_to_benchmark_deadline'
  );
$lab_arena_code_review_schema$;
ALTER FUNCTION public.lab_arena_code_review_schema_v1()
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_begin_submission_review(
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_claim_token_hash TEXT,
  p_reservation_microusd BIGINT,
  p_review_model TEXT,
  p_file_count INTEGER,
  p_source_bytes BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_begin_submission_review$
DECLARE
  v_round_id TEXT;
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_now TIMESTAMPTZ;
  v_freeze_at TIMESTAMPTZ;
  v_deadline TIMESTAMPTZ;
  v_retry_at TIMESTAMPTZ;
  v_expires TIMESTAMPTZ;
  v_attempt SMALLINT;
  v_attempt_limit SMALLINT;
  v_backoff_seconds INTEGER;
  v_retryable BOOLEAN;
  v_call_identity TEXT;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_review_cost BIGINT;
BEGIN
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
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
  v_now := pg_catalog.clock_timestamp();
  v_freeze_at :=
    (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      - INTERVAL '1 hour';
  v_deadline := COALESCE(
    (v_round.configuration_doc #>> '{schedule,benchmark_deadline}')::TIMESTAMPTZ,
    (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      + INTERVAL '30 minutes'
  );
  IF v_freeze_at IS NULL OR v_deadline IS NULL OR v_freeze_at >= v_deadline THEN
    RAISE EXCEPTION 'lab_arena_code_review_schedule_invalid'
      USING ERRCODE = '22023';
  END IF;

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
        'reserved_microusd', p_reservation_microusd
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
          'retryable', TRUE,
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
            WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
            ELSE 'code_review_transport_error'
          END,
          'retryable', v_now < v_deadline AND v_round.status = 'open',
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
        ),
        code_review_started_at = v_now,
        code_review_expires_at = NULL
    WHERE submission_id = p_submission_id;
    RETURN pg_catalog.jsonb_build_object(
      'status', CASE
        WHEN v_now >= v_deadline THEN 'deadline'
        WHEN v_round.status <> 'open' THEN 'round_closed'
        ELSE 'recovered'
      END,
      'code_review_status', 'error',
      'attempt', v_submission.code_review_attempts
    );
  END IF;

  IF v_round.status <> 'open' THEN
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

  IF v_submission.code_review_status = 'error' THEN
    v_retryable := CASE
      WHEN pg_catalog.jsonb_typeof(v_submission.code_review_doc -> 'retryable')
             = 'boolean'
      THEN (v_submission.code_review_doc ->> 'retryable')::BOOLEAN
      ELSE NULL
    END;
    v_attempt_limit := CASE
      WHEN v_retryable IS TRUE THEN 6
      WHEN v_retryable IS FALSE THEN v_submission.code_review_attempts
      ELSE 3
    END;
    IF v_submission.code_review_attempts >= v_attempt_limit THEN
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
      'review_model', p_review_model,
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
    'reserved_microusd', p_reservation_microusd
  );
END;
$lab_arena_begin_submission_review$;
ALTER FUNCTION public.lab_arena_begin_submission_review(
  TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, BIGINT
) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_finish_submission_review(
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_claim_token_hash TEXT,
  p_status TEXT,
  p_review_doc JSONB,
  p_actual_microusd BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_finish_submission_review$
DECLARE
  v_round_id TEXT;
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_call_identity TEXT;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_ledger_status TEXT;
  v_call_cost BIGINT;
  v_review_cost BIGINT;
  v_now TIMESTAMPTZ;
  v_deadline TIMESTAMPTZ;
  v_effective_status TEXT;
  v_effective_doc JSONB;
  v_diagnostic JSONB := '{}'::JSONB;
  v_error_code TEXT;
  v_http_status INTEGER;
  v_retryable BOOLEAN;
BEGIN
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
     OR COALESCE(p_miner_hotkey, '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
     OR COALESCE(p_claim_token_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR p_status NOT IN ('passed', 'rejected', 'error')
     OR pg_catalog.jsonb_typeof(p_review_doc) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_review_doc::TEXT) > 30000
     OR (p_actual_microusd IS NOT NULL
         AND p_actual_microusd NOT BETWEEN 0 AND 1000000000) THEN
    RAISE EXCEPTION 'lab_arena_code_review_input_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF p_status = 'error' THEN
    v_error_code := p_review_doc ->> 'error_code';
    IF v_error_code IS NULL
       OR v_error_code NOT IN (
         'code_review_source_size_mismatch',
         'review_context_limit_invalid', 'review_model_invalid',
         'review_output_cannot_report_coverage', 'review_output_limit_invalid',
         'review_source_empty', 'review_source_exceeds_context',
         'review_source_incomplete', 'review_source_invalid',
         'review_source_not_text', 'review_source_not_utf8',
         'review_source_unreadable', 'source_archive_invalid',
         'source_archive_failed', 'source_archive_too_large',
         'source_archive_unsafe', 'source_contains_credentials',
         'source_directory_missing', 'source_empty',
         'source_entry_type_invalid', 'source_file_count_exceeded',
         'source_git_automation_forbidden', 'source_license_invalid',
         'source_path_invalid', 'source_unpacked_too_large',
         'source_unreadable', 'source_credentials_invalid',
         'harness_file_missing', 'harness_invalid', 'harness_too_large',
         'code_review_preparation_unavailable',
         'code_review_provider_authentication', 'code_review_provider_credit',
         'code_review_provider_rate_limited',
         'code_review_provider_request_rejected',
         'code_review_provider_timeout', 'code_review_provider_unavailable',
         'code_review_transport_error', 'code_review_credential_echo',
         'code_review_cost_unavailable', 'code_review_deadline_exceeded',
         'code_review_round_closed', 'code_review_submission_ineligible',
         'review_response_invalid'
       )
       OR (
         p_review_doc ? 'provider_http_status'
         AND (
           pg_catalog.jsonb_typeof(p_review_doc -> 'provider_http_status') <> 'number'
           OR (p_review_doc ->> 'provider_http_status')::INTEGER NOT BETWEEN 100 AND 599
         )
       )
       OR (
         p_review_doc ? 'error_reason'
         AND (
           pg_catalog.jsonb_typeof(p_review_doc -> 'error_reason') <> 'string'
           OR p_review_doc ->> 'error_reason' NOT IN (
             'envelope', 'model_mismatch', 'choice', 'not_finished',
             'message', 'refusal_or_tools', 'content', 'content_json',
             'document_keys', 'verdict', 'summary', 'coverage',
             'coverage_order', 'findings', 'verdict_findings',
             'finding_keys', 'finding_classification', 'finding_evidence',
             'finding_evidence_mismatch', 'finding_explanation'
           )
         )
       )
       OR (
         p_review_doc ? 'retryable'
         AND pg_catalog.jsonb_typeof(p_review_doc -> 'retryable') <> 'boolean'
       ) THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_http_status := (p_review_doc ->> 'provider_http_status')::INTEGER;
    v_retryable := (p_review_doc ->> 'retryable')::BOOLEAN;
    IF (v_retryable IS TRUE AND v_error_code NOT IN (
          'code_review_preparation_unavailable',
          'code_review_provider_rate_limited', 'code_review_provider_timeout',
          'code_review_provider_unavailable', 'code_review_transport_error'
        ))
       OR (v_retryable IS FALSE AND v_error_code IN (
          'code_review_provider_rate_limited', 'code_review_provider_timeout',
          'code_review_transport_error'
        )) THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_diagnostic := pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
      'error_code', v_error_code,
      'provider_http_status', v_http_status,
      'error_reason', p_review_doc ->> 'error_reason',
      'retryable', v_retryable
    ));
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
     OR v_submission.status NOT IN ('accepted', 'frozen', 'rejected') THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  v_now := pg_catalog.clock_timestamp();
  v_deadline := COALESCE(
    (v_round.configuration_doc #>> '{schedule,benchmark_deadline}')::TIMESTAMPTZ,
    (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      + INTERVAL '30 minutes'
  );
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
  IF v_submission.code_review_status IN ('passed', 'rejected', 'error')
     AND v_submission.code_review_claim = p_claim_token_hash THEN
    IF v_submission.code_review_status = p_status THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing',
        'code_review_status', v_submission.code_review_status,
        'attempt', v_submission.code_review_attempts
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale',
      'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts
    );
  END IF;
  IF v_submission.code_review_status <> 'reviewing'
     OR v_submission.code_review_claim <> p_claim_token_hash THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale',
      'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts
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
  IF v_head.entry_kind <> 'dispatch' OR v_reservation.entry_id IS NULL THEN
    RAISE EXCEPTION 'lab_arena_code_review_ledger_invalid'
      USING ERRCODE = '55000';
  END IF;
  IF p_review_doc ->> 'model' IS DISTINCT FROM
       v_reservation.entry_doc ->> 'review_model'
     OR p_review_doc -> 'file_count' IS DISTINCT FROM
       v_reservation.entry_doc -> 'file_count'
     OR p_review_doc -> 'source_bytes' IS DISTINCT FROM
       v_reservation.entry_doc -> 'source_bytes'
     OR (p_status = 'passed' AND (
       p_actual_microusd IS NULL
       OR p_review_doc -> 'passed' IS DISTINCT FROM 'true'::JSONB
       OR p_review_doc ->> 'verdict' IS DISTINCT FROM 'pass'
       OR (v_reservation.entry_doc ->> 'file_count')::INTEGER <= 0
       OR (v_reservation.entry_doc ->> 'source_bytes')::BIGINT <= 0
     ))
     OR (p_status = 'rejected' AND (
       p_review_doc -> 'passed' IS DISTINCT FROM 'false'::JSONB
       OR p_review_doc ->> 'verdict' IS DISTINCT FROM 'reject'
       OR (v_reservation.entry_doc ->> 'file_count')::INTEGER <= 0
       OR (v_reservation.entry_doc ->> 'source_bytes')::BIGINT <= 0
     )) THEN
    RAISE EXCEPTION 'lab_arena_code_review_input_invalid'
      USING ERRCODE = '22023';
  END IF;

  IF p_status IN ('passed', 'rejected') AND (
       pg_catalog.jsonb_typeof(p_review_doc -> 'categories') IS DISTINCT FROM 'array'
       OR EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements_text(
           p_review_doc -> 'categories'
         ) AS category(value)
         WHERE category.value NOT IN (
           'hardcoded_prepared_answers', 'fabricated_evidence',
           'malicious_behavior', 'reviewer_manipulation'
         )
       )
     ) THEN
    RAISE EXCEPTION 'lab_arena_code_review_input_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_effective_status := p_status;
  v_effective_doc := pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.code_review.result.v1',
    'verdict', p_status,
    'model', p_review_doc ->> 'model',
    'file_count', p_review_doc -> 'file_count',
    'source_bytes', p_review_doc -> 'source_bytes'
  ) || CASE
    WHEN p_status IN ('passed', 'rejected') THEN pg_catalog.jsonb_build_object(
      'passed', p_status = 'passed',
      'verdict', CASE WHEN p_status = 'passed' THEN 'pass' ELSE 'reject' END,
      'categories', p_review_doc -> 'categories'
    )
    ELSE v_diagnostic
  END;
  -- Recheck after both locks. A provider result can settle after the deadline,
  -- but it cannot create a new participant-admissible pass.
  v_now := pg_catalog.clock_timestamp();
  IF (v_now >= v_deadline OR v_round.status <> 'open'
      OR v_submission.status <> 'accepted')
     AND p_status IN ('passed', 'rejected') THEN
    v_effective_status := 'error';
    v_diagnostic := pg_catalog.jsonb_build_object(
      'error_code', CASE
        WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
        WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
        ELSE 'code_review_submission_ineligible'
      END,
      'retryable', FALSE
    );
    v_effective_doc := pg_catalog.jsonb_build_object(
      'schema_version', 'leadpoet.lab_arena.code_review.result.v1',
      'verdict', 'error',
      'model', p_review_doc ->> 'model',
      'file_count', p_review_doc -> 'file_count',
      'source_bytes', p_review_doc -> 'source_bytes'
    )
      || v_diagnostic;
  END IF;

  IF p_actual_microusd IS NULL THEN
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
        'reason', 'code_review_cost_unknown',
        'review_status', v_effective_status,
        'reserved_microusd', v_reservation.amount_microusd
      ) || v_diagnostic
    );
    v_ledger_status := 'uncertain';
    v_call_cost := v_reservation.amount_microusd;
  ELSE
    INSERT INTO public.lab_arena_ledger (
      entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
      call_identity, provider, operation_id, funding_source,
      amount_microusd, entry_doc, terminal_response
    ) VALUES (
      'settlement', v_submission.miner_hotkey, v_submission.round_id,
      v_submission.submission_id, NULL, NULL, v_call_identity,
      'openrouter', 'openrouter.code_review', 'miner_key',
      p_actual_microusd,
      pg_catalog.jsonb_build_object(
        'reserved_microusd', v_reservation.amount_microusd,
        'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
        'variance_microusd', p_actual_microusd - v_reservation.amount_microusd,
        'review_status', v_effective_status
      ) || v_diagnostic,
      pg_catalog.jsonb_build_object('review_status', v_effective_status)
    );
    v_ledger_status := 'settled';
    v_call_cost := p_actual_microusd;
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

  UPDATE public.lab_arena_submissions
  SET code_review_status = v_effective_status,
      code_review_doc = v_effective_doc || pg_catalog.jsonb_build_object(
        'cost_microusd', v_call_cost,
        'cost_status', v_ledger_status,
        'review_cost_microusd', v_review_cost
      ),
      code_review_started_at = CASE
        WHEN v_effective_status = 'error' THEN v_now
        ELSE code_review_started_at
      END,
      code_review_expires_at = NULL
  WHERE submission_id = p_submission_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE
      WHEN v_effective_status = 'error' AND p_status <> 'error'
        THEN CASE v_diagnostic ->> 'error_code'
          WHEN 'code_review_deadline_exceeded' THEN 'deadline'
          WHEN 'code_review_round_closed' THEN 'round_closed'
          ELSE 'submission_ineligible'
        END
      ELSE v_effective_status
    END,
    'code_review_status', v_effective_status,
    'attempt', v_submission.code_review_attempts,
    'ledger_status', v_ledger_status,
    'call_identity', v_call_identity,
    'actual_microusd', p_actual_microusd,
    'cost_microusd', v_call_cost,
    'review_cost_microusd', v_review_cost
  );
END;
$lab_arena_finish_submission_review$;
ALTER FUNCTION public.lab_arena_finish_submission_review(
  TEXT, TEXT, TEXT, TEXT, JSONB, BIGINT
) OWNER TO lab_arena_owner;

-- Deadline participant finalization uses this existing transition RPC. Close
-- an in-flight review in the same locked transaction that rejects its source,
-- so no dispatched call is left without a conservative ledger terminal.
CREATE OR REPLACE FUNCTION public.lab_arena_update_submission(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_expected_status TEXT,
  p_next_status TEXT,
  p_patch JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_update_submission$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_patch JSONB := COALESCE(p_patch, '{}'::JSONB);
  v_call_identity TEXT;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_review_cost BIGINT;
  v_call_cost BIGINT;
  v_cost_status TEXT;
BEGIN
  IF pg_catalog.jsonb_typeof(v_patch) <> 'object' THEN
    RAISE EXCEPTION 'lab_arena_patch_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'window_closed', 'round_status', v_round.status
    );
  END IF;
  SELECT * INTO v_submission
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id AND round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_submission.status <> p_expected_status THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'submission_status', v_submission.status
    );
  END IF;

  IF p_expected_status = 'uploading' AND p_next_status = 'accepted' THEN
    IF v_submission.source_ref IS NULL
       OR v_submission.source_size_bytes IS NULL
       OR (v_patch - 'is_king') <> '{}'::JSONB THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_submissions
    SET status = 'accepted',
        is_king = COALESCE((v_patch ->> 'is_king')::BOOLEAN, is_king),
        updated_at = pg_catalog.clock_timestamp()
    WHERE submission_id = p_submission_id;
  ELSIF p_expected_status IN ('uploading', 'accepted')
        AND p_next_status = 'rejected' THEN
    IF COALESCE(v_patch ->> 'rejection_rule', '') = ''
       OR (v_patch - 'rejection_rule') <> '{}'::JSONB THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF p_expected_status = 'accepted'
       AND v_patch ->> 'rejection_rule' = 'code_review_incomplete'
       AND v_submission.code_review_status = 'reviewing' THEN
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
            'reason', 'code_review_deadline_rejection',
            'error_code', 'code_review_deadline_exceeded',
            'retryable', FALSE,
            'reserved_microusd', v_reservation.amount_microusd
          )
        );
        v_call_cost := v_reservation.amount_microusd;
        v_cost_status := 'uncertain';
      ELSIF v_head.entry_kind IN ('settlement', 'uncertain') THEN
        v_call_cost := v_head.amount_microusd;
        v_cost_status := CASE
          WHEN v_head.entry_kind = 'settlement' THEN 'settled'
          ELSE 'uncertain'
        END;
      ELSE
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
      UPDATE public.lab_arena_submissions
      SET status = 'rejected',
          rejection_rule = v_patch ->> 'rejection_rule',
          code_review_status = 'error',
          code_review_doc = pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.lab_arena.code_review.result.v1',
            'verdict', 'error',
            'error_code', 'code_review_deadline_exceeded',
            'retryable', FALSE,
            'model', v_reservation.entry_doc ->> 'review_model',
            'file_count', v_reservation.entry_doc -> 'file_count',
            'source_bytes', v_reservation.entry_doc -> 'source_bytes',
            'cost_microusd', v_call_cost,
            'cost_status', v_cost_status,
            'review_cost_microusd', v_review_cost
          ),
          code_review_started_at = pg_catalog.clock_timestamp(),
          code_review_expires_at = NULL,
          updated_at = pg_catalog.clock_timestamp()
      WHERE submission_id = p_submission_id;
    ELSE
      UPDATE public.lab_arena_submissions
      SET status = 'rejected',
          rejection_rule = v_patch ->> 'rejection_rule',
          updated_at = pg_catalog.clock_timestamp()
      WHERE submission_id = p_submission_id;
    END IF;
  ELSIF p_expected_status = 'accepted' AND p_next_status = 'frozen' THEN
    IF (v_patch - 'is_king') <> '{}'::JSONB THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_submissions
    SET status = 'frozen',
        frozen_at = pg_catalog.clock_timestamp(),
        is_king = COALESCE((v_patch ->> 'is_king')::BOOLEAN, is_king),
        updated_at = pg_catalog.clock_timestamp()
    WHERE submission_id = p_submission_id;
  ELSE
    RAISE EXCEPTION 'lab_arena_transition_invalid' USING ERRCODE = '22023';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'submission_status', p_next_status
  );
END;
$lab_arena_update_submission$;
ALTER FUNCTION public.lab_arena_update_submission(TEXT, TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;

DO $lab_arena_code_review_retry_acl$
DECLARE
  signature TEXT;
  role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_code_review_schema_v1()',
    'public.lab_arena_begin_submission_review(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, BIGINT)',
    'public.lab_arena_finish_submission_review(TEXT, TEXT, TEXT, TEXT, JSONB, BIGINT)'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
        EXECUTE pg_catalog.format(
          'REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name
        );
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format(
      'GRANT EXECUTE ON FUNCTION %s TO lab_arena_service', signature
    );
  END LOOP;
END;
$lab_arena_code_review_retry_acl$;

COMMENT ON FUNCTION public.lab_arena_begin_submission_review(
  TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, BIGINT
) IS
  'Claims one review using bounded classified retry policy before the benchmark deadline.';
COMMENT ON FUNCTION public.lab_arena_finish_submission_review(
  TEXT, TEXT, TEXT, TEXT, JSONB, BIGINT
) IS
  'Settles one exact review claim with bounded diagnostics and no late pass admission.';

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
