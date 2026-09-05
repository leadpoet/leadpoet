-- 185-lab-arena-miner-credentials.sql
-- Admit miner runtime credentials as KMS ciphertext and account their calls.

BEGIN;

DO $lab_arena_185_requires_184$
BEGIN
  IF COALESCE((public.lab_arena_schema_version_v1() ->> 'version')::INTEGER, 0)
     NOT IN (184, 185) THEN
    RAISE EXCEPTION 'apply 184-lab-arena-scoring-failure-isolation.sql first';
  END IF;
END;
$lab_arena_185_requires_184$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_submission_credentials (
  submission_id TEXT NOT NULL
    REFERENCES public.lab_arena_submissions (submission_id),
  miner_hotkey TEXT NOT NULL
    CHECK (miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  provider TEXT NOT NULL CHECK (provider IN ('openrouter', 'deepline')),
  ciphertext BYTEA NOT NULL
    CHECK (pg_catalog.octet_length(ciphertext) BETWEEN 1 AND 8192),
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  PRIMARY KEY (submission_id, provider)
);
ALTER TABLE public.lab_arena_submission_credentials OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_submission_credentials ENABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS lab_arena_submission_credentials_append_only
  ON public.lab_arena_submission_credentials;
CREATE TRIGGER lab_arena_submission_credentials_append_only
  BEFORE UPDATE OR DELETE ON public.lab_arena_submission_credentials
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

-- Keep the legacy transition RPC for the public baseline and later
-- accepted/frozen transitions, but make a miner's uploading -> accepted edge
-- impossible until both ciphertext rows exist. The atomic admission RPC
-- inserts both rows before it updates the source slot.
CREATE OR REPLACE FUNCTION public.lab_arena_submission_credentials_required_v1()
RETURNS trigger
LANGUAGE plpgsql
VOLATILE
SET search_path = pg_catalog, public
AS $lab_arena_submission_credentials_required$
DECLARE
  v_round public.lab_arena_rounds;
  v_count INTEGER;
  v_is_baseline BOOLEAN;
BEGIN
  IF OLD.status = 'uploading' AND NEW.status = 'accepted' THEN
    SELECT * INTO v_round
    FROM public.lab_arena_rounds
    WHERE round_id = NEW.round_id;
    v_is_baseline :=
      NEW.is_king
      AND NEW.submission_id =
        'baseline-' || pg_catalog.regexp_replace(NEW.round_id, '^arena-', '')
      AND NEW.miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey';
    IF NOT COALESCE(v_is_baseline, FALSE) THEN
      SELECT COUNT(*) INTO v_count
      FROM public.lab_arena_submission_credentials
      WHERE submission_id = NEW.submission_id
        AND miner_hotkey = NEW.miner_hotkey
        AND provider IN ('openrouter', 'deepline');
      IF v_count <> 2 THEN
        RAISE EXCEPTION 'lab_arena_submission_credentials_required'
          USING ERRCODE = '23514';
      END IF;
    END IF;
  END IF;
  RETURN NEW;
END;
$lab_arena_submission_credentials_required$;
ALTER FUNCTION public.lab_arena_submission_credentials_required_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_submission_credentials_required_v1()
  FROM PUBLIC, lab_arena_service;

DROP TRIGGER IF EXISTS lab_arena_submission_credentials_required
  ON public.lab_arena_submissions;
CREATE TRIGGER lab_arena_submission_credentials_required
  BEFORE UPDATE ON public.lab_arena_submissions
  FOR EACH ROW EXECUTE FUNCTION
    public.lab_arena_submission_credentials_required_v1();

REVOKE ALL ON TABLE public.lab_arena_submission_credentials
  FROM PUBLIC, lab_arena_service;
DO $lab_arena_credential_table_acl$
DECLARE
  role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON TABLE public.lab_arena_submission_credentials FROM %I',
        role_name
      );
    END IF;
  END LOOP;
END;
$lab_arena_credential_table_acl$;

-- Save both runtime ciphertexts and accept the already source-validated slot
-- in one database transaction. Existing credentials are never replaced.
CREATE OR REPLACE FUNCTION public.lab_arena_accept_submission_with_credentials(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_credentials JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_accept_submission_with_credentials$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_keys TEXT[];
  v_existing INTEGER;
  v_openrouter TEXT;
  v_deepline TEXT;
BEGIN
  IF pg_catalog.jsonb_typeof(p_credentials) IS DISTINCT FROM 'object' THEN
    RAISE EXCEPTION 'lab_arena_credentials_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT COALESCE(pg_catalog.array_agg(key ORDER BY key), ARRAY[]::TEXT[])
  INTO v_keys
  FROM pg_catalog.jsonb_object_keys(p_credentials) AS key;
  v_openrouter := p_credentials ->> 'openrouter';
  v_deepline := p_credentials ->> 'deepline';
  IF v_keys <> ARRAY['deepline', 'openrouter']::TEXT[]
     OR pg_catalog.char_length(COALESCE(v_openrouter, '')) NOT BETWEEN 4 AND 10924
     OR pg_catalog.char_length(COALESCE(v_deepline, '')) NOT BETWEEN 4 AND 10924
     OR v_openrouter !~ '^[A-Za-z0-9+/]+={0,2}$'
     OR v_deepline !~ '^[A-Za-z0-9+/]+={0,2}$'
     OR pg_catalog.char_length(v_openrouter) % 4 <> 0
     OR pg_catalog.char_length(v_deepline) % 4 <> 0 THEN
    RAISE EXCEPTION 'lab_arena_credentials_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open'
     OR COALESCE(
       pg_catalog.clock_timestamp() <
         (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ,
       TRUE
     )
     OR COALESCE(
       pg_catalog.clock_timestamp() >=
         (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ,
       TRUE
     ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'window_closed', 'round_status', v_round.status
    );
  END IF;

  SELECT * INTO v_submission
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id AND round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND OR v_submission.miner_hotkey IS DISTINCT FROM p_miner_hotkey THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_submission.is_king
     AND v_submission.submission_id =
       'baseline-' || pg_catalog.regexp_replace(v_submission.round_id, '^arena-', '')
     AND v_submission.miner_hotkey =
       v_round.configuration_doc ->> 'baseline_hotkey' THEN
    RAISE EXCEPTION 'lab_arena_baseline_credentials_forbidden' USING ERRCODE = '42501';
  END IF;
  IF v_submission.status IN ('accepted', 'frozen') THEN
    SELECT COUNT(*) INTO v_existing
    FROM public.lab_arena_submission_credentials
    WHERE submission_id = p_submission_id
      AND miner_hotkey = p_miner_hotkey
      AND provider IN ('openrouter', 'deepline');
    IF v_existing = 2 THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'submission_status', v_submission.status
      );
    END IF;
    RAISE EXCEPTION 'lab_arena_submission_credentials_missing' USING ERRCODE = '23514';
  END IF;
  IF v_submission.status <> 'uploading'
     OR v_submission.source_ref IS NULL
     OR v_submission.source_size_bytes IS NULL THEN
    RAISE EXCEPTION 'lab_arena_submission_not_uploading' USING ERRCODE = '22023';
  END IF;

  INSERT INTO public.lab_arena_submission_credentials (
    submission_id, miner_hotkey, provider, ciphertext
  ) VALUES
    (p_submission_id, p_miner_hotkey, 'openrouter', pg_catalog.decode(v_openrouter, 'base64')),
    (p_submission_id, p_miner_hotkey, 'deepline', pg_catalog.decode(v_deepline, 'base64'));
  UPDATE public.lab_arena_submissions
  SET status = 'accepted', updated_at = pg_catalog.clock_timestamp()
  WHERE submission_id = p_submission_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'submission_status', 'accepted'
  );
END;
$lab_arena_accept_submission_with_credentials$;
ALTER FUNCTION public.lab_arena_accept_submission_with_credentials(TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;

-- Read one ciphertext only for its exact accepted/frozen submission owner.
-- The management key has no stored row and therefore cannot cross this API.
CREATE OR REPLACE FUNCTION public.lab_arena_get_submission_credential(
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_provider TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_get_submission_credential$
DECLARE
  v_credential public.lab_arena_submission_credentials;
BEGIN
  IF p_provider NOT IN ('openrouter', 'deepline') THEN
    RAISE EXCEPTION 'lab_arena_credential_provider_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT credentials.* INTO v_credential
  FROM public.lab_arena_submission_credentials AS credentials
  JOIN public.lab_arena_submissions AS submissions
    ON submissions.submission_id = credentials.submission_id
  WHERE credentials.submission_id = p_submission_id
    AND credentials.miner_hotkey = p_miner_hotkey
    AND credentials.provider = p_provider
    AND submissions.miner_hotkey = p_miner_hotkey
    AND submissions.status IN ('accepted', 'frozen');
  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status', 'missing');
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'available',
    'submission_id', v_credential.submission_id,
    'miner_hotkey', v_credential.miner_hotkey,
    'provider', v_credential.provider,
    'ciphertext_b64', pg_catalog.replace(
      pg_catalog.encode(v_credential.ciphertext, 'base64'), E'\n', ''
    )
  );
END;
$lab_arena_get_submission_credential$;
ALTER FUNCTION public.lab_arena_get_submission_credential(TEXT, TEXT, TEXT)
  OWNER TO lab_arena_owner;

ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_runs_terminal_cause_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_terminal_cause_check
  CHECK (terminal_cause IS NULL OR terminal_cause IN (
    'accepted', 'model_timeout', 'invalid_output', 'budget_exhausted',
    'credential_error', 'model_error', 'lease_expired', 'worker_lost',
    'result_rejected', 'provider_error', 'stage_closed', 'judge_error',
    'judge_timeout'
  ));

ALTER TABLE public.lab_arena_ledger
  DROP CONSTRAINT IF EXISTS lab_arena_ledger_funding_source_check;
ALTER TABLE public.lab_arena_ledger
  ADD CONSTRAINT lab_arena_ledger_funding_source_check
  CHECK (funding_source IS NULL OR funding_source IN ('host', 'miner_key'));

-- Reserve one call after proving that only the configured daily baseline uses
-- host funds and every miner submission uses its own runtime key.
CREATE OR REPLACE FUNCTION public.lab_arena_reserve_call(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_call_identity TEXT,
  p_operation_id TEXT,
  p_provider TEXT,
  p_funding_source TEXT,
  p_amount_microusd BIGINT,
  p_call_doc JSONB,
  p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_reserve_call$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_head public.lab_arena_ledger;
  v_submission public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_quota INTEGER;
  v_stage_quota BIGINT;
  v_consumed BIGINT;
  v_money_cap BIGINT;
  v_spent BIGINT;
  v_reason TEXT := NULL;
  v_expires TIMESTAMPTZ;
BEGIN
  IF COALESCE(p_call_identity, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_operation_id, '') !~ '^[a-z0-9_.]{1,64}$'
     OR p_provider NOT IN ('scrapingdog', 'deepline', 'openrouter')
     OR p_funding_source NOT IN ('host', 'miner_key')
     OR COALESCE(p_amount_microusd, -1) < 0
     OR pg_catalog.jsonb_typeof(p_call_doc) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_call_doc::TEXT) > 65536
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_reserve_input_invalid' USING ERRCODE = '22023';
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds WHERE round_id = v_run.round_id;
  SELECT * INTO v_submission
  FROM public.lab_arena_submissions WHERE submission_id = v_run.submission_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  v_is_baseline :=
    v_submission.is_king
    AND v_run.submission_id =
      'baseline-' || pg_catalog.regexp_replace(v_run.round_id, '^arena-', '')
    AND v_run.miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey';
  IF (v_is_baseline AND p_funding_source <> 'host')
     OR (NOT v_is_baseline AND p_funding_source <> 'miner_key') THEN
    RAISE EXCEPTION 'lab_arena_funding_source_mismatch' USING ERRCODE = '42501';
  END IF;

  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NOT NULL THEN
    IF v_head.run_id <> p_run_id THEN
      RAISE EXCEPTION 'lab_arena_call_identity_foreign' USING ERRCODE = '23505';
    END IF;
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  v_quota := CASE v_run.kind
    WHEN 'score' THEN ((v_round.configuration_doc -> 'scoring_call_quotas') ->> p_provider)::INTEGER
    ELSE ((v_round.configuration_doc -> 'call_quotas') ->> p_provider)::INTEGER END;
  IF v_quota IS NULL OR v_quota < 1 THEN
    RAISE EXCEPTION 'lab_arena_quota_missing' USING ERRCODE = '22023';
  END IF;
  v_stage_quota := v_quota::BIGINT
    * (CASE v_run.stage
        WHEN 1 THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::BIGINT
        ELSE (v_round.configuration_doc ->> 'stage_2_icp_count')::BIGINT
      END)
    * (v_round.configuration_doc ->> 'max_attempts_per_assignment')::BIGINT;

  v_consumed := public.lab_arena__run_consumed(p_run_id, p_provider);
  IF v_consumed >= v_quota THEN
    v_reason := 'per_icp_quota';
  END IF;
  IF v_reason IS NULL THEN
    SELECT * INTO v_submission
    FROM public.lab_arena_submissions
    WHERE submission_id = v_run.submission_id
    FOR NO KEY UPDATE;
    v_consumed := public.lab_arena__submission_stage_consumed(
      v_run.submission_id, v_run.stage, p_provider, v_run.kind
    );
    IF v_consumed >= v_stage_quota THEN
      v_reason := 'stage_quota';
    END IF;
    IF v_reason IS NULL AND p_provider = 'openrouter' THEN
      v_money_cap := CASE v_run.kind
        WHEN 'score' THEN (v_round.configuration_doc ->> 'scoring_cap_microusd')::BIGINT
        ELSE (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT END;
      IF v_money_cap IS NULL OR v_money_cap < 1 THEN
        RAISE EXCEPTION 'lab_arena_money_cap_missing' USING ERRCODE = '22023';
      END IF;
      v_spent := public.lab_arena__submission_kind_spend(
        v_run.submission_id, v_run.kind
      );
      IF v_spent > v_money_cap - p_amount_microusd THEN
        v_reason := 'money_cap';
      END IF;
    END IF;
  END IF;
  v_expires := pg_catalog.clock_timestamp()
    + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  IF v_reason IS NOT NULL THEN
    INSERT INTO public.lab_arena_ledger (
      entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
      call_identity, provider, operation_id, funding_source,
      amount_microusd, entry_doc
    ) VALUES (
      'refusal', v_run.miner_hotkey, v_run.round_id, v_run.submission_id,
      p_run_id, v_run.stage, p_call_identity, p_provider, p_operation_id,
      p_funding_source, 0,
      pg_catalog.jsonb_build_object(
        'reason', v_reason, 'requested_microusd', p_amount_microusd,
        'call', p_call_doc
      )
    );
    UPDATE public.lab_arena_runs
    SET lease_expires_at = v_expires WHERE run_id = p_run_id;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'refused', 'idempotent', FALSE, 'reason', v_reason,
      'call_identity', p_call_identity, 'lease_expires_at', v_expires
    );
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc
  ) VALUES (
    'reservation', v_run.miner_hotkey, v_run.round_id, v_run.submission_id,
    p_run_id, v_run.stage, p_call_identity, p_provider, p_operation_id,
    p_funding_source, p_amount_microusd, p_call_doc
  );
  UPDATE public.lab_arena_runs
  SET lease_expires_at = v_expires WHERE run_id = p_run_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'reserved', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'amount_microusd', p_amount_microusd,
    'lease_expires_at', v_expires
  );
END;
$lab_arena_reserve_call$;
ALTER FUNCTION public.lab_arena_reserve_call(
  TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
) OWNER TO lab_arena_owner;

-- Credential failures are final miner outcomes. They never consume a second
-- validator attempt. The same cause is valid for a scoring assignment, where
-- the existing scorer close path makes only that challenger ineligible.
CREATE OR REPLACE FUNCTION public.lab_arena_complete_attempt(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_result JSONB,
  p_terminal_cause TEXT,
  p_output_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_complete_attempt$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_existing public.lab_arena_runs;
  v_open INTEGER;
  v_status TEXT;
BEGIN
  IF pg_catalog.jsonb_typeof(p_result) IS DISTINCT FROM 'object'
     OR (p_result ->> 'terminal_status') IS DISTINCT FROM p_terminal_cause
     OR p_terminal_cause NOT IN (
       'accepted', 'model_timeout', 'invalid_output', 'budget_exhausted',
       'credential_error', 'model_error', 'provider_error', 'judge_error',
       'judge_timeout'
     )
     OR (p_terminal_cause = 'accepted'
         AND pg_catalog.char_length(COALESCE(p_output_ref, '')) NOT BETWEEN 1 AND 1024)
     OR (p_terminal_cause <> 'accepted' AND COALESCE(p_output_ref, '') <> '') THEN
    RAISE EXCEPTION 'lab_arena_complete_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_existing
  FROM public.lab_arena_runs WHERE run_id = p_run_id;
  IF FOUND AND v_existing.status IN ('accepted', 'failed') THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', v_existing.status, 'idempotent', TRUE,
      'run_id', p_run_id, 'attempt', v_existing.attempt
    );
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  IF (v_run.kind = 'execute' AND p_terminal_cause IN ('judge_error', 'judge_timeout'))
     OR (v_run.kind = 'score' AND p_terminal_cause NOT IN (
       'accepted', 'credential_error', 'judge_error', 'judge_timeout'
     )) THEN
    RAISE EXCEPTION 'lab_arena_complete_cause_kind_mismatch' USING ERRCODE = '22023';
  END IF;
  SELECT COUNT(*) INTO v_open FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.run_id = p_run_id AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS heads WHERE heads.entry_kind IN ('reservation', 'dispatch');
  IF v_open > 0 THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'accounting_open', 'open_calls', v_open
    );
  END IF;
  v_status := CASE
    WHEN p_terminal_cause = 'accepted' THEN 'accepted' ELSE 'failed' END;
  UPDATE public.lab_arena_runs
  SET status = v_status,
      result_doc = p_result,
      terminal_cause = p_terminal_cause,
      output_ref = NULLIF(p_output_ref, '')
  WHERE run_id = p_run_id;
  IF v_status = 'failed'
     AND p_terminal_cause NOT IN ('budget_exhausted', 'credential_error')
     AND v_run.attempt < 2 THEN
    SELECT * INTO v_round
    FROM public.lab_arena_rounds WHERE round_id = v_run.round_id;
    IF v_run.stage_generation = v_round.stage_generation THEN
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
        icp_position, attempt, status, lease_generation, stage_generation,
        kind, scored_run_id, previous_runner_hotkey
      ) VALUES (
        v_run.assignment_id || ':' || (v_run.attempt + 1)::TEXT,
        v_run.assignment_id, v_run.round_id, v_run.submission_id,
        v_run.miner_hotkey, v_run.stage, v_run.icp_position,
        v_run.attempt + 1, 'pending', v_run.lease_generation,
        v_round.stage_generation, v_run.kind, v_run.scored_run_id,
        v_run.runner_hotkey
      );
      RETURN pg_catalog.jsonb_build_object(
        'status', v_status, 'idempotent', FALSE, 'run_id', p_run_id,
        'attempt', v_run.attempt,
        'confirmation_attempt', v_run.attempt + 1
      );
    END IF;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', v_status, 'idempotent', FALSE,
    'run_id', p_run_id, 'attempt', v_run.attempt
  );
END;
$lab_arena_complete_attempt$;
ALTER FUNCTION public.lab_arena_complete_attempt(TEXT, TEXT, JSONB, TEXT, TEXT)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_close_stage(
  p_round_id TEXT,
  p_stage SMALLINT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_close_stage$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_generation BIGINT;
  v_incomplete INTEGER;
  v_next TEXT;
BEGIN
  IF p_stage NOT IN (1, 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> ('stage' || p_stage::TEXT) THEN
    IF (p_stage = 1 AND v_round.status IN (
          'stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
          'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged',
          'scored', 'published'
        ))
       OR (p_stage = 2 AND v_round.status IN (
          'stage2_closed', 'stage2_scoring', 'stage2_judged',
          'scored', 'published'
        )) THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND stage = p_stage AND kind = 'execute'
      AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    IF v_run.status = 'leased' THEN
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object(
          'closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status
        )
    WHERE run_id = v_run.run_id;
  END LOOP;
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT runs.assignment_id
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id
      AND runs.stage = p_stage
      AND runs.kind = 'execute'
    GROUP BY runs.assignment_id
    HAVING bool_and(runs.status <> 'accepted')
       AND NOT bool_or(COALESCE(runs.terminal_cause, '') IN (
         'model_timeout', 'invalid_output', 'budget_exhausted',
         'credential_error', 'model_error'
       ))
  ) AS incomplete;
  IF v_incomplete > 0 THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled',
        status_generation = status_generation + 1,
        stage_generation = v_generation,
        cancel_reason =
          'capacity:stage' || p_stage::TEXT || ':' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage' || p_stage::TEXT || '_closed';
    UPDATE public.lab_arena_rounds
    SET status = v_next,
        status_generation = status_generation + 1,
        stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE WHEN v_next = 'cancelled' THEN 'cancelled' ELSE 'closed' END,
    'round_status', v_next,
    'incomplete_assignments', v_incomplete,
    'stage_generation', v_generation
  );
END;
$lab_arena_close_stage$;
ALTER FUNCTION public.lab_arena_close_stage(TEXT, SMALLINT)
  OWNER TO lab_arena_owner;

DO $lab_arena_credential_function_acl$
DECLARE
  signature TEXT;
  role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_accept_submission_with_credentials(TEXT, TEXT, TEXT, JSONB)',
    'public.lab_arena_get_submission_credential(TEXT, TEXT, TEXT)'
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
$lab_arena_credential_function_acl$;

CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_schema_version$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.schema_version.v1',
    'version', 185
  );
$lab_arena_schema_version$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
