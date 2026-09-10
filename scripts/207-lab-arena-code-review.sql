-- Persist one bounded, miner-funded full-source review before Arena scoring.
-- The provider call uses the existing append-only ledger. It does not create
-- a run kind, receipt graph, manifest, or source-code copy in PostgreSQL.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_207_requires_206$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_submission_costs(text)'
     ) IS NULL
     OR NOT EXISTS (
       SELECT 1
       FROM pg_catalog.pg_constraint AS constraint_row
       WHERE constraint_row.conrelid =
               'public.lab_arena_ledger'::pg_catalog.regclass
         AND constraint_row.conname =
               'lab_arena_ledger_funding_source_check'
         AND pg_catalog.pg_get_constraintdef(constraint_row.oid)
               LIKE '%miner_key%'
     ) THEN
    RAISE EXCEPTION
      'apply 206-lab-arena-combined-provider-budget.sql first';
  END IF;
END;
$lab_arena_207_requires_206$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_submissions
  ADD COLUMN IF NOT EXISTS code_review_status TEXT NOT NULL DEFAULT 'pending',
  ADD COLUMN IF NOT EXISTS code_review_doc JSONB,
  ADD COLUMN IF NOT EXISTS code_review_claim TEXT,
  ADD COLUMN IF NOT EXISTS code_review_started_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS code_review_expires_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS code_review_attempts SMALLINT NOT NULL DEFAULT 0;

ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_code_review_status_check;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_code_review_status_check
  CHECK (code_review_status IN
    ('pending', 'reviewing', 'passed', 'rejected', 'error'));

ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_code_review_doc_check;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_code_review_doc_check
  CHECK (
    code_review_doc IS NULL
    OR (
      pg_catalog.jsonb_typeof(code_review_doc) = 'object'
      AND pg_catalog.octet_length(code_review_doc::TEXT) <= 32768
    )
  );

ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_code_review_claim_check;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_code_review_claim_check
  CHECK (
    code_review_claim IS NULL
    OR code_review_claim ~ '^sha256:[0-9a-f]{64}$'
  );

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
      AND code_review_attempts BETWEEN 1 AND 3
      AND code_review_doc IS NULL
      AND code_review_claim IS NOT NULL
      AND code_review_started_at IS NOT NULL
      AND code_review_expires_at > code_review_started_at
    )
    OR (
      code_review_status IN ('passed', 'rejected', 'error')
      AND code_review_attempts BETWEEN 1 AND 3
      AND code_review_doc IS NOT NULL
      AND code_review_claim IS NOT NULL
      AND code_review_started_at IS NOT NULL
      AND code_review_expires_at IS NULL
    )
  );

-- Frozen source and identity fields remain immutable. A current-round review
-- can still finish after a freeze raced its provider request. Only the six
-- review bookkeeping fields and updated_at can change on such a row.
CREATE OR REPLACE FUNCTION public.lab_arena_submissions_frozen_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_submissions_frozen$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'lab_arena_submissions rows are never deleted'
      USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'frozen'
     AND (
       pg_catalog.to_jsonb(NEW) - ARRAY[
         'code_review_status', 'code_review_doc', 'code_review_claim',
         'code_review_started_at', 'code_review_expires_at',
         'code_review_attempts', 'updated_at'
       ]::TEXT[]
     ) IS DISTINCT FROM (
       pg_catalog.to_jsonb(OLD) - ARRAY[
         'code_review_status', 'code_review_doc', 'code_review_claim',
         'code_review_started_at', 'code_review_expires_at',
         'code_review_attempts', 'updated_at'
       ]::TEXT[]
     ) THEN
    RAISE EXCEPTION 'frozen submission is immutable' USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' THEN
    RAISE EXCEPTION 'rejected submission cannot be reopened'
      USING ERRCODE = '42501';
  END IF;
  NEW.updated_at := pg_catalog.clock_timestamp();
  RETURN NEW;
END;
$lab_arena_submissions_frozen$;
ALTER FUNCTION public.lab_arena_submissions_frozen_v1()
  OWNER TO lab_arena_owner;

-- All newly frozen miner submissions must have a completed passing review.
-- The one canonical organizer baseline is the only exemption.
CREATE OR REPLACE FUNCTION public.lab_arena_code_review_freeze_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_code_review_freeze_guard$
DECLARE
  v_round public.lab_arena_rounds;
  v_is_baseline BOOLEAN;
BEGIN
  IF OLD.status = 'accepted' AND NEW.status = 'frozen' THEN
    SELECT * INTO v_round
    FROM public.lab_arena_rounds
    WHERE round_id = NEW.round_id;
    v_is_baseline :=
      NEW.is_king
      AND NEW.submission_id =
        'baseline-' || pg_catalog.regexp_replace(NEW.round_id, '^arena-', '')
      AND NEW.miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey';
    IF NOT COALESCE(v_is_baseline, FALSE)
       AND NEW.code_review_status <> 'passed' THEN
      RAISE EXCEPTION 'lab_arena_code_review_required'
        USING ERRCODE = '23514';
    END IF;
  END IF;
  RETURN NEW;
END;
$lab_arena_code_review_freeze_guard$;
ALTER FUNCTION public.lab_arena_code_review_freeze_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_code_review_freeze_guard_v1()
  FROM PUBLIC, lab_arena_service;

DROP TRIGGER IF EXISTS lab_arena_code_review_freeze_guard
  ON public.lab_arena_submissions;
CREATE TRIGGER lab_arena_code_review_freeze_guard
  BEFORE UPDATE ON public.lab_arena_submissions
  FOR EACH ROW EXECUTE FUNCTION
    public.lab_arena_code_review_freeze_guard_v1();

-- Lease eligibility is enforced in the same row-locking statement that picks
-- the run. This prevents both execute and score work from being stranded in a
-- leased state when a caller sees an incomplete review after claiming it.
-- This is the migration-190 body with only the review predicate added.
CREATE OR REPLACE FUNCTION public.lab_arena_claim_assignment(
  p_round_id TEXT,
  p_runner_hotkey TEXT,
  p_declared_parallelism INTEGER,
  p_slot_ceiling INTEGER,
  p_excluded_miner_hotkeys TEXT[],
  p_request_id TEXT,
  p_request_hash TEXT,
  p_lease_token_hash TEXT,
  p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_claim_assignment$
DECLARE
  v_round public.lab_arena_rounds;
  v_existing public.lab_arena_runs;
  v_run public.lab_arena_runs;
  v_submission public.lab_arena_submissions;
  v_stage SMALLINT;
  v_limit INTEGER;
  v_active INTEGER;
  v_expires TIMESTAMPTZ;
  v_response JSONB;
  v_claims_paused BOOLEAN;
BEGIN
  IF COALESCE(p_runner_hotkey, '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
     OR COALESCE(p_declared_parallelism, 0) < 1 OR COALESCE(p_slot_ceiling, 0) < 1
     OR COALESCE(p_request_id, '') !~ '^[0-9a-f]{32}$'
     OR COALESCE(p_request_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_lease_token_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_claim_input_invalid' USING ERRCODE = '22023';
  END IF;
  -- This lock is taken before every new claim and held through its lease write.
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'lab_arena.runner:' || p_round_id || ':' || p_runner_hotkey,
      0
    )
  );
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena.claim:' || p_request_id, 0)
  );
  SELECT * INTO v_existing FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND claim_request_id = p_request_id;
  IF FOUND THEN
    IF v_existing.claim_request_hash = p_request_hash THEN
      RETURN v_existing.claim_response;
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'request_id_reused');
  END IF;
  SELECT operator_paused OR guard_commitment <> '' INTO v_claims_paused
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF COALESCE(v_claims_paused, TRUE) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'paused');
  END IF;
  IF v_round.status NOT IN ('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring') THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stage_closed', 'round_status', v_round.status);
  END IF;
  v_stage := CASE WHEN v_round.status IN ('stage1', 'stage1_scoring') THEN 1 ELSE 2 END;
  IF NOT (v_round.configuration_doc -> 'runner_hotkeys' ? p_runner_hotkey) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_allowlisted');
  END IF;
  v_limit := LEAST(p_declared_parallelism, p_slot_ceiling);
  SELECT COUNT(*) INTO v_active FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND runner_hotkey = p_runner_hotkey AND status = 'leased'
    AND lease_expires_at > pg_catalog.clock_timestamp()
    AND stage_generation = v_round.stage_generation;
  IF v_active >= v_limit THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'no_free_slot', 'active_leases', v_active, 'slot_limit', v_limit
    );
  END IF;
  SELECT * INTO v_run FROM public.lab_arena_runs AS runs
  WHERE runs.round_id = p_round_id AND runs.stage = v_stage AND runs.status = 'pending'
    AND runs.stage_generation = v_round.stage_generation
    -- A configured organizer baseline remains claimable when its hotkey shares
    -- a coldkey with a runner. Self-execution exclusion still applies to every
    -- miner submission.
    AND (
      runs.miner_hotkey <> ALL (COALESCE(p_excluded_miner_hotkeys, ARRAY[]::TEXT[]))
      OR EXISTS (
        SELECT 1
        FROM public.lab_arena_submissions AS baseline_submission
        WHERE baseline_submission.submission_id = runs.submission_id
          AND baseline_submission.round_id = runs.round_id
          AND baseline_submission.status = 'frozen'
          AND baseline_submission.is_king
          AND baseline_submission.miner_hotkey =
              (v_round.configuration_doc ->> 'baseline_hotkey')
      )
    )
    -- Every miner run is filtered before its row is locked and leased. The
    -- canonical organizer baseline remains the only review exemption.
    AND EXISTS (
      SELECT 1
      FROM public.lab_arena_submissions AS reviewed_submission
      WHERE reviewed_submission.submission_id = runs.submission_id
        AND reviewed_submission.round_id = runs.round_id
        AND (
          reviewed_submission.code_review_status = 'passed'
          OR (
            reviewed_submission.status = 'frozen'
            AND reviewed_submission.is_king
            AND reviewed_submission.submission_id =
              'baseline-' || pg_catalog.regexp_replace(
                reviewed_submission.round_id, '^arena-', ''
              )
            AND reviewed_submission.miner_hotkey =
              (v_round.configuration_doc ->> 'baseline_hotkey')
          )
        )
    )
    -- No execute lease can use a retired image-only submission.
    AND (
      runs.kind <> 'execute'
      OR EXISTS (
        SELECT 1
        FROM public.lab_arena_submissions AS source_submission
        WHERE source_submission.submission_id = runs.submission_id
          AND source_submission.round_id = runs.round_id
          AND source_submission.status = 'frozen'
          AND source_submission.source_ref IS NOT NULL
          AND source_submission.source_size_bytes IS NOT NULL
      )
    )
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs AS others
           WHERE others.round_id = p_round_id AND others.runner_hotkey IS NOT NULL
             AND others.runner_hotkey <> p_runner_hotkey
             AND others.status = 'leased'
             AND others.lease_expires_at > pg_catalog.clock_timestamp()
             AND others.stage_generation = v_round.stage_generation))
  ORDER BY runs.icp_position, runs.created_at, runs.assignment_id
  FOR UPDATE SKIP LOCKED
  LIMIT 1;
  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status', 'no_pending');
  END IF;
  SELECT * INTO v_submission FROM public.lab_arena_submissions
  WHERE submission_id = v_run.submission_id;
  IF v_run.kind = 'execute'
     AND (v_submission.source_ref IS NULL
          OR v_submission.source_size_bytes IS NULL) THEN
    RAISE EXCEPTION 'lab_arena_source_missing' USING ERRCODE = '23502';
  END IF;
  v_expires := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  v_response := pg_catalog.jsonb_build_object(
    'status', 'leased',
    'request_id', p_request_id,
    'run_id', v_run.run_id,
    'assignment_id', v_run.assignment_id,
    'submission_id', v_run.submission_id,
    'miner_hotkey', v_run.miner_hotkey,
    'source_ref', CASE WHEN v_run.kind = 'execute' THEN v_submission.source_ref END,
    'source_size_bytes', CASE WHEN v_run.kind = 'execute' THEN v_submission.source_size_bytes END,
    'stage', v_run.stage,
    'icp_position', v_run.icp_position,
    'attempt', v_run.attempt,
    'kind', v_run.kind,
    'scored_run_id', v_run.scored_run_id,
    'lease_generation', v_run.lease_generation + 1,
    'stage_generation', v_round.stage_generation,
    'lease_expires_at', v_expires
  );
  UPDATE public.lab_arena_runs
  SET status = 'leased', runner_hotkey = p_runner_hotkey, lease_token_hash = p_lease_token_hash,
      lease_generation = lease_generation + 1, lease_expires_at = v_expires,
      claim_request_id = p_request_id, claim_request_hash = p_request_hash, claim_response = v_response
  WHERE run_id = v_run.run_id;
  RETURN v_response;
END;
$lab_arena_claim_assignment$;
ALTER FUNCTION public.lab_arena_claim_assignment(
  TEXT, TEXT, INTEGER, INTEGER, TEXT[], TEXT, TEXT, TEXT, INTEGER
) OWNER TO lab_arena_owner;

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
    'max_attempts', 3
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
  v_now TIMESTAMPTZ := pg_catalog.clock_timestamp();
  v_retry_at TIMESTAMPTZ;
  v_expires TIMESTAMPTZ;
  v_attempt SMALLINT;
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
        'status', 'busy',
        'code_review_status', 'reviewing',
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
          'reserved_microusd', v_reservation.amount_microusd
        )
      );
    ELSIF v_head.entry_kind NOT IN ('settlement', 'uncertain') THEN
      RAISE EXCEPTION 'lab_arena_code_review_ledger_invalid'
        USING ERRCODE = '55000';
    END IF;

    IF v_submission.code_review_attempts >= 3 THEN
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
      SET code_review_status = 'error',
          code_review_doc = pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.lab_arena.code_review.result.v1',
            'verdict', 'error',
            'reason_code', 'review_attempts_exhausted',
            'cost_microusd', v_reservation.amount_microusd,
            'cost_status', 'uncertain',
            'review_cost_microusd', v_review_cost
          ),
          code_review_started_at = v_now,
          code_review_expires_at = NULL
      WHERE submission_id = p_submission_id;
      RETURN pg_catalog.jsonb_build_object(
        'status', 'exhausted', 'code_review_status', 'error',
        'attempt', v_submission.code_review_attempts
      );
    END IF;
  ELSIF v_submission.code_review_status = 'error' THEN
    IF v_submission.code_review_attempts >= 3 THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'exhausted', 'code_review_status', 'error',
        'attempt', v_submission.code_review_attempts
      );
    END IF;
    v_retry_at := v_submission.code_review_started_at
      + pg_catalog.make_interval(secs => 60);
    IF v_retry_at > v_now THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'backoff', 'code_review_status', 'error',
        'attempt', v_submission.code_review_attempts,
        'retry_after', v_retry_at
      );
    END IF;
  ELSIF v_submission.code_review_status <> 'pending' THEN
    RAISE EXCEPTION 'lab_arena_code_review_state_invalid'
      USING ERRCODE = '55000';
  END IF;

  v_attempt := v_submission.code_review_attempts + 1;
  v_expires := v_now + pg_catalog.make_interval(secs => 600);
  v_call_identity := 'sha256:' || pg_catalog.encode(
    extensions.digest(
      pg_catalog.convert_to(
        'lab_arena.code_review:' || p_submission_id || ':' ||
          v_attempt::TEXT,
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
  v_now TIMESTAMPTZ := pg_catalog.clock_timestamp();
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

  -- Bind the terminal document to the source metadata and model reserved by
  -- begin. A passing result also requires a complete, accounted provider call.
  IF p_review_doc ->> 'model' IS DISTINCT FROM
       v_reservation.entry_doc ->> 'review_model'
     OR p_review_doc -> 'file_count' IS DISTINCT FROM
       v_reservation.entry_doc -> 'file_count'
     OR p_review_doc -> 'source_bytes' IS DISTINCT FROM
       v_reservation.entry_doc -> 'source_bytes'
     OR (
       p_status = 'passed'
       AND (
         p_actual_microusd IS NULL
         OR p_review_doc -> 'passed' IS DISTINCT FROM 'true'::JSONB
         OR p_review_doc ->> 'verdict' IS DISTINCT FROM 'pass'
         OR (v_reservation.entry_doc ->> 'file_count')::INTEGER <= 0
         OR (v_reservation.entry_doc ->> 'source_bytes')::BIGINT <= 0
       )
     )
     OR (
       p_status = 'rejected'
       AND (
         p_review_doc -> 'passed' IS DISTINCT FROM 'false'::JSONB
         OR p_review_doc ->> 'verdict' IS DISTINCT FROM 'reject'
         OR (v_reservation.entry_doc ->> 'file_count')::INTEGER <= 0
         OR (v_reservation.entry_doc ->> 'source_bytes')::BIGINT <= 0
       )
     ) THEN
    RAISE EXCEPTION 'lab_arena_code_review_input_invalid'
      USING ERRCODE = '22023';
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
        'review_status', p_status,
        'reserved_microusd', v_reservation.amount_microusd
      )
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
        'released_microusd',
          v_reservation.amount_microusd - p_actual_microusd,
        'variance_microusd',
          p_actual_microusd - v_reservation.amount_microusd,
        'review_status', p_status
      ),
      pg_catalog.jsonb_build_object('review_status', p_status)
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
  SET code_review_status = p_status,
      code_review_doc = p_review_doc || pg_catalog.jsonb_build_object(
        'cost_microusd', v_call_cost,
        'cost_status', v_ledger_status,
        'review_cost_microusd', v_review_cost
      ),
      code_review_started_at = CASE
        WHEN p_status = 'error' THEN v_now
        ELSE code_review_started_at
      END,
      code_review_expires_at = NULL
  WHERE submission_id = p_submission_id;

  RETURN pg_catalog.jsonb_build_object(
    'status', p_status,
    'code_review_status', p_status,
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

DO $lab_arena_code_review_acl$
DECLARE
  signature TEXT;
  role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_code_review_schema_v1()',
    'public.lab_arena_begin_submission_review(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, BIGINT)',
    'public.lab_arena_finish_submission_review(TEXT, TEXT, TEXT, TEXT, JSONB, BIGINT)'
  ] LOOP
    EXECUTE pg_catalog.format(
      'REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature
    );
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name
      ) THEN
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
$lab_arena_code_review_acl$;

COMMENT ON COLUMN public.lab_arena_submissions.code_review_doc IS
  'Bounded safe review result only; never submitted source, credentials, or raw provider payloads.';
COMMENT ON FUNCTION public.lab_arena_begin_submission_review(
  TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, BIGINT
) IS
  'Atomically claims one bounded miner-funded full-source review and appends its reservation and dispatch.';
COMMENT ON FUNCTION public.lab_arena_finish_submission_review(
  TEXT, TEXT, TEXT, TEXT, JSONB, BIGINT
) IS
  'Finishes the exact active review claim and appends settlement or conservative uncertain cost.';

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
