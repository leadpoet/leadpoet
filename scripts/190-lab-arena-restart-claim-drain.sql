-- 190-lab-arena-restart-claim-drain.sql
-- Pause new Arena leases while canonical gateway and validator restarts drain
-- every already-issued lease to a durable completion receipt.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- The first installation cannot assume that replacing a function stops an
-- already-running copy of its old body. Take relation locks which conflict
-- with every claim/write transaction before installing the permanent write
-- trigger. The trigger then protects a cached old body if it reaches its
-- pending-to-leased update after this transaction commits. A reapply preserves
-- the live control row and does not require an idle competition.
DO $lab_arena_restart_bootstrap$
DECLARE
  v_installed BOOLEAN := pg_catalog.to_regclass(
    'public.lab_arena_restart_claim_control'
  ) IS NOT NULL;
BEGIN
  IF NOT v_installed THEN
    -- NOWAIT makes this migration lose any race with live completion work.
    -- The migration transaction rolls back without making that work a
    -- deadlock victim; the repository migration runner may retry the whole
    -- idempotent migration after the writer finishes.
    LOCK TABLE public.lab_arena_rounds, public.lab_arena_runs
      IN ACCESS EXCLUSIVE MODE NOWAIT;
  END IF;
END;
$lab_arena_restart_bootstrap$;

CREATE TABLE IF NOT EXISTS public.lab_arena_restart_claim_control (
  singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
  operator_paused BOOLEAN NOT NULL DEFAULT FALSE,
  pause_reason TEXT NOT NULL DEFAULT '',
  actor_ref TEXT NOT NULL DEFAULT '',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  guard_commitment TEXT NOT NULL DEFAULT '',
  owner_commitment TEXT NOT NULL DEFAULT '',
  guard_generation BIGINT NOT NULL DEFAULT 0 CHECK (guard_generation >= 0),
  guard_expires_at TIMESTAMPTZ,
  candidate_commit TEXT NOT NULL DEFAULT '',
  restart_scope TEXT NOT NULL DEFAULT '',
  restart_phase TEXT NOT NULL DEFAULT '',
  captured_leases JSONB NOT NULL DEFAULT '[]'::JSONB,
  CHECK (pg_catalog.jsonb_typeof(captured_leases) = 'array'),
  CHECK (
    (guard_commitment = '' AND owner_commitment = ''
      AND guard_expires_at IS NULL AND candidate_commit = ''
      AND restart_scope = '' AND restart_phase = ''
      AND captured_leases = '[]'::JSONB)
    OR
    (guard_commitment ~ '^sha256:[0-9a-f]{64}$'
      AND owner_commitment ~ '^sha256:[0-9a-f]{64}$'
      AND guard_generation > 0 AND guard_expires_at IS NOT NULL
      AND candidate_commit ~ '^[0-9a-f]{40}$'
      AND restart_scope IN ('gateway', 'validator', 'all')
      AND restart_phase IN (
        'draining', 'gateway_destructive', 'gateway_ready',
        'validator_destructive', 'validator_ready'
      ))
  )
);
ALTER TABLE public.lab_arena_restart_claim_control OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_restart_claim_control ENABLE ROW LEVEL SECURITY;
INSERT INTO public.lab_arena_restart_claim_control(singleton)
VALUES (TRUE) ON CONFLICT (singleton) DO NOTHING;
REVOKE ALL ON TABLE public.lab_arena_restart_claim_control
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_restart_claim_gate_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_restart_claim_gate$
DECLARE
  v_paused BOOLEAN;
BEGIN
  IF OLD.status IS DISTINCT FROM 'leased' AND NEW.status = 'leased' THEN
    PERFORM pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended('lab-arena-claim-control', 0)
    );
    SELECT operator_paused OR guard_commitment <> '' INTO v_paused
    FROM public.lab_arena_restart_claim_control WHERE singleton;
    IF COALESCE(v_paused, TRUE) THEN
      RAISE EXCEPTION 'lab_arena_claims_paused' USING ERRCODE = '55000';
    END IF;
  END IF;
  RETURN NEW;
END;
$lab_arena_restart_claim_gate$;
ALTER FUNCTION public.lab_arena_restart_claim_gate_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_restart_claim_gate_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DROP TRIGGER IF EXISTS lab_arena_runs_restart_claim_gate
  ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_runs_restart_claim_gate
BEFORE UPDATE OF status ON public.lab_arena_runs
FOR EACH ROW EXECUTE FUNCTION public.lab_arena_restart_claim_gate_v1();

-- The exact migration-182 claim body is reinstalled below. Its only semantic
-- additions are the control lock/read and a paused result after idempotent
-- request replay has had an opportunity to return an already-issued lease.
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

CREATE OR REPLACE FUNCTION public.lab_arena__restart_drain_state_v1()
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_restart_drain_state$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_captured INTEGER := 0;
  v_accepted INTEGER := 0;
  v_reported INTEGER := 0;
  v_leased INTEGER := 0;
  v_lost INTEGER := 0;
  v_current_leased INTEGER := 0;
  v_pending_retries INTEGER := 0;
  v_snapshot_commitment TEXT;
  v_outcome_commitment TEXT;
BEGIN
  SELECT * INTO v_control
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_restart_control_missing';
  END IF;
  v_captured := pg_catalog.jsonb_array_length(v_control.captured_leases);
  WITH captured AS (
    SELECT item ->> 'run_id' AS run_id,
           (item ->> 'lease_generation')::BIGINT AS lease_generation
    FROM pg_catalog.jsonb_array_elements(v_control.captured_leases) AS item
  ), classified AS (
    SELECT captured.run_id,
      CASE
        WHEN runs.run_id IS NULL
          OR runs.lease_generation <> captured.lease_generation THEN 'lost'
        WHEN runs.status = 'leased' THEN 'leased'
        WHEN runs.status = 'accepted'
          AND runs.terminal_cause = 'accepted'
          AND runs.result_doc ->> 'terminal_status' = 'accepted'
          AND pg_catalog.char_length(COALESCE(runs.output_ref, '')) BETWEEN 1 AND 1024
          AND NOT EXISTS (
            SELECT 1 FROM (
              SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
              FROM public.lab_arena_ledger AS ledger
              WHERE ledger.run_id = runs.run_id
                AND ledger.call_identity IS NOT NULL
              ORDER BY ledger.call_identity, ledger.entry_id DESC
            ) heads WHERE heads.entry_kind IN ('reservation', 'dispatch')
          ) THEN 'accepted'
        WHEN runs.status = 'failed'
          AND runs.terminal_cause IN (
            'model_timeout', 'invalid_output', 'budget_exhausted',
            'credential_error', 'model_error', 'provider_error',
            'judge_error', 'judge_timeout'
          )
          AND runs.result_doc ->> 'terminal_status' = runs.terminal_cause
          AND pg_catalog.jsonb_typeof(runs.result_doc) = 'object'
          AND COALESCE(runs.output_ref, '') = ''
          AND (
            (runs.kind = 'execute'
              AND runs.terminal_cause NOT IN ('judge_error', 'judge_timeout'))
            OR
            (runs.kind = 'score'
              AND runs.terminal_cause IN (
                'credential_error', 'judge_error', 'judge_timeout'
              ))
          )
          AND NOT EXISTS (
            SELECT 1 FROM (
              SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
              FROM public.lab_arena_ledger AS ledger
              WHERE ledger.run_id = runs.run_id
                AND ledger.call_identity IS NOT NULL
              ORDER BY ledger.call_identity, ledger.entry_id DESC
            ) heads WHERE heads.entry_kind IN ('reservation', 'dispatch')
          ) THEN 'reported'
        ELSE 'lost'
      END AS outcome
    FROM captured LEFT JOIN public.lab_arena_runs AS runs USING (run_id)
  )
  SELECT
    COUNT(*) FILTER (WHERE outcome = 'accepted')::INTEGER,
    COUNT(*) FILTER (WHERE outcome = 'reported')::INTEGER,
    COUNT(*) FILTER (WHERE outcome = 'leased')::INTEGER,
    COUNT(*) FILTER (WHERE outcome = 'lost')::INTEGER
  INTO v_accepted, v_reported, v_leased, v_lost
  FROM classified;
  SELECT COUNT(*)::INTEGER INTO v_current_leased
  FROM public.lab_arena_runs WHERE status = 'leased';
  SELECT COUNT(*)::INTEGER INTO v_pending_retries
  FROM public.lab_arena_runs AS pending
  WHERE pending.status = 'pending' AND EXISTS (
    SELECT 1
    FROM pg_catalog.jsonb_array_elements(v_control.captured_leases) AS item
    JOIN public.lab_arena_runs AS captured
      ON captured.run_id = item ->> 'run_id'
    WHERE pending.assignment_id = captured.assignment_id
      AND pending.attempt > captured.attempt
  );
  v_snapshot_commitment := 'sha256:' || pg_catalog.encode(
    extensions.digest(
      pg_catalog.convert_to(v_control.captured_leases::TEXT, 'UTF8'), 'sha256'
    ), 'hex'
  );
  v_outcome_commitment := 'sha256:' || pg_catalog.encode(
    extensions.digest(pg_catalog.convert_to(
      v_snapshot_commitment || ':' || v_captured::TEXT || ':'
      || v_accepted::TEXT || ':' || v_reported::TEXT || ':'
      || v_leased::TEXT || ':' || v_lost::TEXT || ':'
      || v_current_leased::TEXT || ':' || v_pending_retries::TEXT,
      'UTF8'
    ), 'sha256'), 'hex'
  );
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.restart_drain_state.v1',
    'captured_count', v_captured,
    'accepted_receipt_count', v_accepted,
    'reported_terminal_receipt_count', v_reported,
    'still_leased_count', v_leased,
    'lost_or_mutated_count', v_lost,
    'current_leased_count', v_current_leased,
    'pending_retry_count', v_pending_retries,
    'snapshot_commitment', v_snapshot_commitment,
    'outcome_commitment', v_outcome_commitment,
    'preserved', v_current_leased = 0 AND v_leased = 0 AND v_lost = 0
      AND v_captured = v_accepted + v_reported
  );
END;
$lab_arena_restart_drain_state$;
ALTER FUNCTION public.lab_arena__restart_drain_state_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__restart_drain_state_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_restart_guard_state_v1()
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_restart_guard_state$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_drain JSONB;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton;
  v_drain := public.lab_arena__restart_drain_state_v1();
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.restart_guard_state.v1',
    'paused', v_control.operator_paused OR v_control.guard_commitment <> '',
    'operator_paused', v_control.operator_paused,
    'guard_present', v_control.guard_commitment <> '',
    'guard_active', v_control.guard_commitment <> ''
      AND v_control.guard_expires_at > pg_catalog.clock_timestamp(),
    'guard_commitment', v_control.guard_commitment,
    'owner_commitment', v_control.owner_commitment,
    'guard_generation', v_control.guard_generation,
    'guard_expires_at', v_control.guard_expires_at,
    'candidate_commit', v_control.candidate_commit,
    'restart_scope', v_control.restart_scope,
    'restart_phase', v_control.restart_phase,
    'drain', v_drain
  );
END;
$lab_arena_restart_guard_state$;
ALTER FUNCTION public.lab_arena_restart_guard_state_v1()
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_acquire_restart_guard_v1(
  p_guard_id TEXT,
  p_owner_id TEXT,
  p_expected_generation BIGINT,
  p_lease_seconds INTEGER,
  p_candidate_commit TEXT,
  p_restart_scope TEXT,
  p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_acquire_restart_guard$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_guard TEXT;
  v_owner TEXT;
  v_snapshot JSONB;
BEGIN
  IF COALESCE(p_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR COALESCE(p_owner_id, '') !~ '^lab_arena_restart_owner:[0-9a-f]{64}$'
    OR COALESCE(p_expected_generation, -1) < 0
    OR COALESCE(p_lease_seconds, 0) NOT BETWEEN 60 AND 14400
    OR COALESCE(p_candidate_commit, '') !~ '^[0-9a-f]{40}$'
    OR COALESCE(p_restart_scope, '') NOT IN ('gateway', 'validator', 'all')
    OR COALESCE(btrim(p_actor_ref), '') = '' THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_input_invalid';
  END IF;
  v_guard := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_owner := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_owner_id, 'UTF8'), 'sha256'), 'hex');
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF v_control.guard_generation <> p_expected_generation THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_generation_differs';
  END IF;
  IF v_control.guard_commitment <> '' THEN
    IF v_control.guard_commitment <> v_guard
      OR v_control.owner_commitment <> v_owner
      OR v_control.candidate_commit <> p_candidate_commit
      OR v_control.restart_scope <> p_restart_scope THEN
      RAISE EXCEPTION 'lab_arena_restart_guard_owned_by_another_invocation';
    END IF;
    UPDATE public.lab_arena_restart_claim_control
    SET guard_expires_at = GREATEST(
      guard_expires_at,
      pg_catalog.clock_timestamp()
        + pg_catalog.make_interval(secs => p_lease_seconds)
    ), actor_ref = left(p_actor_ref, 200), updated_at = pg_catalog.clock_timestamp()
    WHERE singleton;
  ELSE
    IF v_control.guard_generation = 9223372036854775807 THEN
      RAISE EXCEPTION 'lab_arena_restart_guard_generation_exhausted';
    END IF;
    SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
      'run_id', run_id, 'lease_generation', lease_generation
    ) ORDER BY run_id), '[]'::JSONB)
    INTO v_snapshot FROM public.lab_arena_runs WHERE status = 'leased';
    UPDATE public.lab_arena_restart_claim_control SET
      pause_reason = 'canonical_restart_guard',
      actor_ref = left(p_actor_ref, 200),
      updated_at = pg_catalog.clock_timestamp(),
      guard_commitment = v_guard,
      owner_commitment = v_owner,
      guard_generation = guard_generation + 1,
      guard_expires_at = pg_catalog.clock_timestamp()
        + pg_catalog.make_interval(secs => p_lease_seconds),
      candidate_commit = p_candidate_commit,
      restart_scope = p_restart_scope,
      restart_phase = 'draining',
      captured_leases = v_snapshot
    WHERE singleton;
  END IF;
  RETURN public.lab_arena_restart_guard_state_v1();
END;
$lab_arena_acquire_restart_guard$;
ALTER FUNCTION public.lab_arena_acquire_restart_guard_v1(
  TEXT, TEXT, BIGINT, INTEGER, TEXT, TEXT, TEXT
) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_retarget_restart_guard_v1(
  p_current_guard_id TEXT,
  p_owner_id TEXT,
  p_expected_generation BIGINT,
  p_new_guard_id TEXT,
  p_new_candidate_commit TEXT,
  p_restart_scope TEXT,
  p_lease_seconds INTEGER,
  p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_retarget_restart_guard$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_state JSONB;
  v_current_guard TEXT;
  v_new_guard TEXT;
  v_owner TEXT;
BEGIN
  IF COALESCE(p_current_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR COALESCE(p_owner_id, '') !~ '^lab_arena_restart_owner:[0-9a-f]{64}$'
    OR COALESCE(p_expected_generation, -1) <= 0
    OR COALESCE(p_new_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR p_new_guard_id IS NOT DISTINCT FROM p_current_guard_id
    OR COALESCE(p_new_candidate_commit, '') !~ '^[0-9a-f]{40}$'
    OR COALESCE(p_restart_scope, '') NOT IN ('gateway', 'validator', 'all')
    OR COALESCE(p_lease_seconds, 0) NOT BETWEEN 60 AND 14400
    OR COALESCE(btrim(p_actor_ref), '') = '' THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_input_invalid';
  END IF;
  v_current_guard := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_current_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_new_guard := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_new_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_owner := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_owner_id, 'UTF8'), 'sha256'), 'hex');
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF v_control.guard_commitment = ''
    OR v_control.guard_commitment IS DISTINCT FROM v_current_guard
    OR v_control.owner_commitment IS DISTINCT FROM v_owner
    OR v_control.guard_generation IS DISTINCT FROM p_expected_generation
    OR v_control.restart_scope IS DISTINCT FROM p_restart_scope
    OR v_control.restart_phase = 'draining'
    OR v_control.candidate_commit IS NOT DISTINCT FROM p_new_candidate_commit THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_retarget_forbidden';
  END IF;
  v_state := public.lab_arena__restart_drain_state_v1();
  IF (v_state ->> 'preserved')::BOOLEAN IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_restart_drain_not_preserved';
  END IF;
  IF v_control.guard_generation = 9223372036854775807 THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_generation_exhausted';
  END IF;
  UPDATE public.lab_arena_restart_claim_control SET
    guard_commitment = v_new_guard,
    guard_generation = guard_generation + 1,
    guard_expires_at = pg_catalog.clock_timestamp()
      + pg_catalog.make_interval(secs => p_lease_seconds),
    candidate_commit = p_new_candidate_commit,
    actor_ref = left(p_actor_ref, 200),
    updated_at = pg_catalog.clock_timestamp()
  WHERE singleton;
  RETURN public.lab_arena_restart_guard_state_v1();
END;
$lab_arena_retarget_restart_guard$;
ALTER FUNCTION public.lab_arena_retarget_restart_guard_v1(
  TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, INTEGER, TEXT
) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_restart_quiescence_v1(
  p_guard_id TEXT, p_owner_id TEXT, p_guard_generation BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_restart_quiescence$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_guard TEXT;
  v_owner TEXT;
  v_drain JSONB;
BEGIN
  IF COALESCE(p_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR COALESCE(p_owner_id, '') !~ '^lab_arena_restart_owner:[0-9a-f]{64}$'
    OR COALESCE(p_guard_generation, -1) <= 0 THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_input_invalid';
  END IF;
  v_guard := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_owner := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_owner_id, 'UTF8'), 'sha256'), 'hex');
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton;
  IF v_control.guard_commitment = ''
    OR v_control.guard_commitment IS DISTINCT FROM v_guard
    OR v_control.owner_commitment IS DISTINCT FROM v_owner
    OR v_control.guard_generation IS DISTINCT FROM p_guard_generation THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_owner_or_generation_differs';
  END IF;
  v_drain := public.lab_arena__restart_drain_state_v1();
  RETURN (v_drain - 'schema_version') || pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.restart_quiescence.v1',
    'guard_active', v_control.guard_expires_at > pg_catalog.clock_timestamp(),
    'guard_generation', v_control.guard_generation,
    'restart_scope', v_control.restart_scope,
    'restart_phase', v_control.restart_phase
  );
END;
$lab_arena_restart_quiescence$;
ALTER FUNCTION public.lab_arena_restart_quiescence_v1(TEXT, TEXT, BIGINT)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_authorize_restart_phase_v1(
  p_guard_id TEXT, p_owner_id TEXT, p_guard_generation BIGINT, p_phase TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_authorize_restart_phase$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_state JSONB;
  v_guard TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_owner TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_owner_id, 'UTF8'), 'sha256'), 'hex');
BEGIN
  IF COALESCE(p_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR COALESCE(p_owner_id, '') !~ '^lab_arena_restart_owner:[0-9a-f]{64}$'
    OR COALESCE(p_guard_generation, -1) <= 0
    OR COALESCE(p_phase, '') NOT IN ('gateway_destructive', 'validator_destructive') THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_input_invalid';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF v_control.guard_commitment = ''
    OR v_control.guard_commitment IS DISTINCT FROM v_guard
    OR v_control.owner_commitment IS DISTINCT FROM v_owner
    OR v_control.guard_generation IS DISTINCT FROM p_guard_generation
    OR v_control.guard_expires_at IS NULL
    OR v_control.guard_expires_at <= pg_catalog.clock_timestamp() THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_not_active_owner';
  END IF;
  v_state := public.lab_arena__restart_drain_state_v1();
  IF (v_state ->> 'preserved')::BOOLEAN IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_restart_drain_not_preserved';
  END IF;
  -- An exact-owner canonical retry repeats the normal destructive path. It
  -- may rewind only its own retained phase; it cannot skip a component or
  -- release the guard early.
  IF (p_phase = 'gateway_destructive'
      AND v_control.restart_scope = 'gateway'
      AND v_control.restart_phase IN (
        'draining', 'gateway_destructive', 'gateway_ready'
      ))
    OR (p_phase = 'gateway_destructive'
      AND v_control.restart_scope = 'all'
      AND v_control.restart_phase IN (
        'draining', 'gateway_destructive', 'gateway_ready',
        'validator_destructive', 'validator_ready'
      ))
    OR (p_phase = 'validator_destructive'
      AND v_control.restart_scope = 'validator'
      AND v_control.restart_phase IN (
        'draining', 'validator_destructive', 'validator_ready'
      ))
    OR (p_phase = 'validator_destructive'
      AND v_control.restart_scope = 'all'
      AND v_control.restart_phase = 'gateway_ready') THEN
    UPDATE public.lab_arena_restart_claim_control SET
      restart_phase = p_phase, updated_at = pg_catalog.clock_timestamp()
    WHERE singleton;
  ELSE
    RAISE EXCEPTION 'lab_arena_restart_phase_transition_invalid';
  END IF;
  RETURN public.lab_arena_restart_guard_state_v1();
END;
$lab_arena_authorize_restart_phase$;
ALTER FUNCTION public.lab_arena_authorize_restart_phase_v1(
  TEXT, TEXT, BIGINT, TEXT
) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_mark_restart_ready_v1(
  p_guard_id TEXT, p_owner_id TEXT, p_guard_generation BIGINT, p_phase TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_mark_restart_ready$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_guard TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_owner TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_owner_id, 'UTF8'), 'sha256'), 'hex');
BEGIN
  IF COALESCE(p_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR COALESCE(p_owner_id, '') !~ '^lab_arena_restart_owner:[0-9a-f]{64}$'
    OR COALESCE(p_guard_generation, -1) <= 0
    OR COALESCE(p_phase, '') NOT IN ('gateway_ready', 'validator_ready') THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_input_invalid';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF v_control.guard_commitment = ''
    OR v_control.guard_commitment IS DISTINCT FROM v_guard
    OR v_control.owner_commitment IS DISTINCT FROM v_owner
    OR v_control.guard_generation IS DISTINCT FROM p_guard_generation THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_owner_or_generation_differs';
  END IF;
  IF (p_phase = 'gateway_ready' AND v_control.restart_phase = 'gateway_destructive')
    OR (p_phase = 'validator_ready' AND v_control.restart_phase = 'validator_destructive') THEN
    UPDATE public.lab_arena_restart_claim_control SET
      restart_phase = p_phase, updated_at = pg_catalog.clock_timestamp()
    WHERE singleton;
  ELSE
    RAISE EXCEPTION 'lab_arena_restart_phase_transition_invalid';
  END IF;
  RETURN public.lab_arena_restart_guard_state_v1();
END;
$lab_arena_mark_restart_ready$;
ALTER FUNCTION public.lab_arena_mark_restart_ready_v1(TEXT, TEXT, BIGINT, TEXT)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_abort_restart_guard_v1(
  p_guard_id TEXT, p_owner_id TEXT, p_guard_generation BIGINT, p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_abort_restart_guard$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_guard TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_owner TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_owner_id, 'UTF8'), 'sha256'), 'hex');
BEGIN
  IF COALESCE(p_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR COALESCE(p_owner_id, '') !~ '^lab_arena_restart_owner:[0-9a-f]{64}$'
    OR COALESCE(p_guard_generation, -1) <= 0
    OR COALESCE(btrim(p_actor_ref), '') = '' THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_input_invalid';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF v_control.guard_commitment = ''
    OR v_control.guard_commitment IS DISTINCT FROM v_guard
    OR v_control.owner_commitment IS DISTINCT FROM v_owner
    OR v_control.guard_generation IS DISTINCT FROM p_guard_generation
    OR v_control.restart_phase IS DISTINCT FROM 'draining' THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_abort_forbidden';
  END IF;
  UPDATE public.lab_arena_restart_claim_control SET
    pause_reason = CASE WHEN operator_paused THEN pause_reason ELSE '' END,
    actor_ref = left(p_actor_ref, 200), updated_at = pg_catalog.clock_timestamp(),
    guard_commitment = '', owner_commitment = '', guard_expires_at = NULL,
    candidate_commit = '', restart_scope = '', restart_phase = '',
    captured_leases = '[]'::JSONB
  WHERE singleton;
  RETURN public.lab_arena_restart_guard_state_v1();
END;
$lab_arena_abort_restart_guard$;
ALTER FUNCTION public.lab_arena_abort_restart_guard_v1(TEXT, TEXT, BIGINT, TEXT)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_release_restart_guard_v1(
  p_guard_id TEXT, p_owner_id TEXT, p_guard_generation BIGINT, p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_release_restart_guard$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_guard TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_guard_id, 'UTF8'), 'sha256'), 'hex');
  v_owner TEXT := 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.convert_to(p_owner_id, 'UTF8'), 'sha256'), 'hex');
BEGIN
  IF COALESCE(p_guard_id, '') !~ '^lab_arena_restart_guard:[0-9a-f]{64}$'
    OR COALESCE(p_owner_id, '') !~ '^lab_arena_restart_owner:[0-9a-f]{64}$'
    OR COALESCE(p_guard_generation, -1) <= 0
    OR COALESCE(btrim(p_actor_ref), '') = '' THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_input_invalid';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF v_control.guard_commitment = ''
    OR v_control.guard_commitment IS DISTINCT FROM v_guard
    OR v_control.owner_commitment IS DISTINCT FROM v_owner
    OR v_control.guard_generation IS DISTINCT FROM p_guard_generation
    OR NOT (
      (v_control.restart_scope = 'gateway' AND v_control.restart_phase = 'gateway_ready')
      OR (v_control.restart_scope IN ('validator', 'all')
        AND v_control.restart_phase = 'validator_ready')
    ) THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_release_forbidden';
  END IF;
  UPDATE public.lab_arena_restart_claim_control SET
    pause_reason = CASE WHEN operator_paused THEN pause_reason ELSE '' END,
    actor_ref = left(p_actor_ref, 200), updated_at = pg_catalog.clock_timestamp(),
    guard_commitment = '', owner_commitment = '', guard_expires_at = NULL,
    candidate_commit = '', restart_scope = '', restart_phase = '',
    captured_leases = '[]'::JSONB
  WHERE singleton;
  RETURN public.lab_arena_restart_guard_state_v1();
END;
$lab_arena_release_restart_guard$;
ALTER FUNCTION public.lab_arena_release_restart_guard_v1(TEXT, TEXT, BIGINT, TEXT)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$ SELECT pg_catalog.jsonb_build_object(
  'schema_version', 'leadpoet.lab_arena.schema_version.v1', 'version', 190
) $$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;

REVOKE ALL ON FUNCTION public.lab_arena_restart_guard_state_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON FUNCTION public.lab_arena_acquire_restart_guard_v1(
  TEXT, TEXT, BIGINT, INTEGER, TEXT, TEXT, TEXT
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON FUNCTION public.lab_arena_restart_quiescence_v1(
  TEXT, TEXT, BIGINT
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON FUNCTION public.lab_arena_retarget_restart_guard_v1(
  TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, INTEGER, TEXT
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON FUNCTION public.lab_arena_authorize_restart_phase_v1(
  TEXT, TEXT, BIGINT, TEXT
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON FUNCTION public.lab_arena_mark_restart_ready_v1(
  TEXT, TEXT, BIGINT, TEXT
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON FUNCTION public.lab_arena_abort_restart_guard_v1(
  TEXT, TEXT, BIGINT, TEXT
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON FUNCTION public.lab_arena_release_restart_guard_v1(
  TEXT, TEXT, BIGINT, TEXT
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_restart_guard_state_v1()
  TO lab_arena_service, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_acquire_restart_guard_v1(
  TEXT, TEXT, BIGINT, INTEGER, TEXT, TEXT, TEXT
) TO lab_arena_service, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_restart_quiescence_v1(
  TEXT, TEXT, BIGINT
) TO lab_arena_service, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_retarget_restart_guard_v1(
  TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, INTEGER, TEXT
) TO lab_arena_service, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_authorize_restart_phase_v1(
  TEXT, TEXT, BIGINT, TEXT
) TO lab_arena_service, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_mark_restart_ready_v1(
  TEXT, TEXT, BIGINT, TEXT
) TO lab_arena_service, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_abort_restart_guard_v1(
  TEXT, TEXT, BIGINT, TEXT
) TO lab_arena_service, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_release_restart_guard_v1(
  TEXT, TEXT, BIGINT, TEXT
) TO lab_arena_service, service_role;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1()
  TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
