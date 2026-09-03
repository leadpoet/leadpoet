-- 178-lab-arena-v1.sql
--
-- Leadpoet Lab Arena V1 durable data (labarena.md section 11).
--
-- Creates the Arena roles, six lab_arena_* tables, indexes, append-only and
-- write-once triggers, row level security, and the SECURITY DEFINER functions
-- through which every Arena write passes. The Arena service holds only the
-- NOLOGIN role lab_arena_service reached through PostgREST; it never holds the
-- project service key. The migration is additive and idempotent: it applies
-- twice safely and creates the authenticator membership only when that role
-- exists, because the disposable-PostgreSQL harness and Production Parity
-- Fast run without it.
--
-- Lock order inside every function: round, assignment (run), participant
-- budget (submission), account, ledger.

BEGIN;

-- ---------------------------------------------------------------------------
-- Roles (migration-156 pattern: advisory lock + pg_roles guard)
-- ---------------------------------------------------------------------------

DO $lab_arena_roles$
DECLARE
  membership RECORD;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('leadpoet.lab-arena-roles.v1', 0)
  );

  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
  ) THEN
    EXECUTE 'CREATE ROLE lab_arena_owner NOLOGIN';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_service'
  ) THEN
    EXECUTE 'CREATE ROLE lab_arena_service NOLOGIN';
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles
    WHERE rolname IN ('lab_arena_owner', 'lab_arena_service')
      AND (rolsuper OR rolbypassrls OR rolcanlogin)
  ) THEN
    RAISE EXCEPTION 'lab arena roles must be NOLOGIN, non-superuser, and must not bypass RLS';
  END IF;

  -- Hosted Supabase rejects superuser-option syntax in ALTER ROLE, so
  -- NOSUPERUSER and NOBYPASSRLS stay at their CREATE ROLE defaults and are
  -- asserted above by catalog readback rather than written.
  ALTER ROLE lab_arena_owner WITH NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
  ALTER ROLE lab_arena_service WITH NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
  -- The service role shares its PostgreSQL instance with the gateway when the
  -- Arena is not given its own database. Every Arena statement runs as this
  -- role (PostgREST applies impersonated-role settings per request), so its
  -- statements, lock waits, and idle transactions are bounded: an Arena burst
  -- can never hold the gateway's rows or connections for long.
  -- A stage transition at the challenger cap moves about five thousand rows
  -- in one statement (measured near one second); thirty seconds bounds a
  -- runaway statement with headroom for that.
  ALTER ROLE lab_arena_service SET statement_timeout = '30s';
  ALTER ROLE lab_arena_service SET lock_timeout = '5s';
  ALTER ROLE lab_arena_service SET idle_in_transaction_session_timeout = '60s';

  -- The service role is a member of nothing except what PostgREST needs.
  FOR membership IN
    SELECT granted.rolname
    FROM pg_catalog.pg_auth_members member
    JOIN pg_catalog.pg_roles granted ON granted.oid = member.roleid
    JOIN pg_catalog.pg_roles recipient ON recipient.oid = member.member
    WHERE recipient.rolname = 'lab_arena_service'
  LOOP
    EXECUTE pg_catalog.format('REVOKE %I FROM lab_arena_service', membership.rolname);
  END LOOP;

  -- The migration runner must be able to hand table ownership to the owner
  -- role. Membership is granted only when it is missing.
  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_auth_members member
    JOIN pg_catalog.pg_roles granted ON granted.oid = member.roleid
    JOIN pg_catalog.pg_roles recipient ON recipient.oid = member.member
    WHERE granted.rolname = 'lab_arena_owner'
      AND recipient.rolname = current_user
  ) AND NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = current_user AND rolsuper
  ) THEN
    EXECUTE pg_catalog.format('GRANT lab_arena_owner TO %I', current_user);
  END IF;

  -- PostgREST switches from authenticator to the JWT role; grant membership
  -- only when the authenticator role exists (hosted Supabase). The harness
  -- and Production Parity Fast run without it.
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'authenticator')
     AND NOT EXISTS (
       SELECT 1
       FROM pg_catalog.pg_auth_members member
       JOIN pg_catalog.pg_roles granted ON granted.oid = member.roleid
       JOIN pg_catalog.pg_roles recipient ON recipient.oid = member.member
       WHERE granted.rolname = 'lab_arena_service'
         AND recipient.rolname = 'authenticator'
     ) THEN
    EXECUTE 'GRANT lab_arena_service TO authenticator';
  END IF;

  REVOKE CREATE ON SCHEMA public FROM lab_arena_service;
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  GRANT USAGE ON SCHEMA public TO lab_arena_service;
  GRANT USAGE ON SCHEMA public TO lab_arena_owner;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = 'extensions') THEN
    GRANT USAGE ON SCHEMA extensions TO lab_arena_owner;
  END IF;
END;
$lab_arena_roles$;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.lab_arena_rounds (
  round_id TEXT PRIMARY KEY
    CHECK (round_id ~ '^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}(-[a-z0-9]{1,16})?$'),
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open', 'committed', 'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
                      'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published', 'cancelled')),
  status_generation BIGINT NOT NULL DEFAULT 0 CHECK (status_generation >= 0),
  stage_generation BIGINT NOT NULL DEFAULT 0 CHECK (stage_generation >= 0),
  configuration_hash TEXT NOT NULL CHECK (configuration_hash ~ '^sha256:[0-9a-f]{64}$'),
  configuration_doc JSONB NOT NULL,
  journal JSONB NOT NULL DEFAULT '[]'::JSONB CHECK (jsonb_typeof(journal) = 'array'),
  journal_head_hash TEXT NOT NULL DEFAULT '',
  commitment_hash TEXT CHECK (commitment_hash IS NULL OR commitment_hash ~ '^sha256:[0-9a-f]{64}$'),
  commitment_doc JSONB,
  participant_set_hash TEXT,
  participants JSONB,
  benchmark_ref TEXT,
  evaluation_date TEXT,
  stage1_scoring_plan_hash TEXT,
  stage1_scoring_plan_doc JSONB,
  stage2_scoring_plan_hash TEXT,
  stage2_scoring_plan_doc JSONB,
  finalists JSONB,
  stage1_scores_ref TEXT,
  stage1_score_bundle_hash TEXT,
  final_scores_ref TEXT,
  final_score_bundle_hash TEXT,
  result_bundle_hash TEXT CHECK (result_bundle_hash IS NULL OR result_bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
  publication_doc JSONB,
  king_outcome TEXT CHECK (king_outcome IS NULL OR king_outcome IN ('crowned', 'defended', 'retained_ineligible', 'no_king')),
  king_hotkey TEXT,
  king_start_epoch BIGINT CHECK (king_start_epoch IS NULL OR king_start_epoch >= 0),
  effective_reward_epoch BIGINT CHECK (effective_reward_epoch IS NULL OR effective_reward_epoch >= 0),
  reward_basis_hash TEXT CHECK (reward_basis_hash IS NULL OR reward_basis_hash ~ '^sha256:[0-9a-f]{64}$'),
  reward_basis_doc JSONB,
  cancel_reason TEXT,
  published_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_rounds_configuration_hash_uq
  ON public.lab_arena_rounds (configuration_hash);
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_rounds_commitment_hash_uq
  ON public.lab_arena_rounds (commitment_hash) WHERE commitment_hash IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_rounds_result_bundle_hash_uq
  ON public.lab_arena_rounds (result_bundle_hash) WHERE result_bundle_hash IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_rounds_reward_basis_hash_uq
  ON public.lab_arena_rounds (reward_basis_hash) WHERE reward_basis_hash IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_rounds_effective_reward_epoch_uq
  ON public.lab_arena_rounds (effective_reward_epoch)
  WHERE status = 'published' AND effective_reward_epoch IS NOT NULL;
CREATE INDEX IF NOT EXISTS lab_arena_rounds_status_idx
  ON public.lab_arena_rounds (status, created_at);

CREATE TABLE IF NOT EXISTS public.lab_arena_submissions (
  submission_id TEXT PRIMARY KEY CHECK (submission_id ~ '^[A-Za-z0-9._:-]{1,64}$'),
  round_id TEXT NOT NULL REFERENCES public.lab_arena_rounds (round_id),
  miner_hotkey TEXT NOT NULL CHECK (miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  status TEXT NOT NULL DEFAULT 'uploaded'
    CHECK (status IN ('uploaded', 'accepted', 'rejected', 'frozen')),
  is_king BOOLEAN NOT NULL DEFAULT FALSE,
  -- Image by digest: what the miner named, and what the Arena pinned after
  -- resolving and mirroring it (the single-platform manifest digest, the
  -- Arena repository reference, and the process pinned from the image config).
  submitted_reference TEXT,
  submitted_digest TEXT CHECK (submitted_digest IS NULL OR submitted_digest ~ '^sha256:[0-9a-f]{64}$'),
  image_reference TEXT,
  image_digest TEXT CHECK (image_digest IS NULL OR image_digest ~ '^sha256:[0-9a-f]{64}$'),
  entry_command JSONB,
  image_environment JSONB,
  working_dir TEXT,
  image_size_bytes BIGINT,
  consent JSONB,
  rejection_rule TEXT,
  submission_doc JSONB NOT NULL DEFAULT '{}'::JSONB,
  frozen_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

-- A database created by the package-era shape of this migration gains the image columns.
ALTER TABLE public.lab_arena_submissions ADD COLUMN IF NOT EXISTS submitted_reference TEXT;
ALTER TABLE public.lab_arena_submissions ADD COLUMN IF NOT EXISTS submitted_digest TEXT;
ALTER TABLE public.lab_arena_submissions ADD COLUMN IF NOT EXISTS image_reference TEXT;
ALTER TABLE public.lab_arena_submissions ADD COLUMN IF NOT EXISTS entry_command JSONB;
ALTER TABLE public.lab_arena_submissions ADD COLUMN IF NOT EXISTS image_environment JSONB;
ALTER TABLE public.lab_arena_submissions ADD COLUMN IF NOT EXISTS working_dir TEXT;
ALTER TABLE public.lab_arena_submissions ADD COLUMN IF NOT EXISTS image_size_bytes BIGINT;

CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_submissions_one_accepted_per_miner_uq
  ON public.lab_arena_submissions (round_id, miner_hotkey)
  WHERE status IN ('accepted', 'frozen');
-- Two miners cannot enter the same pinned image: the second is rejected at admission.
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_submissions_image_digest_uq
  ON public.lab_arena_submissions (round_id, image_digest)
  WHERE image_digest IS NOT NULL AND status IN ('accepted', 'frozen');
DROP INDEX IF EXISTS public.lab_arena_submissions_source_tree_uq;
CREATE INDEX IF NOT EXISTS lab_arena_submissions_round_idx
  ON public.lab_arena_submissions (round_id, status);

CREATE TABLE IF NOT EXISTS public.lab_arena_runs (
  run_id TEXT PRIMARY KEY,
  assignment_id TEXT NOT NULL,
  round_id TEXT NOT NULL REFERENCES public.lab_arena_rounds (round_id),
  submission_id TEXT NOT NULL REFERENCES public.lab_arena_submissions (submission_id),
  miner_hotkey TEXT NOT NULL,
  stage SMALLINT NOT NULL CHECK (stage IN (1, 2)),
  icp_position SMALLINT NOT NULL CHECK (icp_position BETWEEN 0 AND 49),
  icp_hash TEXT NOT NULL CHECK (icp_hash ~ '^sha256:[0-9a-f]{64}$'),
  attempt SMALLINT NOT NULL CHECK (attempt BETWEEN 0 AND 2),
  -- 'execute' runs a miner's model on one ICP; 'score' runs the Arena judge on
  -- one accepted output (a scoring work item) and is accounted against the
  -- scored miner's keys. Score runs remember the execution they judge.
  kind TEXT NOT NULL DEFAULT 'execute' CHECK (kind IN ('execute', 'score')),
  work_item_id TEXT CHECK (work_item_id IS NULL OR work_item_id ~ '^sha256:[0-9a-f]{64}$'),
  scored_run_id TEXT,
  -- A confirmation attempt remembers who ran the failed attempt before it, so
  -- another validator confirms the failure when the round has more than one.
  previous_runner_hotkey TEXT,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'leased', 'submitted', 'accepted', 'failed')),
  runner_hotkey TEXT,
  lease_token_hash TEXT,
  lease_generation BIGINT NOT NULL DEFAULT 0,
  stage_generation BIGINT NOT NULL DEFAULT 0,
  lease_expires_at TIMESTAMPTZ,
  claim_request_id TEXT,
  claim_request_hash TEXT,
  claim_response JSONB,
  event_cursor INTEGER NOT NULL DEFAULT 0 CHECK (event_cursor >= 0),
  event_head_hash TEXT NOT NULL DEFAULT '',
  receipt_doc JSONB,
  receipt_hash TEXT,
  output_hash TEXT,
  output_ref TEXT,
  provider_call_root TEXT,
  private_event_root TEXT,
  cost_root TEXT,
  terminal_cause TEXT CHECK (terminal_cause IS NULL OR terminal_cause IN (
    'accepted', 'model_timeout', 'invalid_output', 'budget_exhausted', 'model_error',
    'lease_expired', 'worker_lost', 'receipt_rejected', 'preflight_failed', 'stage_closed',
    'judge_error', 'judge_timeout', 'judge_key_refused', 'replay_rejected')),
  terminal_doc JSONB,
  per_icp_score NUMERIC(12, 6),
  score_ref TEXT,
  score_doc JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  CONSTRAINT lab_arena_runs_preflight_failed_has_no_attempt CHECK (
    (attempt = 0) = (terminal_cause = 'preflight_failed')
  )
);

CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_runs_assignment_attempt_uq
  ON public.lab_arena_runs (assignment_id, attempt);
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_runs_one_active_attempt_uq
  ON public.lab_arena_runs (assignment_id)
  WHERE status IN ('pending', 'leased', 'submitted');
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_runs_one_accepted_uq
  ON public.lab_arena_runs (assignment_id)
  WHERE status = 'accepted';
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_runs_claim_request_uq
  ON public.lab_arena_runs (round_id, claim_request_id)
  WHERE claim_request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS lab_arena_runs_pending_order_idx
  ON public.lab_arena_runs (round_id, stage, icp_position, created_at, assignment_id)
  WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS lab_arena_runs_leases_idx
  ON public.lab_arena_runs (round_id, runner_hotkey, lease_expires_at)
  WHERE status = 'leased';
CREATE INDEX IF NOT EXISTS lab_arena_runs_submission_stage_idx
  ON public.lab_arena_runs (submission_id, stage);

CREATE TABLE IF NOT EXISTS public.lab_arena_events (
  event_id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES public.lab_arena_runs (run_id),
  sequence INTEGER NOT NULL CHECK (sequence >= 0),
  event_type TEXT NOT NULL,
  event_doc JSONB NOT NULL,
  prev_hash TEXT NOT NULL,
  event_hash TEXT NOT NULL CHECK (event_hash ~ '^sha256:[0-9a-f]{64}$'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  UNIQUE (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS public.lab_arena_accounts (
  miner_hotkey TEXT PRIMARY KEY CHECK (miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  -- One entry per provider: {"ciphertext", "key_hash", "preflight_status", "preflight", "observed_at"}.
  -- The miner's own keys pay for every provider call; there is no Arena balance.
  credentials JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (pg_catalog.jsonb_typeof(credentials) = 'object'),
  preflight_status TEXT NOT NULL DEFAULT 'none'
    CHECK (preflight_status IN ('none', 'ok', 'failed')),
  observed_limit_microusd BIGINT,
  observed_limit_remaining_microusd BIGINT,
  observed_usage_microusd BIGINT,
  observed_at TIMESTAMPTZ,
  outstanding_openrouter_reservation_microusd BIGINT NOT NULL DEFAULT 0
    CHECK (outstanding_openrouter_reservation_microusd >= 0),
  settled_since_preflight_microusd BIGINT NOT NULL DEFAULT 0
    CHECK (settled_since_preflight_microusd >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lab_arena_ledger (
  entry_id BIGSERIAL PRIMARY KEY,
  entry_kind TEXT NOT NULL CHECK (entry_kind IN (
    'reservation', 'dispatch', 'settlement', 'uncertain', 'recovery', 'refusal')),
  miner_hotkey TEXT NOT NULL,
  round_id TEXT,
  submission_id TEXT,
  run_id TEXT,
  stage SMALLINT CHECK (stage IS NULL OR stage IN (1, 2)),
  call_identity TEXT CHECK (call_identity IS NULL OR call_identity ~ '^sha256:[0-9a-f]{64}$'),
  provider TEXT,
  operation_id TEXT,
  funding_source TEXT CHECK (funding_source IS NULL OR funding_source = 'miner_key'),
  amount_microusd BIGINT NOT NULL CHECK (amount_microusd >= 0),
  entry_doc JSONB NOT NULL DEFAULT '{}'::JSONB,
  terminal_response JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_ledger_reservation_uq
  ON public.lab_arena_ledger (call_identity) WHERE entry_kind = 'reservation';
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_ledger_dispatch_uq
  ON public.lab_arena_ledger (call_identity) WHERE entry_kind = 'dispatch';
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_ledger_terminal_uq
  ON public.lab_arena_ledger (call_identity)
  WHERE entry_kind IN ('settlement', 'uncertain', 'recovery', 'refusal');
CREATE INDEX IF NOT EXISTS lab_arena_ledger_call_idx
  ON public.lab_arena_ledger (call_identity, entry_id DESC) WHERE call_identity IS NOT NULL;
CREATE INDEX IF NOT EXISTS lab_arena_ledger_run_idx
  ON public.lab_arena_ledger (run_id, entry_id) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS lab_arena_ledger_submission_stage_idx
  ON public.lab_arena_ledger (submission_id, stage, entry_id) WHERE submission_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Ownership, triggers, and the service whoami readback
-- ---------------------------------------------------------------------------

ALTER TABLE public.lab_arena_rounds OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_submissions OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_runs OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_events OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_accounts OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_ledger OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_append_only_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_append_only$
BEGIN
  RAISE EXCEPTION '% is append-only' , TG_TABLE_NAME USING ERRCODE = '42501';
END;
$lab_arena_append_only$;
ALTER FUNCTION public.lab_arena_append_only_v1() OWNER TO lab_arena_owner;

DROP TRIGGER IF EXISTS lab_arena_events_append_only ON public.lab_arena_events;
CREATE TRIGGER lab_arena_events_append_only
  BEFORE UPDATE OR DELETE ON public.lab_arena_events
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();
DROP TRIGGER IF EXISTS lab_arena_ledger_append_only ON public.lab_arena_ledger;
CREATE TRIGGER lab_arena_ledger_append_only
  BEFORE UPDATE OR DELETE ON public.lab_arena_ledger
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_rounds_write_once_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_rounds_write_once$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'lab_arena_rounds rows are never deleted' USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'published' THEN
    RAISE EXCEPTION 'published round is immutable' USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'cancelled' AND NEW.status <> 'cancelled' THEN
    RAISE EXCEPTION 'cancelled round cannot be reopened' USING ERRCODE = '42501';
  END IF;
  IF NEW.configuration_hash <> OLD.configuration_hash
     OR NEW.configuration_doc <> OLD.configuration_doc THEN
    RAISE EXCEPTION 'round configuration is write-once' USING ERRCODE = '42501';
  END IF;
  IF (OLD.commitment_hash IS NOT NULL AND NEW.commitment_hash IS DISTINCT FROM OLD.commitment_hash)
     OR (OLD.commitment_doc IS NOT NULL AND NEW.commitment_doc IS DISTINCT FROM OLD.commitment_doc)
     OR (OLD.participant_set_hash IS NOT NULL AND NEW.participant_set_hash IS DISTINCT FROM OLD.participant_set_hash)
     OR (OLD.stage1_scoring_plan_hash IS NOT NULL AND NEW.stage1_scoring_plan_hash IS DISTINCT FROM OLD.stage1_scoring_plan_hash)
     OR (OLD.stage2_scoring_plan_hash IS NOT NULL AND NEW.stage2_scoring_plan_hash IS DISTINCT FROM OLD.stage2_scoring_plan_hash)
     OR (OLD.finalists IS NOT NULL AND NEW.finalists IS DISTINCT FROM OLD.finalists)
     OR (OLD.stage1_score_bundle_hash IS NOT NULL AND NEW.stage1_score_bundle_hash IS DISTINCT FROM OLD.stage1_score_bundle_hash)
     OR (OLD.final_score_bundle_hash IS NOT NULL AND NEW.final_score_bundle_hash IS DISTINCT FROM OLD.final_score_bundle_hash)
     OR (OLD.result_bundle_hash IS NOT NULL AND NEW.result_bundle_hash IS DISTINCT FROM OLD.result_bundle_hash)
     OR (OLD.publication_doc IS NOT NULL AND NEW.publication_doc IS DISTINCT FROM OLD.publication_doc)
     OR (OLD.reward_basis_hash IS NOT NULL AND NEW.reward_basis_hash IS DISTINCT FROM OLD.reward_basis_hash)
     OR (OLD.reward_basis_doc IS NOT NULL AND NEW.reward_basis_doc IS DISTINCT FROM OLD.reward_basis_doc)
     OR (OLD.king_outcome IS NOT NULL AND NEW.king_outcome IS DISTINCT FROM OLD.king_outcome)
     OR (OLD.effective_reward_epoch IS NOT NULL AND NEW.effective_reward_epoch IS DISTINCT FROM OLD.effective_reward_epoch) THEN
    RAISE EXCEPTION 'round publication and commitment columns are write-once' USING ERRCODE = '42501';
  END IF;
  IF NEW.status = 'published' AND (
       NEW.publication_doc IS NULL OR NEW.reward_basis_doc IS NULL
       OR NEW.reward_basis_hash IS NULL OR NEW.result_bundle_hash IS NULL
       OR NEW.king_outcome IS NULL OR NEW.effective_reward_epoch IS NULL) THEN
    RAISE EXCEPTION 'publication requires every signed reward-basis column' USING ERRCODE = '23514';
  END IF;
  NEW.updated_at := pg_catalog.clock_timestamp();
  RETURN NEW;
END;
$lab_arena_rounds_write_once$;
ALTER FUNCTION public.lab_arena_rounds_write_once_v1() OWNER TO lab_arena_owner;

DROP TRIGGER IF EXISTS lab_arena_rounds_write_once ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_rounds_write_once
  BEFORE UPDATE OR DELETE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_rounds_write_once_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_submissions_frozen_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_submissions_frozen$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'lab_arena_submissions rows are never deleted' USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'frozen' THEN
    RAISE EXCEPTION 'frozen submission is immutable' USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' THEN
    RAISE EXCEPTION 'rejected submission cannot be reopened' USING ERRCODE = '42501';
  END IF;
  NEW.updated_at := pg_catalog.clock_timestamp();
  RETURN NEW;
END;
$lab_arena_submissions_frozen$;
ALTER FUNCTION public.lab_arena_submissions_frozen_v1() OWNER TO lab_arena_owner;

DROP TRIGGER IF EXISTS lab_arena_submissions_frozen ON public.lab_arena_submissions;
CREATE TRIGGER lab_arena_submissions_frozen
  BEFORE UPDATE OR DELETE ON public.lab_arena_submissions
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_submissions_frozen_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_runs_terminal_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_runs_terminal$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'lab_arena_runs rows are never deleted' USING ERRCODE = '42501';
  END IF;
  -- One explicit exception to terminal immutability: the gateway may fail an
  -- accepted score run as replay_rejected when its replay of the recorded
  -- provider responses cannot reproduce the reported scoring. The validator's
  -- receipt, output, event chain, and lease columns stay exactly as delivered,
  -- so the rejection is visible and re-checkable from the ledger.
  IF OLD.status IN ('accepted', 'failed') AND (
       ((NEW.status <> OLD.status OR NEW.terminal_cause IS DISTINCT FROM OLD.terminal_cause)
        AND NOT (OLD.status = 'accepted' AND OLD.kind = 'score'
                 AND NEW.status = 'failed' AND NEW.terminal_cause = 'replay_rejected'))
       OR NEW.receipt_doc IS DISTINCT FROM OLD.receipt_doc
       OR NEW.receipt_hash IS DISTINCT FROM OLD.receipt_hash
       OR NEW.output_hash IS DISTINCT FROM OLD.output_hash
       OR NEW.output_ref IS DISTINCT FROM OLD.output_ref
       OR NEW.event_cursor <> OLD.event_cursor
       OR NEW.event_head_hash <> OLD.event_head_hash
       OR NEW.lease_generation <> OLD.lease_generation) THEN
    RAISE EXCEPTION 'terminal attempt is immutable' USING ERRCODE = '42501';
  END IF;
  IF OLD.per_icp_score IS NOT NULL AND NEW.per_icp_score IS DISTINCT FROM OLD.per_icp_score THEN
    RAISE EXCEPTION 'attempt score is write-once' USING ERRCODE = '42501';
  END IF;
  IF OLD.claim_request_id IS NOT NULL AND NEW.claim_request_id IS DISTINCT FROM OLD.claim_request_id THEN
    RAISE EXCEPTION 'claim request binding is write-once' USING ERRCODE = '42501';
  END IF;
  NEW.updated_at := pg_catalog.clock_timestamp();
  RETURN NEW;
END;
$lab_arena_runs_terminal$;
ALTER FUNCTION public.lab_arena_runs_terminal_v1() OWNER TO lab_arena_owner;

DROP TRIGGER IF EXISTS lab_arena_runs_terminal ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_runs_terminal
  BEFORE UPDATE OR DELETE ON public.lab_arena_runs
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_runs_terminal_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_accounts_touch_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_accounts_touch$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'lab_arena_accounts rows are never deleted' USING ERRCODE = '42501';
  END IF;
  NEW.updated_at := pg_catalog.clock_timestamp();
  RETURN NEW;
END;
$lab_arena_accounts_touch$;
ALTER FUNCTION public.lab_arena_accounts_touch_v1() OWNER TO lab_arena_owner;

DROP TRIGGER IF EXISTS lab_arena_accounts_touch ON public.lab_arena_accounts;
CREATE TRIGGER lab_arena_accounts_touch
  BEFORE UPDATE OR DELETE ON public.lab_arena_accounts
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_accounts_touch_v1();

-- whoami runs as the caller (SECURITY INVOKER) so the service can prove
-- that PostgREST switched to lab_arena_service and that the role is neither
-- superuser nor BYPASSRLS (section 11.1).
CREATE OR REPLACE FUNCTION public.lab_arena_whoami()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $lab_arena_whoami$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.whoami.v1',
    'current_user', current_user::TEXT,
    'session_user', session_user::TEXT,
    'jwt_role', (NULLIF(pg_catalog.current_setting('request.jwt.claims', TRUE), '')::JSONB) ->> 'role',
    'rolsuper', roles.rolsuper,
    'rolbypassrls', roles.rolbypassrls,
    'rolcanlogin', roles.rolcanlogin,
    'rolinherit', roles.rolinherit,
    'rolcreaterole', roles.rolcreaterole,
    'rolcreatedb', roles.rolcreatedb
  )
  FROM pg_catalog.pg_roles AS roles
  WHERE roles.rolname = current_user;
$lab_arena_whoami$;
ALTER FUNCTION public.lab_arena_whoami() OWNER TO lab_arena_owner;

-- ---------------------------------------------------------------------------
-- Internal helpers (owner-only; never granted to the service)
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.lab_arena__ledger_head(p_call_identity TEXT)
RETURNS public.lab_arena_ledger
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__ledger_head$
  SELECT ledger.*
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.call_identity = p_call_identity
  ORDER BY ledger.entry_id DESC
  LIMIT 1;
$lab_arena__ledger_head$;
ALTER FUNCTION public.lab_arena__ledger_head(TEXT) OWNER TO lab_arena_owner;

-- Money a run has consumed: the head entry of every call identity, counting
-- live reservations, dispatched calls, settlements, and uncertain marks.
-- Recovered and refused calls consume nothing.
-- Calls of one provider consumed by one attempt: every call identity whose
-- newest ledger entry is a reservation, dispatch, settlement, or uncertain
-- outcome counts once against the per-ICP quota; recovered and refused calls
-- do not.
CREATE OR REPLACE FUNCTION public.lab_arena__run_consumed(p_run_id TEXT, p_provider TEXT)
RETURNS BIGINT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__run_consumed$
  SELECT COUNT(*)::BIGINT
  FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.run_id = p_run_id AND ledger.provider = p_provider AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head
  WHERE head.entry_kind IN ('reservation', 'dispatch', 'settlement', 'uncertain');
$lab_arena__run_consumed$;
ALTER FUNCTION public.lab_arena__run_consumed(TEXT, TEXT) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__submission_stage_consumed(p_submission_id TEXT, p_stage SMALLINT, p_provider TEXT, p_kind TEXT)
RETURNS BIGINT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__submission_stage_consumed$
  SELECT COUNT(*)::BIGINT
  FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND ledger.stage = p_stage
      AND ledger.provider = p_provider
      AND runs.kind = p_kind
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head
  WHERE head.entry_kind IN ('reservation', 'dispatch', 'settlement', 'uncertain');
$lab_arena__submission_stage_consumed$;
ALTER FUNCTION public.lab_arena__submission_stage_consumed(TEXT, SMALLINT, TEXT, TEXT) OWNER TO lab_arena_owner;

-- Aggregate key state: 'ok' only when every required provider has a key
-- whose newest preflight passed; 'none' before any key; otherwise 'failed'.
CREATE OR REPLACE FUNCTION public.lab_arena__aggregate_preflight(p_credentials JSONB)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, public
AS $lab_arena__aggregate_preflight$
  SELECT CASE
    WHEN p_credentials IS NULL OR p_credentials = '{}'::JSONB THEN 'none'
    WHEN (SELECT bool_and(COALESCE(p_credentials -> provider ->> 'preflight_status', '') = 'ok')
          FROM unnest(ARRAY['scrapingdog', 'deepline', 'openrouter']) AS provider) THEN 'ok'
    ELSE 'failed' END;
$lab_arena__aggregate_preflight$;
ALTER FUNCTION public.lab_arena__aggregate_preflight(JSONB) OWNER TO lab_arena_owner;

-- Terminal funding effects. TAO credit was deducted at reservation; a
-- recovery returns it and a settlement/uncertain keeps it. OpenRouter
-- capacity is tracked as an outstanding reservation total plus the amount
-- settled since the last preflight observation.
CREATE OR REPLACE FUNCTION public.lab_arena__apply_terminal_funding(
  p_reservation public.lab_arena_ledger,
  p_terminal_kind TEXT,
  p_actual_microusd BIGINT
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__apply_terminal_funding$
DECLARE
  v_account public.lab_arena_accounts;
BEGIN
  SELECT * INTO v_account FROM public.lab_arena_accounts
  WHERE miner_hotkey = p_reservation.miner_hotkey FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_account_missing' USING ERRCODE = '23503';
  END IF;
  IF p_reservation.funding_source IS DISTINCT FROM 'miner_key' THEN
    RAISE EXCEPTION 'lab_arena_funding_source_invalid' USING ERRCODE = '22023';
  END IF;
  -- Only OpenRouter reports a key limit; its outstanding reservation and
  -- settled totals track the remaining capacity observed at preflight.
  IF p_reservation.provider = 'openrouter' THEN
    UPDATE public.lab_arena_accounts
    SET outstanding_openrouter_reservation_microusd =
          GREATEST(0, outstanding_openrouter_reservation_microusd - p_reservation.amount_microusd),
        settled_since_preflight_microusd = settled_since_preflight_microusd + CASE
          WHEN p_terminal_kind = 'recovery' THEN 0
          WHEN p_terminal_kind = 'uncertain' THEN p_reservation.amount_microusd
          ELSE COALESCE(p_actual_microusd, p_reservation.amount_microusd)
        END
    WHERE miner_hotkey = p_reservation.miner_hotkey;
  END IF;
END;
$lab_arena__apply_terminal_funding$;
ALTER FUNCTION public.lab_arena__apply_terminal_funding(public.lab_arena_ledger, TEXT, BIGINT) OWNER TO lab_arena_owner;

-- Close every open call of one run: undispatched reservations are recovered
-- once, dispatched calls without a terminal result become uncertain at their
-- full reservation (section 7.5). Used by lease expiry, stage close, and
-- cancellation. Returns the number of calls touched.
CREATE OR REPLACE FUNCTION public.lab_arena__terminate_open_calls(p_run_id TEXT, p_reason TEXT)
RETURNS INTEGER
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__terminate_open_calls$
DECLARE
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_count INTEGER := 0;
BEGIN
  FOR v_head IN
    SELECT * FROM (
      SELECT DISTINCT ON (ledger.call_identity) ledger.*
      FROM public.lab_arena_ledger AS ledger
      WHERE ledger.run_id = p_run_id AND ledger.call_identity IS NOT NULL
      ORDER BY ledger.call_identity, ledger.entry_id DESC
    ) AS heads
    WHERE heads.entry_kind IN ('reservation', 'dispatch')
    ORDER BY heads.call_identity
  LOOP
    SELECT * INTO v_reservation FROM public.lab_arena_ledger
    WHERE call_identity = v_head.call_identity AND entry_kind = 'reservation';
    IF v_head.entry_kind = 'reservation' THEN
      INSERT INTO public.lab_arena_ledger (
        entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
        provider, operation_id, funding_source, amount_microusd, entry_doc
      ) VALUES (
        'recovery', v_reservation.miner_hotkey, v_reservation.round_id, v_reservation.submission_id,
        v_reservation.run_id, v_reservation.stage, v_reservation.call_identity, v_reservation.provider,
        v_reservation.operation_id, v_reservation.funding_source, 0,
        pg_catalog.jsonb_build_object('reason', p_reason, 'reserved_microusd', v_reservation.amount_microusd)
      );
      PERFORM public.lab_arena__apply_terminal_funding(v_reservation, 'recovery', 0);
    ELSE
      INSERT INTO public.lab_arena_ledger (
        entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
        provider, operation_id, funding_source, amount_microusd, entry_doc
      ) VALUES (
        'uncertain', v_reservation.miner_hotkey, v_reservation.round_id, v_reservation.submission_id,
        v_reservation.run_id, v_reservation.stage, v_reservation.call_identity, v_reservation.provider,
        v_reservation.operation_id, v_reservation.funding_source, v_reservation.amount_microusd,
        pg_catalog.jsonb_build_object('reason', p_reason)
      );
      PERFORM public.lab_arena__apply_terminal_funding(v_reservation, 'uncertain', v_reservation.amount_microusd);
    END IF;
    v_count := v_count + 1;
  END LOOP;
  RETURN v_count;
END;
$lab_arena__terminate_open_calls$;
ALTER FUNCTION public.lab_arena__terminate_open_calls(TEXT, TEXT) OWNER TO lab_arena_owner;

-- Lease/stage validity for run-scoped calls. Locks the round FOR SHARE and
-- the run FOR UPDATE (lock order: round, assignment) and returns the run row.
-- Raises a structured error code the callers translate into a stale status.
CREATE OR REPLACE FUNCTION public.lab_arena__lock_current_lease(
  p_run_id TEXT,
  p_lease_token_hash TEXT
)
RETURNS public.lab_arena_runs
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__lock_current_lease$
DECLARE
  v_round_id TEXT;
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
BEGIN
  SELECT round_id INTO v_round_id FROM public.lab_arena_runs WHERE run_id = p_run_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_run_missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = v_round_id FOR SHARE;
  SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = p_run_id FOR UPDATE;
  IF v_run.status <> 'leased'
     OR v_run.lease_token_hash IS DISTINCT FROM p_lease_token_hash
     OR v_run.lease_expires_at IS NULL
     OR v_run.lease_expires_at <= pg_catalog.clock_timestamp()
     OR v_run.stage_generation <> v_round.stage_generation
     OR v_round.status <> ('stage' || v_run.stage::TEXT || CASE v_run.kind WHEN 'score' THEN '_scoring' ELSE '' END) THEN
    RAISE EXCEPTION 'lab_arena_lease_stale' USING ERRCODE = 'P0003';
  END IF;
  RETURN v_run;
END;
$lab_arena__lock_current_lease$;
ALTER FUNCTION public.lab_arena__lock_current_lease(TEXT, TEXT) OWNER TO lab_arena_owner;

-- Append one worker-built private event under an already-locked run.
CREATE OR REPLACE FUNCTION public.lab_arena__append_event_locked(
  p_run public.lab_arena_runs,
  p_event JSONB
)
RETURNS public.lab_arena_runs
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__append_event_locked$
DECLARE
  v_sequence INTEGER;
  v_prev TEXT;
  v_hash TEXT;
  v_type TEXT;
BEGIN
  IF pg_catalog.jsonb_typeof(p_event) <> 'object'
     OR NOT (p_event ? 'sequence') OR NOT (p_event ? 'prev_hash')
     OR NOT (p_event ? 'event_hash') OR NOT (p_event ? 'event_type')
     OR pg_catalog.octet_length(p_event::TEXT) > 524288 THEN
    RAISE EXCEPTION 'lab_arena_event_invalid' USING ERRCODE = '22023';
  END IF;
  v_sequence := (p_event ->> 'sequence')::INTEGER;
  v_prev := p_event ->> 'prev_hash';
  v_hash := p_event ->> 'event_hash';
  v_type := p_event ->> 'event_type';
  IF v_hash !~ '^sha256:[0-9a-f]{64}$' OR v_type !~ '^[a-z_]{1,64}$' THEN
    RAISE EXCEPTION 'lab_arena_event_invalid' USING ERRCODE = '22023';
  END IF;
  IF v_sequence <> p_run.event_cursor OR v_prev <> p_run.event_head_hash THEN
    RAISE EXCEPTION 'lab_arena_event_sequence' USING ERRCODE = 'P0004';
  END IF;
  INSERT INTO public.lab_arena_events (run_id, sequence, event_type, event_doc, prev_hash, event_hash)
  VALUES (p_run.run_id, v_sequence, v_type, p_event, v_prev, v_hash);
  UPDATE public.lab_arena_runs
  SET event_cursor = v_sequence + 1, event_head_hash = v_hash
  WHERE run_id = p_run.run_id
  RETURNING * INTO p_run;
  RETURN p_run;
END;
$lab_arena__append_event_locked$;
ALTER FUNCTION public.lab_arena__append_event_locked(public.lab_arena_runs, JSONB) OWNER TO lab_arena_owner;

-- ---------------------------------------------------------------------------
-- Round lifecycle, generation journal, submissions, accounts, deposits
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.lab_arena_create_round(
  p_round_id TEXT,
  p_configuration_hash TEXT,
  p_configuration_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_create_round$
DECLARE
  v_existing public.lab_arena_rounds;
BEGIN
  IF COALESCE(p_round_id, '') !~ '^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}(-[a-z0-9]{1,16})?$'
     OR COALESCE(p_configuration_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR pg_catalog.jsonb_typeof(p_configuration_doc) IS DISTINCT FROM 'object'
     OR (p_configuration_doc ->> 'configuration_hash') IS DISTINCT FROM p_configuration_hash THEN
    RAISE EXCEPTION 'lab_arena_round_input_invalid' USING ERRCODE = '22023';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('lab_arena.rounds', 0));
  SELECT * INTO v_existing FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  IF FOUND THEN
    IF v_existing.configuration_hash = p_configuration_hash THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_id', p_round_id,
        'round_status', v_existing.status, 'status_generation', v_existing.status_generation);
    END IF;
    RAISE EXCEPTION 'lab_arena_round_configuration_conflict' USING ERRCODE = '23505';
  END IF;
  INSERT INTO public.lab_arena_rounds (round_id, status, configuration_hash, configuration_doc)
  VALUES (p_round_id, 'open', p_configuration_hash, p_configuration_doc);
  RETURN pg_catalog.jsonb_build_object(
    'status', 'created', 'round_id', p_round_id, 'round_status', 'open', 'status_generation', 0);
END;
$lab_arena_create_round$;
ALTER FUNCTION public.lab_arena_create_round(TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;

-- Compare-and-set round transitions with write-once patches. Stage opening
-- and closing have their own functions because they touch assignments.
CREATE OR REPLACE FUNCTION public.lab_arena_transition_round(
  p_round_id TEXT,
  p_expected_status TEXT,
  p_next_status TEXT,
  p_patch JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_transition_round$
DECLARE
  v_round public.lab_arena_rounds;
  v_patch JSONB := COALESCE(p_patch, '{}'::JSONB);
  v_keys TEXT[];
  v_allowed TEXT[];
  v_key TEXT;
  v_existing_same BOOLEAN := TRUE;
BEGIN
  IF pg_catalog.jsonb_typeof(v_patch) <> 'object' THEN
    RAISE EXCEPTION 'lab_arena_patch_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> p_expected_status THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
  END IF;
  SELECT COALESCE(pg_catalog.array_agg(key ORDER BY key), ARRAY[]::TEXT[]) INTO v_keys
  FROM pg_catalog.jsonb_object_keys(v_patch) AS key;

  IF p_expected_status = 'open' AND p_next_status = 'committed' THEN
    v_allowed := ARRAY['commitment_hash', 'commitment_doc', 'participant_set_hash', 'participants', 'benchmark_ref', 'evaluation_date'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys) THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF (v_patch ->> 'commitment_hash') !~ '^sha256:[0-9a-f]{64}$'
       OR (v_patch -> 'commitment_doc' ->> 'commitment_hash') IS DISTINCT FROM (v_patch ->> 'commitment_hash')
       OR (v_patch -> 'commitment_doc' ->> 'configuration_hash') IS DISTINCT FROM v_round.configuration_hash
       OR pg_catalog.jsonb_typeof(v_patch -> 'participants') <> 'array' THEN
      RAISE EXCEPTION 'lab_arena_commitment_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'committed', status_generation = status_generation + 1,
        commitment_hash = v_patch ->> 'commitment_hash',
        commitment_doc = v_patch -> 'commitment_doc',
        participant_set_hash = v_patch ->> 'participant_set_hash',
        participants = v_patch -> 'participants',
        benchmark_ref = v_patch ->> 'benchmark_ref',
        evaluation_date = v_patch ->> 'evaluation_date'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage1_closed' AND p_next_status = 'stage1_closed' THEN
    v_allowed := ARRAY['stage1_scoring_plan_doc', 'stage1_scoring_plan_hash'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys) OR (v_patch ->> 'stage1_scoring_plan_hash') !~ '^sha256:[0-9a-f]{64}$' THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_round.stage1_scoring_plan_hash IS NOT NULL THEN
      IF v_round.stage1_scoring_plan_hash = (v_patch ->> 'stage1_scoring_plan_hash') THEN
        RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
      END IF;
      RAISE EXCEPTION 'lab_arena_scoring_plan_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_rounds
    SET stage1_scoring_plan_hash = v_patch ->> 'stage1_scoring_plan_hash',
        stage1_scoring_plan_doc = v_patch -> 'stage1_scoring_plan_doc'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage2_closed' AND p_next_status = 'stage2_closed' THEN
    v_allowed := ARRAY['stage2_scoring_plan_doc', 'stage2_scoring_plan_hash'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys) OR (v_patch ->> 'stage2_scoring_plan_hash') !~ '^sha256:[0-9a-f]{64}$' THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_round.stage2_scoring_plan_hash IS NOT NULL THEN
      IF v_round.stage2_scoring_plan_hash = (v_patch ->> 'stage2_scoring_plan_hash') THEN
        RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
      END IF;
      RAISE EXCEPTION 'lab_arena_scoring_plan_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_rounds
    SET stage2_scoring_plan_hash = v_patch ->> 'stage2_scoring_plan_hash',
        stage2_scoring_plan_doc = v_patch -> 'stage2_scoring_plan_doc'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage1_judged' AND p_next_status = 'stage1_scored' THEN
    v_allowed := ARRAY['finalists', 'stage1_score_bundle_hash', 'stage1_scores_ref'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'finalists') <> 'array'
       OR (v_patch ->> 'stage1_score_bundle_hash') !~ '^sha256:[0-9a-f]{64}$'
       OR v_round.stage1_scoring_plan_hash IS NULL THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'stage1_scored', status_generation = status_generation + 1,
        finalists = v_patch -> 'finalists',
        stage1_scores_ref = v_patch ->> 'stage1_scores_ref',
        stage1_score_bundle_hash = v_patch ->> 'stage1_score_bundle_hash'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage2_judged' AND p_next_status = 'scored' THEN
    v_allowed := ARRAY['final_score_bundle_hash', 'final_scores_ref'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR (v_patch ->> 'final_score_bundle_hash') !~ '^sha256:[0-9a-f]{64}$'
       OR v_round.stage2_scoring_plan_hash IS NULL THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'scored', status_generation = status_generation + 1,
        final_scores_ref = v_patch ->> 'final_scores_ref',
        final_score_bundle_hash = v_patch ->> 'final_score_bundle_hash'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'scored' AND p_next_status = 'published' THEN
    v_allowed := ARRAY['effective_reward_epoch', 'king_hotkey', 'king_outcome', 'king_start_epoch',
                       'publication_doc', 'result_bundle_hash', 'reward_basis_doc', 'reward_basis_hash'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR (v_patch ->> 'result_bundle_hash') !~ '^sha256:[0-9a-f]{64}$'
       OR (v_patch ->> 'reward_basis_hash') !~ '^sha256:[0-9a-f]{64}$'
       OR (v_patch ->> 'king_outcome') NOT IN ('crowned', 'defended', 'retained_ineligible', 'no_king')
       OR pg_catalog.jsonb_typeof(v_patch -> 'effective_reward_epoch') <> 'number'
       OR pg_catalog.jsonb_typeof(v_patch -> 'king_start_epoch') <> 'number'
       OR (v_patch -> 'reward_basis_doc' ->> 'reward_basis_hash') IS DISTINCT FROM (v_patch ->> 'reward_basis_hash')
       OR (v_patch -> 'reward_basis_doc' ->> 'result_bundle_hash') IS DISTINCT FROM (v_patch ->> 'result_bundle_hash')
       OR (v_patch -> 'reward_basis_doc' ->> 'king_outcome') IS DISTINCT FROM (v_patch ->> 'king_outcome')
       OR (v_patch -> 'reward_basis_doc' ->> 'effective_reward_epoch') IS DISTINCT FROM (v_patch ->> 'effective_reward_epoch')
       OR (v_patch -> 'publication_doc' ->> 'result_bundle_hash') IS DISTINCT FROM (v_patch ->> 'result_bundle_hash') THEN
      RAISE EXCEPTION 'lab_arena_publication_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'published', status_generation = status_generation + 1,
        result_bundle_hash = v_patch ->> 'result_bundle_hash',
        publication_doc = v_patch -> 'publication_doc',
        king_outcome = v_patch ->> 'king_outcome',
        king_hotkey = NULLIF(v_patch ->> 'king_hotkey', ''),
        king_start_epoch = (v_patch ->> 'king_start_epoch')::BIGINT,
        effective_reward_epoch = (v_patch ->> 'effective_reward_epoch')::BIGINT,
        reward_basis_hash = v_patch ->> 'reward_basis_hash',
        reward_basis_doc = v_patch -> 'reward_basis_doc',
        published_at = pg_catalog.clock_timestamp()
    WHERE round_id = p_round_id;
  ELSE
    RAISE EXCEPTION 'lab_arena_transition_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
END;
$lab_arena_transition_round$;
ALTER FUNCTION public.lab_arena_transition_round(TEXT, TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_append_journal_entry(
  p_round_id TEXT,
  p_entry JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_append_journal_entry$
DECLARE
  v_round public.lab_arena_rounds;
  v_length INTEGER;
  v_sequence INTEGER;
  v_prev TEXT;
  v_hash TEXT;
BEGIN
  IF pg_catalog.jsonb_typeof(p_entry) IS DISTINCT FROM 'object'
     OR NOT (p_entry ? 'sequence') OR NOT (p_entry ? 'prev_hash') OR NOT (p_entry ? 'entry_hash')
     OR (p_entry ->> 'entry_hash') !~ '^sha256:[0-9a-f]{64}$'
     OR pg_catalog.octet_length(p_entry::TEXT) > 262144 THEN
    RAISE EXCEPTION 'lab_arena_journal_entry_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status,
      'journal_length', pg_catalog.jsonb_array_length(v_round.journal), 'journal_head_hash', v_round.journal_head_hash);
  END IF;
  v_length := pg_catalog.jsonb_array_length(v_round.journal);
  v_sequence := (p_entry ->> 'sequence')::INTEGER;
  v_prev := p_entry ->> 'prev_hash';
  v_hash := p_entry ->> 'entry_hash';
  IF v_sequence = v_length - 1 AND (v_round.journal -> (v_length - 1) ->> 'entry_hash') = v_hash THEN
    RETURN pg_catalog.jsonb_build_object('status', 'existing', 'journal_length', v_length, 'journal_head_hash', v_hash);
  END IF;
  IF v_sequence <> v_length OR v_prev <> v_round.journal_head_hash THEN
    RAISE EXCEPTION 'lab_arena_journal_chain_mismatch' USING ERRCODE = 'P0004';
  END IF;
  UPDATE public.lab_arena_rounds
  SET journal = journal || pg_catalog.jsonb_build_array(p_entry), journal_head_hash = v_hash
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object('status', 'appended', 'journal_length', v_length + 1, 'journal_head_hash', v_hash);
END;
$lab_arena_append_journal_entry$;
ALTER FUNCTION public.lab_arena_append_journal_entry(TEXT, JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_register_submission(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_register_submission$
DECLARE
  v_round public.lab_arena_rounds;
  v_existing public.lab_arena_submissions;
BEGIN
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
     OR pg_catalog.jsonb_typeof(p_doc) IS DISTINCT FROM 'object'
     OR (p_doc ->> 'submitted_digest') !~ '^sha256:[0-9a-f]{64}$'
     OR pg_catalog.char_length(COALESCE(p_doc ->> 'submitted_reference', '')) NOT BETWEEN 1 AND 512 THEN
    RAISE EXCEPTION 'lab_arena_submission_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'window_closed', 'round_status', v_round.status);
  END IF;
  SELECT * INTO v_existing FROM public.lab_arena_submissions WHERE submission_id = p_submission_id;
  IF FOUND THEN
    IF v_existing.round_id = p_round_id AND v_existing.miner_hotkey = p_miner_hotkey
       AND v_existing.submitted_digest = (p_doc ->> 'submitted_digest') THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing', 'submission_status', v_existing.status);
    END IF;
    RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
  END IF;
  INSERT INTO public.lab_arena_submissions (
    submission_id, round_id, miner_hotkey, status, is_king, submitted_reference, submitted_digest, consent, submission_doc
  ) VALUES (
    p_submission_id, p_round_id, p_miner_hotkey, 'uploaded',
    COALESCE((p_doc ->> 'is_king')::BOOLEAN, FALSE),
    p_doc ->> 'submitted_reference', p_doc ->> 'submitted_digest', p_doc -> 'consent', p_doc
  );
  RETURN pg_catalog.jsonb_build_object('status', 'registered', 'submission_status', 'uploaded');
END;
$lab_arena_register_submission$;
ALTER FUNCTION public.lab_arena_register_submission(TEXT, TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;

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
BEGIN
  IF pg_catalog.jsonb_typeof(v_patch) <> 'object' THEN
    RAISE EXCEPTION 'lab_arena_patch_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'window_closed', 'round_status', v_round.status);
  END IF;
  SELECT * INTO v_submission FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id AND round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_submission.status <> p_expected_status THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'submission_status', v_submission.status);
  END IF;
  BEGIN
    IF p_expected_status = 'uploaded' AND p_next_status = 'accepted' THEN
      -- Acceptance pins the resolved image: its digest, the Arena repository
      -- reference runners pull, and the process from the image config.
      IF (v_patch ->> 'image_digest') !~ '^sha256:[0-9a-f]{64}$'
         OR pg_catalog.char_length(COALESCE(v_patch ->> 'image_reference', '')) NOT BETWEEN 1 AND 512
         OR pg_catalog.jsonb_typeof(v_patch -> 'entry_command') IS DISTINCT FROM 'array' THEN
        RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
      END IF;
      IF pg_catalog.jsonb_array_length(v_patch -> 'entry_command') < 1 THEN
        RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
      END IF;
      UPDATE public.lab_arena_submissions
      SET status = 'accepted',
          image_digest = v_patch ->> 'image_digest',
          image_reference = v_patch ->> 'image_reference',
          submitted_reference = COALESCE(v_patch ->> 'submitted_reference', submitted_reference),
          submitted_digest = COALESCE(v_patch ->> 'submitted_digest', submitted_digest),
          entry_command = v_patch -> 'entry_command',
          image_environment = COALESCE(v_patch -> 'image_environment', '{}'::JSONB),
          working_dir = COALESCE(v_patch ->> 'working_dir', ''),
          image_size_bytes = (v_patch ->> 'image_size_bytes')::BIGINT,
          is_king = COALESCE((v_patch ->> 'is_king')::BOOLEAN, is_king)
      WHERE submission_id = p_submission_id;
    ELSIF p_expected_status IN ('uploaded', 'accepted') AND p_next_status = 'rejected' THEN
      -- An accepted submission is rejected only at freeze (preflight or capacity), always under a published rule.
      IF COALESCE(v_patch ->> 'rejection_rule', '') = '' THEN
        RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
      END IF;
      UPDATE public.lab_arena_submissions
      SET status = 'rejected',
          rejection_rule = v_patch ->> 'rejection_rule'
      WHERE submission_id = p_submission_id;
    ELSIF p_expected_status = 'accepted' AND p_next_status = 'frozen' THEN
      UPDATE public.lab_arena_submissions
      SET status = 'frozen', frozen_at = pg_catalog.clock_timestamp(),
          is_king = COALESCE((v_patch ->> 'is_king')::BOOLEAN, is_king)
      WHERE submission_id = p_submission_id;
    ELSE
      RAISE EXCEPTION 'lab_arena_transition_invalid' USING ERRCODE = '22023';
    END IF;
  EXCEPTION WHEN unique_violation THEN
    RETURN pg_catalog.jsonb_build_object('status', 'duplicate_artifact', 'submission_status', v_submission.status);
  END;
  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'submission_status', p_next_status);
END;
$lab_arena_update_submission$;
ALTER FUNCTION public.lab_arena_update_submission(TEXT, TEXT, TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__account_view(p_account public.lab_arena_accounts)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, public
AS $lab_arena__account_view$
  SELECT pg_catalog.jsonb_build_object(
    'miner_hotkey', p_account.miner_hotkey,
    'credentials', (
      SELECT COALESCE(pg_catalog.jsonb_object_agg(entry.key, (entry.value - 'ciphertext') || pg_catalog.jsonb_build_object('has_key', entry.value ? 'ciphertext')), '{}'::JSONB)
      FROM pg_catalog.jsonb_each(p_account.credentials) AS entry
    ),
    'preflight_status', p_account.preflight_status,
    'observed_limit_microusd', p_account.observed_limit_microusd,
    'observed_limit_remaining_microusd', p_account.observed_limit_remaining_microusd,
    'observed_usage_microusd', p_account.observed_usage_microusd,
    'observed_at', p_account.observed_at,
    'outstanding_openrouter_reservation_microusd', p_account.outstanding_openrouter_reservation_microusd,
    'settled_since_preflight_microusd', p_account.settled_since_preflight_microusd
  );
$lab_arena__account_view$;
ALTER FUNCTION public.lab_arena__account_view(public.lab_arena_accounts) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_upsert_account_credential(
  p_miner_hotkey TEXT,
  p_provider TEXT,
  p_ciphertext TEXT,
  p_key_hash TEXT,
  p_preflight JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_upsert_account_credential$
DECLARE
  v_account public.lab_arena_accounts;
  v_entry JSONB;
BEGIN
  IF COALESCE(p_miner_hotkey, '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
     OR p_provider NOT IN ('scrapingdog', 'deepline', 'openrouter')
     OR COALESCE(p_ciphertext, '') = '' OR pg_catalog.octet_length(p_ciphertext) > 8192
     OR COALESCE(p_key_hash, '') !~ '^[0-9a-f]{64}$'
     OR pg_catalog.jsonb_typeof(p_preflight) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_preflight::TEXT) > 8192
     OR (p_preflight ->> 'preflight_status') NOT IN ('ok', 'failed') THEN
    RAISE EXCEPTION 'lab_arena_credential_input_invalid' USING ERRCODE = '22023';
  END IF;
  INSERT INTO public.lab_arena_accounts (miner_hotkey) VALUES (p_miner_hotkey) ON CONFLICT DO NOTHING;
  SELECT * INTO v_account FROM public.lab_arena_accounts WHERE miner_hotkey = p_miner_hotkey FOR UPDATE;
  v_entry := pg_catalog.jsonb_build_object(
    'ciphertext', p_ciphertext,
    'key_hash', p_key_hash,
    'preflight_status', p_preflight ->> 'preflight_status',
    'preflight', p_preflight - 'key_hash',
    'observed_at', pg_catalog.clock_timestamp()
  );
  UPDATE public.lab_arena_accounts
  SET credentials = credentials || pg_catalog.jsonb_build_object(p_provider, v_entry),
      preflight_status = public.lab_arena__aggregate_preflight(credentials || pg_catalog.jsonb_build_object(p_provider, v_entry)),
      observed_limit_microusd = CASE WHEN p_provider = 'openrouter' THEN (p_preflight ->> 'limit_microusd')::BIGINT ELSE observed_limit_microusd END,
      observed_limit_remaining_microusd = CASE WHEN p_provider = 'openrouter' THEN (p_preflight ->> 'limit_remaining_microusd')::BIGINT ELSE observed_limit_remaining_microusd END,
      observed_usage_microusd = CASE WHEN p_provider = 'openrouter' THEN (p_preflight ->> 'usage_microusd')::BIGINT ELSE observed_usage_microusd END,
      observed_at = CASE WHEN p_provider = 'openrouter' THEN pg_catalog.clock_timestamp() ELSE observed_at END,
      settled_since_preflight_microusd = CASE WHEN p_provider = 'openrouter' THEN 0 ELSE settled_since_preflight_microusd END
  WHERE miner_hotkey = p_miner_hotkey
  RETURNING * INTO v_account;
  RETURN pg_catalog.jsonb_build_object('status', 'ok') || public.lab_arena__account_view(v_account);
END;
$lab_arena_upsert_account_credential$;
ALTER FUNCTION public.lab_arena_upsert_account_credential(TEXT, TEXT, TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_record_preflight(
  p_miner_hotkey TEXT,
  p_provider TEXT,
  p_preflight JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_record_preflight$
DECLARE
  v_account public.lab_arena_accounts;
  v_entry JSONB;
BEGIN
  IF p_provider NOT IN ('scrapingdog', 'deepline', 'openrouter')
     OR pg_catalog.jsonb_typeof(p_preflight) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_preflight::TEXT) > 8192
     OR (p_preflight ->> 'preflight_status') NOT IN ('ok', 'failed') THEN
    RAISE EXCEPTION 'lab_arena_preflight_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_account FROM public.lab_arena_accounts WHERE miner_hotkey = p_miner_hotkey FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_account_missing' USING ERRCODE = 'P0002';
  END IF;
  v_entry := v_account.credentials -> p_provider;
  IF v_entry IS NULL OR (v_entry ->> 'key_hash') IS DISTINCT FROM (p_preflight ->> 'key_hash') THEN
    RAISE EXCEPTION 'lab_arena_preflight_key_mismatch' USING ERRCODE = '22023';
  END IF;
  v_entry := v_entry || pg_catalog.jsonb_build_object(
    'preflight_status', p_preflight ->> 'preflight_status',
    'preflight', p_preflight - 'key_hash',
    'observed_at', pg_catalog.clock_timestamp()
  );
  UPDATE public.lab_arena_accounts
  SET credentials = credentials || pg_catalog.jsonb_build_object(p_provider, v_entry),
      preflight_status = public.lab_arena__aggregate_preflight(credentials || pg_catalog.jsonb_build_object(p_provider, v_entry)),
      observed_limit_microusd = CASE WHEN p_provider = 'openrouter' THEN (p_preflight ->> 'limit_microusd')::BIGINT ELSE observed_limit_microusd END,
      observed_limit_remaining_microusd = CASE WHEN p_provider = 'openrouter' THEN (p_preflight ->> 'limit_remaining_microusd')::BIGINT ELSE observed_limit_remaining_microusd END,
      observed_usage_microusd = CASE WHEN p_provider = 'openrouter' THEN (p_preflight ->> 'usage_microusd')::BIGINT ELSE observed_usage_microusd END,
      observed_at = CASE WHEN p_provider = 'openrouter' THEN pg_catalog.clock_timestamp() ELSE observed_at END,
      settled_since_preflight_microusd = CASE WHEN p_provider = 'openrouter' THEN 0 ELSE settled_since_preflight_microusd END
  WHERE miner_hotkey = p_miner_hotkey
  RETURNING * INTO v_account;
  RETURN pg_catalog.jsonb_build_object('status', 'ok') || public.lab_arena__account_view(v_account);
END;
$lab_arena_record_preflight$;
ALTER FUNCTION public.lab_arena_record_preflight(TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;


-- ---------------------------------------------------------------------------
-- Stages, claims, provider calls, events, completion, expiry, close, cancel
-- ---------------------------------------------------------------------------

-- Open a stage: compare-and-set the round status, bump the stage generation,
-- and create one pending attempt-1 row per participant and ICP position, or a
-- service-created preflight_failed terminal record (attempt 0) for a king
-- that failed funding or key preflight (section 7.1).
CREATE OR REPLACE FUNCTION public.lab_arena_open_stage(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_participants JSONB,
  p_icp_positions INTEGER[],
  p_icp_hashes TEXT[]
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_open_stage$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected TEXT;
  v_next TEXT;
  v_generation BIGINT;
  v_participant JSONB;
  v_submission public.lab_arena_submissions;
  v_index INTEGER;
  v_position INTEGER;
  v_assignment TEXT;
  v_created INTEGER := 0;
  v_preflight_failed BOOLEAN;
BEGIN
  IF p_stage NOT IN (1, 2)
     OR pg_catalog.jsonb_typeof(p_participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_participants) < 1
     OR p_icp_positions IS NULL OR p_icp_hashes IS NULL
     OR pg_catalog.array_length(p_icp_positions, 1) IS DISTINCT FROM pg_catalog.array_length(p_icp_hashes, 1) THEN
    RAISE EXCEPTION 'lab_arena_stage_input_invalid' USING ERRCODE = '22023';
  END IF;
  v_expected := CASE p_stage WHEN 1 THEN 'committed' ELSE 'stage1_scored' END;
  v_next := 'stage' || p_stage::TEXT;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> v_expected THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_participant IN SELECT value FROM pg_catalog.jsonb_array_elements(p_participants) LOOP
    SELECT * INTO v_submission FROM public.lab_arena_submissions
    WHERE submission_id = (v_participant ->> 'submission_id') AND round_id = p_round_id AND status = 'frozen';
    IF NOT FOUND OR v_submission.miner_hotkey <> (v_participant ->> 'miner_hotkey') THEN
      RAISE EXCEPTION 'lab_arena_participant_not_frozen' USING ERRCODE = '23503';
    END IF;
    v_preflight_failed := COALESCE((v_participant ->> 'preflight_failed')::BOOLEAN, FALSE);
    FOR v_index IN 1 .. pg_catalog.array_length(p_icp_positions, 1) LOOP
      v_position := p_icp_positions[v_index];
      IF (p_stage = 1 AND v_position NOT BETWEEN 0 AND 19)
         OR (p_stage = 2 AND v_position NOT BETWEEN 20 AND 49)
         OR p_icp_hashes[v_index] !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'lab_arena_stage_position_invalid' USING ERRCODE = '22023';
      END IF;
      v_assignment := p_round_id || ':' || v_submission.submission_id || ':' || p_stage::TEXT || ':' || v_position::TEXT;
      IF v_preflight_failed THEN
        INSERT INTO public.lab_arena_runs (
          run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, icp_position, icp_hash,
          attempt, status, stage_generation, terminal_cause, terminal_doc
        ) VALUES (
          v_assignment || ':0', v_assignment, p_round_id, v_submission.submission_id, v_submission.miner_hotkey,
          p_stage, v_position, p_icp_hashes[v_index], 0, 'failed', v_generation, 'preflight_failed',
          pg_catalog.jsonb_build_object('service_created', TRUE)
        );
      ELSE
        INSERT INTO public.lab_arena_runs (
          run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, icp_position, icp_hash,
          attempt, status, stage_generation
        ) VALUES (
          v_assignment || ':1', v_assignment, p_round_id, v_submission.submission_id, v_submission.miner_hotkey,
          p_stage, v_position, p_icp_hashes[v_index], 1, 'pending', v_generation
        );
      END IF;
      v_created := v_created + 1;
    END LOOP;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = v_next, status_generation = status_generation + 1, stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'round_status', v_next,
    'stage_generation', v_generation, 'assignments', v_created);
END;
$lab_arena_open_stage$;
ALTER FUNCTION public.lab_arena_open_stage(TEXT, SMALLINT, JSONB, INTEGER[], TEXT[]) OWNER TO lab_arena_owner;

-- Claim the next pending ICP assignment (section 9.1). Replaying the same
-- request id returns the stored response; a reused id with different bytes
-- is rejected; ICP-major order; self-execution excluded; slot ceiling.
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
BEGIN
  IF COALESCE(p_runner_hotkey, '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
     OR COALESCE(p_declared_parallelism, 0) < 1 OR COALESCE(p_slot_ceiling, 0) < 1
     OR COALESCE(p_request_id, '') !~ '^[0-9a-f]{32}$'
     OR COALESCE(p_request_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_lease_token_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_claim_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  -- Serialize replays of one request id (after the round lock, before the run lock).
  PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('lab_arena.claim:' || p_request_id, 0));
  SELECT * INTO v_existing FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND claim_request_id = p_request_id;
  IF FOUND THEN
    IF v_existing.claim_request_hash = p_request_hash THEN
      RETURN v_existing.claim_response;
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'request_id_reused');
  END IF;
  IF v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage2_scoring') THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stage_closed', 'round_status', v_round.status);
  END IF;
  v_stage := CASE WHEN v_round.status IN ('stage1', 'stage1_scoring') THEN 1 ELSE 2 END;
  IF NOT (v_round.configuration_doc -> 'runner_allowlist' ? p_runner_hotkey) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_allowlisted');
  END IF;
  v_limit := LEAST(p_declared_parallelism, p_slot_ceiling);
  SELECT COUNT(*) INTO v_active FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND runner_hotkey = p_runner_hotkey AND status = 'leased'
    AND lease_expires_at > pg_catalog.clock_timestamp()
    AND stage_generation = v_round.stage_generation;
  IF v_active >= v_limit THEN
    RETURN pg_catalog.jsonb_build_object('status', 'no_free_slot', 'active_leases', v_active, 'slot_limit', v_limit);
  END IF;
  -- Only the current phase's assignments are pending at its stage generation.
  -- Any allowlisted validator scores any item, its own executions included:
  -- the Arena replays every scoring from the recorded judge responses, which
  -- is the single integrity check, so one validator is enough.
  SELECT * INTO v_run FROM public.lab_arena_runs AS runs
  WHERE runs.round_id = p_round_id AND runs.stage = v_stage AND runs.status = 'pending'
    AND runs.stage_generation = v_round.stage_generation
    AND runs.miner_hotkey <> ALL (COALESCE(p_excluded_miner_hotkeys, ARRAY[]::TEXT[]))
    -- A confirmation attempt goes to a different validator while another one
    -- is active in this round; a lone active validator confirms its own.
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs AS others
           WHERE others.round_id = p_round_id AND others.runner_hotkey IS NOT NULL
             AND others.runner_hotkey <> p_runner_hotkey))
  ORDER BY runs.icp_position, runs.created_at, runs.assignment_id
  FOR UPDATE SKIP LOCKED
  LIMIT 1;
  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status', 'no_pending');
  END IF;
  SELECT * INTO v_submission FROM public.lab_arena_submissions WHERE submission_id = v_run.submission_id;
  v_expires := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  v_response := pg_catalog.jsonb_build_object(
    'status', 'leased',
    'request_id', p_request_id,
    'run_id', v_run.run_id,
    'assignment_id', v_run.assignment_id,
    'submission_id', v_run.submission_id,
    'miner_hotkey', v_run.miner_hotkey,
    'image_digest', v_submission.image_digest,
    'stage', v_run.stage,
    'icp_position', v_run.icp_position,
    'icp_hash', v_run.icp_hash,
    'attempt', v_run.attempt,
    'kind', v_run.kind,
    'work_item_id', v_run.work_item_id,
    'scored_run_id', v_run.scored_run_id,
    'lease_generation', v_run.lease_generation + 1,
    'stage_generation', v_round.stage_generation,
    'lease_expires_at', v_expires,
    'event_cursor', v_run.event_cursor,
    'event_head_hash', v_run.event_head_hash
  );
  UPDATE public.lab_arena_runs
  SET status = 'leased', runner_hotkey = p_runner_hotkey, lease_token_hash = p_lease_token_hash,
      lease_generation = lease_generation + 1, lease_expires_at = v_expires,
      claim_request_id = p_request_id, claim_request_hash = p_request_hash, claim_response = v_response
  WHERE run_id = v_run.run_id;
  RETURN v_response;
END;
$lab_arena_claim_assignment$;
ALTER FUNCTION public.lab_arena_claim_assignment(TEXT, TEXT, INTEGER, INTEGER, TEXT[], TEXT, TEXT, TEXT, INTEGER) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__call_state_view(p_head public.lab_arena_ledger, p_run public.lab_arena_runs)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, public
AS $lab_arena__call_state_view$
  SELECT pg_catalog.jsonb_build_object(
    'status', CASE p_head.entry_kind
      WHEN 'reservation' THEN 'reserved'
      WHEN 'dispatch' THEN 'dispatched'
      WHEN 'settlement' THEN 'settled'
      WHEN 'uncertain' THEN 'uncertain'
      WHEN 'recovery' THEN 'recovered'
      WHEN 'refusal' THEN 'refused'
      ELSE 'unknown' END,
    'idempotent', TRUE,
    'call_identity', p_head.call_identity,
    'amount_microusd', p_head.amount_microusd,
    'terminal_response', p_head.terminal_response,
    'reason', p_head.entry_doc ->> 'reason',
    'event_cursor', p_run.event_cursor,
    'event_head_hash', p_run.event_head_hash,
    'lease_expires_at', p_run.lease_expires_at
  );
$lab_arena__call_state_view$;
ALTER FUNCTION public.lab_arena__call_state_view(public.lab_arena_ledger, public.lab_arena_runs) OWNER TO lab_arena_owner;

-- Reserve one provider call (section 7.5 steps 1-3). Refusals are recorded
-- under the call identity so a retry returns the stored refusal.
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
  v_account public.lab_arena_accounts;
  v_quota INTEGER;
  v_stage_quota BIGINT;
  v_consumed BIGINT;
  v_capacity BIGINT;
  v_reason TEXT := NULL;
  v_expires TIMESTAMPTZ;
BEGIN
  IF COALESCE(p_call_identity, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_operation_id, '') !~ '^[a-z0-9_.]{1,64}$'
     OR p_provider NOT IN ('scrapingdog', 'deepline', 'openrouter')
     OR p_funding_source IS DISTINCT FROM 'miner_key'
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
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NOT NULL THEN
    IF v_head.run_id <> p_run_id THEN
      RAISE EXCEPTION 'lab_arena_call_identity_foreign' USING ERRCODE = '23505';
    END IF;
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = v_run.round_id;
  -- Fairness is a fixed call quota per provider (section 7 as revised): the
  -- per-ICP quota bounds this attempt, the stage quota (per-ICP quota times
  -- the stage's ICP count times the attempt limit) bounds the participant.
  v_quota := CASE v_run.kind
    WHEN 'score' THEN ((v_round.configuration_doc -> 'scoring_call_quotas') ->> p_provider)::INTEGER
    ELSE ((v_round.configuration_doc -> 'call_quotas') ->> p_provider)::INTEGER END;
  IF v_quota IS NULL OR v_quota < 1 THEN
    RAISE EXCEPTION 'lab_arena_quota_missing' USING ERRCODE = '22023';
  END IF;
  v_stage_quota := v_quota::BIGINT
    * (CASE v_run.stage WHEN 1 THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::BIGINT ELSE (v_round.configuration_doc ->> 'stage_2_icp_count')::BIGINT END)
    * (v_round.configuration_doc ->> 'max_attempts_per_assignment')::BIGINT;

  -- Per-ICP quota on this attempt.
  v_consumed := public.lab_arena__run_consumed(p_run_id, p_provider);
  IF v_consumed >= v_quota THEN
    v_reason := 'per_icp_quota';
  END IF;
  -- Stage quota on the participant (lock order: participant budget after
  -- assignment). FOR NO KEY UPDATE serializes reservations of one
  -- participant against each other while staying compatible with the
  -- FOR KEY SHARE locks that run-row updates take on the submission through
  -- the foreign key; a FOR UPDATE lock here deadlocks against settlement.
  IF v_reason IS NULL THEN
    SELECT * INTO v_submission FROM public.lab_arena_submissions WHERE submission_id = v_run.submission_id FOR NO KEY UPDATE;
    v_consumed := public.lab_arena__submission_stage_consumed(v_run.submission_id, v_run.stage, p_provider, v_run.kind);
    IF v_consumed >= v_stage_quota THEN
      v_reason := 'stage_quota';
    END IF;
  END IF;
  -- The miner's key for this provider (lock order: account after participant budget).
  IF v_reason IS NULL THEN
    SELECT * INTO v_account FROM public.lab_arena_accounts WHERE miner_hotkey = v_run.miner_hotkey FOR UPDATE;
    IF NOT FOUND THEN
      v_reason := 'no_account';
    ELSIF v_account.preflight_status <> 'ok' OR NOT (v_account.credentials ? p_provider)
          OR (v_account.credentials -> p_provider ->> 'ciphertext') IS NULL THEN
      v_reason := 'key_preflight';
    ELSIF p_provider = 'openrouter' THEN
      v_capacity := CASE
        WHEN v_account.observed_limit_remaining_microusd IS NULL THEN NULL
        ELSE v_account.observed_limit_remaining_microusd
             - v_account.outstanding_openrouter_reservation_microusd
             - v_account.settled_since_preflight_microusd END;
      IF v_capacity IS NOT NULL AND v_capacity < p_amount_microusd THEN
        v_reason := 'key_capacity';
      ELSE
        UPDATE public.lab_arena_accounts
        SET outstanding_openrouter_reservation_microusd = outstanding_openrouter_reservation_microusd + p_amount_microusd
        WHERE miner_hotkey = v_run.miner_hotkey;
      END IF;
    END IF;
  END IF;

  v_expires := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  IF v_reason IS NOT NULL THEN
    INSERT INTO public.lab_arena_ledger (
      entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
      provider, operation_id, funding_source, amount_microusd, entry_doc
    ) VALUES (
      'refusal', v_run.miner_hotkey, v_run.round_id, v_run.submission_id, p_run_id, v_run.stage,
      p_call_identity, p_provider, p_operation_id, p_funding_source, 0,
      pg_catalog.jsonb_build_object('reason', v_reason, 'requested_microusd', p_amount_microusd, 'call', p_call_doc)
    );
    UPDATE public.lab_arena_runs SET lease_expires_at = v_expires WHERE run_id = p_run_id;
    RETURN pg_catalog.jsonb_build_object('status', 'refused', 'idempotent', FALSE, 'reason', v_reason,
      'call_identity', p_call_identity, 'event_cursor', v_run.event_cursor,
      'event_head_hash', v_run.event_head_hash, 'lease_expires_at', v_expires);
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
    provider, operation_id, funding_source, amount_microusd, entry_doc
  ) VALUES (
    'reservation', v_run.miner_hotkey, v_run.round_id, v_run.submission_id, p_run_id, v_run.stage,
    p_call_identity, p_provider, p_operation_id, p_funding_source, p_amount_microusd, p_call_doc
  );
  UPDATE public.lab_arena_runs SET lease_expires_at = v_expires WHERE run_id = p_run_id;
  RETURN pg_catalog.jsonb_build_object('status', 'reserved', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'amount_microusd', p_amount_microusd,
    'event_cursor', v_run.event_cursor, 'event_head_hash', v_run.event_head_hash,
    'lease_expires_at', v_expires);
END;
$lab_arena_reserve_call$;
ALTER FUNCTION public.lab_arena_reserve_call(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER) OWNER TO lab_arena_owner;

-- The dispatch marker (section 7.5 step 4). Committed before the request.
CREATE OR REPLACE FUNCTION public.lab_arena_mark_dispatched(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_call_identity TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_mark_dispatched$
DECLARE
  v_run public.lab_arena_runs;
  v_head public.lab_arena_ledger;
BEGIN
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NULL OR v_head.run_id <> p_run_id THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_reserved');
  END IF;
  IF v_head.entry_kind <> 'reservation' THEN
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
    provider, operation_id, funding_source, amount_microusd, entry_doc
  ) VALUES (
    'dispatch', v_head.miner_hotkey, v_head.round_id, v_head.submission_id, p_run_id, v_head.stage,
    p_call_identity, v_head.provider, v_head.operation_id, v_head.funding_source, v_head.amount_microusd,
    pg_catalog.jsonb_build_object('dispatched_at', pg_catalog.clock_timestamp())
  );
  RETURN pg_catalog.jsonb_build_object('status', 'dispatched', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'amount_microusd', v_head.amount_microusd,
    'event_cursor', v_run.event_cursor, 'event_head_hash', v_run.event_head_hash,
    'lease_expires_at', v_run.lease_expires_at);
END;
$lab_arena_mark_dispatched$;
ALTER FUNCTION public.lab_arena_mark_dispatched(TEXT, TEXT, TEXT) OWNER TO lab_arena_owner;

-- Settle a dispatched call at its actual cost (never above the reservation),
-- store the terminal sanitized response, append the worker's call-summary
-- event in the same transaction, and renew the lease (section 7.5 step 5).
CREATE OR REPLACE FUNCTION public.lab_arena_settle_call(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_call_identity TEXT,
  p_actual_microusd BIGINT,
  p_terminal_response JSONB,
  p_event JSONB,
  p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_settle_call$
DECLARE
  v_run public.lab_arena_runs;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_expires TIMESTAMPTZ;
BEGIN
  IF COALESCE(p_actual_microusd, -1) < 0
     OR pg_catalog.jsonb_typeof(p_terminal_response) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_terminal_response::TEXT) > 4194304
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_settle_input_invalid' USING ERRCODE = '22023';
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NULL OR v_head.run_id <> p_run_id THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_reserved');
  END IF;
  IF v_head.entry_kind <> 'dispatch' THEN
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  SELECT * INTO v_reservation FROM public.lab_arena_ledger
  WHERE call_identity = p_call_identity AND entry_kind = 'reservation';
  -- OpenRouter reservations bound the key capacity, so a settlement may not
  -- exceed them. Other providers reserve no amount and settle at whatever the
  -- provider reported (or zero), which is informational.
  IF v_reservation.provider = 'openrouter' AND p_actual_microusd > v_reservation.amount_microusd THEN
    RAISE EXCEPTION 'lab_arena_settlement_exceeds_reservation' USING ERRCODE = '23514';
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
    provider, operation_id, funding_source, amount_microusd, entry_doc, terminal_response
  ) VALUES (
    'settlement', v_reservation.miner_hotkey, v_reservation.round_id, v_reservation.submission_id, p_run_id,
    v_reservation.stage, p_call_identity, v_reservation.provider, v_reservation.operation_id,
    v_reservation.funding_source, p_actual_microusd,
    pg_catalog.jsonb_build_object('reserved_microusd', v_reservation.amount_microusd,
      'released_microusd', v_reservation.amount_microusd - p_actual_microusd),
    p_terminal_response
  );
  PERFORM public.lab_arena__apply_terminal_funding(v_reservation, 'settlement', p_actual_microusd);
  IF p_event IS NOT NULL THEN
    v_run := public.lab_arena__append_event_locked(v_run, p_event);
  END IF;
  v_expires := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  UPDATE public.lab_arena_runs SET lease_expires_at = v_expires WHERE run_id = p_run_id;
  RETURN pg_catalog.jsonb_build_object('status', 'settled', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'actual_microusd', p_actual_microusd,
    'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
    'terminal_response', p_terminal_response,
    'event_cursor', v_run.event_cursor, 'event_head_hash', v_run.event_head_hash,
    'lease_expires_at', v_expires);
END;
$lab_arena_settle_call$;
ALTER FUNCTION public.lab_arena_settle_call(TEXT, TEXT, TEXT, BIGINT, JSONB, JSONB, INTEGER) OWNER TO lab_arena_owner;

-- Mark a dispatched call uncertain at its full reservation (section 7.5).
CREATE OR REPLACE FUNCTION public.lab_arena_mark_uncertain(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_call_identity TEXT,
  p_call_doc JSONB,
  p_event JSONB,
  p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_mark_uncertain$
DECLARE
  v_run public.lab_arena_runs;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_expires TIMESTAMPTZ;
BEGIN
  IF COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_uncertain_input_invalid' USING ERRCODE = '22023';
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NULL OR v_head.run_id <> p_run_id THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_reserved');
  END IF;
  IF v_head.entry_kind <> 'dispatch' THEN
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  SELECT * INTO v_reservation FROM public.lab_arena_ledger
  WHERE call_identity = p_call_identity AND entry_kind = 'reservation';
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
    provider, operation_id, funding_source, amount_microusd, entry_doc
  ) VALUES (
    'uncertain', v_reservation.miner_hotkey, v_reservation.round_id, v_reservation.submission_id, p_run_id,
    v_reservation.stage, p_call_identity, v_reservation.provider, v_reservation.operation_id,
    v_reservation.funding_source, v_reservation.amount_microusd,
    pg_catalog.jsonb_build_object('reason', 'worker_reported', 'call', COALESCE(p_call_doc, '{}'::JSONB))
  );
  PERFORM public.lab_arena__apply_terminal_funding(v_reservation, 'uncertain', v_reservation.amount_microusd);
  IF p_event IS NOT NULL THEN
    v_run := public.lab_arena__append_event_locked(v_run, p_event);
  END IF;
  v_expires := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  UPDATE public.lab_arena_runs SET lease_expires_at = v_expires WHERE run_id = p_run_id;
  RETURN pg_catalog.jsonb_build_object('status', 'uncertain', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'amount_microusd', v_reservation.amount_microusd,
    'event_cursor', v_run.event_cursor, 'event_head_hash', v_run.event_head_hash,
    'lease_expires_at', v_expires);
END;
$lab_arena_mark_uncertain$;
ALTER FUNCTION public.lab_arena_mark_uncertain(TEXT, TEXT, TEXT, JSONB, JSONB, INTEGER) OWNER TO lab_arena_owner;

-- Append one ordered batch of worker events (section 10). A replayed batch
-- whose hashes already sit at those sequences is idempotent.
CREATE OR REPLACE FUNCTION public.lab_arena_append_events(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_events JSONB,
  p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_append_events$
DECLARE
  v_run public.lab_arena_runs;
  v_event JSONB;
  v_sequence INTEGER;
  v_stored TEXT;
  v_appended INTEGER := 0;
  v_expires TIMESTAMPTZ;
BEGIN
  IF pg_catalog.jsonb_typeof(p_events) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_events) < 1
     OR pg_catalog.jsonb_array_length(p_events) > 256
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_events_input_invalid' USING ERRCODE = '22023';
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  FOR v_event IN SELECT value FROM pg_catalog.jsonb_array_elements(p_events) LOOP
    v_sequence := (v_event ->> 'sequence')::INTEGER;
    IF v_sequence < v_run.event_cursor THEN
      SELECT event_hash INTO v_stored FROM public.lab_arena_events
      WHERE run_id = p_run_id AND sequence = v_sequence;
      IF v_stored IS DISTINCT FROM (v_event ->> 'event_hash') THEN
        RAISE EXCEPTION 'lab_arena_event_sequence' USING ERRCODE = 'P0004';
      END IF;
      CONTINUE;
    END IF;
    v_run := public.lab_arena__append_event_locked(v_run, v_event);
    v_appended := v_appended + 1;
  END LOOP;
  v_expires := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  UPDATE public.lab_arena_runs SET lease_expires_at = v_expires WHERE run_id = p_run_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE WHEN v_appended > 0 THEN 'appended' ELSE 'existing' END,
    'appended', v_appended,
    'event_cursor', v_run.event_cursor, 'event_head_hash', v_run.event_head_hash,
    'lease_expires_at', v_expires);
END;
$lab_arena_append_events$;
ALTER FUNCTION public.lab_arena_append_events(TEXT, TEXT, JSONB, INTEGER) OWNER TO lab_arena_owner;

-- Store one validated receipt (section 9.4). The service verifies the
-- signature, roots, and output contract before calling; the database
-- requires a current lease and closed provider accounting.
CREATE OR REPLACE FUNCTION public.lab_arena_complete_attempt(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_receipt JSONB,
  p_receipt_hash TEXT,
  p_terminal_cause TEXT,
  p_output_hash TEXT,
  p_output_ref TEXT,
  p_provider_call_root TEXT,
  p_private_event_root TEXT,
  p_cost_root TEXT
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
  IF pg_catalog.jsonb_typeof(p_receipt) IS DISTINCT FROM 'object'
     OR COALESCE(p_receipt_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR (p_receipt ->> 'receipt_hash') IS DISTINCT FROM p_receipt_hash
     OR p_terminal_cause NOT IN ('accepted', 'model_timeout', 'invalid_output', 'budget_exhausted', 'model_error',
                                 'judge_error', 'judge_timeout', 'judge_key_refused')
     OR (p_terminal_cause = 'accepted' AND COALESCE(p_output_hash, '') !~ '^sha256:[0-9a-f]{64}$')
     OR COALESCE(p_provider_call_root, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_private_event_root, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_cost_root, '') !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'lab_arena_complete_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_existing FROM public.lab_arena_runs WHERE run_id = p_run_id;
  IF FOUND AND v_existing.status IN ('accepted', 'failed') AND v_existing.receipt_hash = p_receipt_hash THEN
    RETURN pg_catalog.jsonb_build_object('status', v_existing.status, 'idempotent', TRUE,
      'run_id', p_run_id, 'attempt', v_existing.attempt);
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  IF (v_run.kind = 'execute' AND p_terminal_cause IN ('judge_error', 'judge_timeout', 'judge_key_refused'))
     OR (v_run.kind = 'score' AND p_terminal_cause NOT IN ('accepted', 'judge_error', 'judge_timeout', 'judge_key_refused')) THEN
    RAISE EXCEPTION 'lab_arena_complete_cause_kind_mismatch' USING ERRCODE = '22023';
  END IF;
  SELECT COUNT(*) INTO v_open FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.run_id = p_run_id AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS heads WHERE heads.entry_kind IN ('reservation', 'dispatch');
  IF v_open > 0 THEN
    RETURN pg_catalog.jsonb_build_object('status', 'accounting_open', 'open_calls', v_open);
  END IF;
  v_status := CASE WHEN p_terminal_cause = 'accepted' THEN 'accepted' ELSE 'failed' END;
  UPDATE public.lab_arena_runs
  SET status = v_status,
      receipt_doc = p_receipt,
      receipt_hash = p_receipt_hash,
      terminal_cause = p_terminal_cause,
      output_hash = NULLIF(p_output_hash, ''),
      output_ref = NULLIF(p_output_ref, ''),
      provider_call_root = p_provider_call_root,
      private_event_root = p_private_event_root,
      cost_root = p_cost_root
  WHERE run_id = p_run_id;
  -- No single validator's word ends an attempt in a failure: every failure but
  -- the miner's own quota exhaustion or key refusal gets one confirmation
  -- attempt, claimable by a different validator when the round has more than
  -- one. A second failure stands.
  IF v_status = 'failed' AND p_terminal_cause NOT IN ('budget_exhausted', 'judge_key_refused') AND v_run.attempt < 2 THEN
    SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = v_run.round_id;
    IF v_run.stage_generation = v_round.stage_generation THEN
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, icp_position, icp_hash,
        attempt, status, lease_generation, stage_generation, kind, work_item_id, scored_run_id, previous_runner_hotkey
      ) VALUES (
        v_run.assignment_id || ':' || (v_run.attempt + 1)::TEXT, v_run.assignment_id, v_run.round_id,
        v_run.submission_id, v_run.miner_hotkey, v_run.stage, v_run.icp_position, v_run.icp_hash,
        v_run.attempt + 1, 'pending', v_run.lease_generation, v_round.stage_generation,
        v_run.kind, v_run.work_item_id, v_run.scored_run_id, v_run.runner_hotkey
      );
      RETURN pg_catalog.jsonb_build_object('status', v_status, 'idempotent', FALSE,
        'run_id', p_run_id, 'attempt', v_run.attempt, 'confirmation_attempt', v_run.attempt + 1);
    END IF;
  END IF;
  RETURN pg_catalog.jsonb_build_object('status', v_status, 'idempotent', FALSE,
    'run_id', p_run_id, 'attempt', v_run.attempt);
END;
$lab_arena_complete_attempt$;
ALTER FUNCTION public.lab_arena_complete_attempt(TEXT, TEXT, JSONB, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT) OWNER TO lab_arena_owner;

-- Expire leases using database time (section 9.1): recover undispatched
-- reservations once, keep dispatched spend charged as uncertain, terminate
-- the attempt, and create a second attempt with a fresh per-ICP cap when the
-- attempt limit allows.
CREATE OR REPLACE FUNCTION public.lab_arena_expire_leases(p_round_id TEXT)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_expire_leases$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_expired INTEGER := 0;
  v_retried INTEGER := 0;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage2_scoring') THEN
    RETURN pg_catalog.jsonb_build_object('status', 'no_stage', 'expired', 0, 'retried', 0);
  END IF;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status = 'leased'
      AND lease_expires_at <= pg_catalog.clock_timestamp()
    ORDER BY assignment_id
    FOR UPDATE
  LOOP
    PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'lease_expired');
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'lease_expired',
        terminal_doc = pg_catalog.jsonb_build_object('expired_at', pg_catalog.clock_timestamp())
    WHERE run_id = v_run.run_id;
    v_expired := v_expired + 1;
    IF v_run.attempt < 2 AND v_run.stage_generation = v_round.stage_generation THEN
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, icp_position, icp_hash,
        attempt, status, lease_generation, stage_generation, kind, work_item_id, scored_run_id
      ) VALUES (
        v_run.assignment_id || ':' || (v_run.attempt + 1)::TEXT, v_run.assignment_id, v_run.round_id,
        v_run.submission_id, v_run.miner_hotkey, v_run.stage, v_run.icp_position, v_run.icp_hash,
        v_run.attempt + 1, 'pending', v_run.lease_generation, v_round.stage_generation,
        v_run.kind, v_run.work_item_id, v_run.scored_run_id
      );
      v_retried := v_retried + 1;
    END IF;
  END LOOP;
  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'expired', v_expired, 'retried', v_retried);
END;
$lab_arena_expire_leases$;
ALTER FUNCTION public.lab_arena_expire_leases(TEXT) OWNER TO lab_arena_owner;

-- Atomic stage close under the section 2 rule: lock the round, invalidate
-- every lease generation, recover undispatched reservations, mark dispatched
-- calls uncertain, freeze the accepted result set, and cancel the round when
-- any assignment lacks an accepted result for an infrastructure reason.
CREATE OR REPLACE FUNCTION public.lab_arena_close_stage(p_round_id TEXT, p_stage SMALLINT)
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
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> ('stage' || p_stage::TEXT) THEN
    IF (p_stage = 1 AND v_round.status IN ('stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published'))
       OR (p_stage = 2 AND v_round.status IN ('stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published')) THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation);
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND stage = p_stage AND kind = 'execute' AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    IF v_run.status = 'leased' THEN
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object('closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status)
    WHERE run_id = v_run.run_id;
  END LOOP;
  -- An assignment is incomplete when no attempt was accepted and no attempt
  -- ended for a model-caused reason or a preflight failure. A confirmation
  -- attempt the window did not reach leaves the first failure standing.
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT runs.assignment_id
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'execute'
    GROUP BY runs.assignment_id
    HAVING bool_and(runs.status <> 'accepted')
       AND NOT bool_or(COALESCE(runs.terminal_cause, '') IN
         ('model_timeout', 'invalid_output', 'budget_exhausted', 'model_error', 'preflight_failed'))
  ) AS incomplete;
  IF v_incomplete > 0 THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled', status_generation = status_generation + 1, stage_generation = v_generation,
        cancel_reason = 'capacity:stage' || p_stage::TEXT || ':' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage' || p_stage::TEXT || '_closed';
    UPDATE public.lab_arena_rounds
    SET status = v_next, status_generation = status_generation + 1, stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE WHEN v_next = 'cancelled' THEN 'cancelled' ELSE 'closed' END,
    'round_status', v_next, 'incomplete_assignments', v_incomplete, 'stage_generation', v_generation);
END;
$lab_arena_close_stage$;
ALTER FUNCTION public.lab_arena_close_stage(TEXT, SMALLINT) OWNER TO lab_arena_owner;


-- Turn the committed scoring plan into claimable scoring assignments: one
-- pending score run per work item, bound to the accepted execution it judges
-- and accounted against that miner's keys. The round moves to
-- stageN_scoring at a new stage generation so only score runs are claimable.
CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_work_items JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_open_scoring$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected TEXT;
  v_next TEXT;
  v_generation BIGINT;
  v_item JSONB;
  v_scored public.lab_arena_runs;
  v_assignment TEXT;
  v_created INTEGER := 0;
BEGIN
  IF p_stage NOT IN (1, 2) OR pg_catalog.jsonb_typeof(p_work_items) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_scoring_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  v_expected := 'stage' || p_stage::TEXT || '_closed';
  v_next := 'stage' || p_stage::TEXT || '_scoring';
  IF v_round.status <> v_expected THEN
    IF (p_stage = 1 AND v_round.status IN ('stage1_scoring', 'stage1_judged', 'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published'))
       OR (p_stage = 2 AND v_round.status IN ('stage2_scoring', 'stage2_judged', 'scored', 'published')) THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation);
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation);
  END IF;
  IF (p_stage = 1 AND v_round.stage1_scoring_plan_hash IS NULL)
     OR (p_stage = 2 AND v_round.stage2_scoring_plan_hash IS NULL) THEN
    RAISE EXCEPTION 'lab_arena_scoring_plan_missing' USING ERRCODE = '22023';
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_item IN SELECT value FROM pg_catalog.jsonb_array_elements(p_work_items) LOOP
    IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
       OR COALESCE(v_item ->> 'work_item_id', '') !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(v_item ->> 'output_hash', '') !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(v_item ->> 'scored_run_id', '') = '' THEN
      RAISE EXCEPTION 'lab_arena_scoring_item_invalid' USING ERRCODE = '22023';
    END IF;
    SELECT * INTO v_scored FROM public.lab_arena_runs
    WHERE run_id = v_item ->> 'scored_run_id' AND round_id = p_round_id AND stage = p_stage
      AND kind = 'execute' AND status = 'accepted';
    IF NOT FOUND OR v_scored.output_hash IS DISTINCT FROM (v_item ->> 'output_hash') THEN
      RAISE EXCEPTION 'lab_arena_scored_run_invalid' USING ERRCODE = '22023';
    END IF;
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, icp_position, icp_hash,
      attempt, status, stage_generation, kind, work_item_id, scored_run_id
    ) VALUES (
      v_assignment || ':1', v_assignment, p_round_id, v_scored.submission_id, v_scored.miner_hotkey,
      p_stage, v_scored.icp_position, v_scored.icp_hash, 1, 'pending', v_generation, 'score',
      v_item ->> 'work_item_id', v_scored.run_id
    );
    v_created := v_created + 1;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = v_next, status_generation = status_generation + 1, stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'round_status', v_next,
    'stage_generation', v_generation, 'assignments', v_created);
END;
$lab_arena_open_scoring$;
ALTER FUNCTION public.lab_arena_open_scoring(TEXT, SMALLINT, JSONB) OWNER TO lab_arena_owner;

-- Close the scoring window: open score runs fail as stage_closed, and the
-- round is cancelled when any scoring assignment is incomplete for a reason
-- that is not the scored miner's own key (an infrastructure gap is never
-- turned into a miner's zero). Otherwise the round is stageN_judged and the
-- Arena verifies the breakdowns and builds the bundle.
CREATE OR REPLACE FUNCTION public.lab_arena_close_scoring(p_round_id TEXT, p_stage SMALLINT)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_close_scoring$
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
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> ('stage' || p_stage::TEXT || '_scoring') THEN
    IF (p_stage = 1 AND v_round.status IN ('stage1_judged', 'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published'))
       OR (p_stage = 2 AND v_round.status IN ('stage2_judged', 'scored', 'published')) THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation);
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND stage = p_stage AND kind = 'score' AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    IF v_run.status = 'leased' THEN
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object('closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status)
    WHERE run_id = v_run.run_id;
  END LOOP;
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT DISTINCT ON (runs.assignment_id) runs.assignment_id, runs.status, runs.terminal_cause
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.assignment_id, (runs.status = 'accepted') DESC, runs.attempt DESC
  ) AS latest
  WHERE latest.status <> 'accepted'
    AND COALESCE(latest.terminal_cause, '') <> 'judge_key_refused';
  IF v_incomplete > 0 THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled', status_generation = status_generation + 1, stage_generation = v_generation,
        cancel_reason = 'capacity:scoring' || p_stage::TEXT || ':' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage' || p_stage::TEXT || '_judged';
    UPDATE public.lab_arena_rounds
    SET status = v_next, status_generation = status_generation + 1, stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE WHEN v_next = 'cancelled' THEN 'cancelled' ELSE 'closed' END,
    'round_status', v_next, 'incomplete_assignments', v_incomplete, 'stage_generation', v_generation);
END;
$lab_arena_close_scoring$;
ALTER FUNCTION public.lab_arena_close_scoring(TEXT, SMALLINT) OWNER TO lab_arena_owner;


-- Reject scorings the Arena could not reproduce from their recorded judge
-- responses: each named accepted score run fails as replay_rejected and gets
-- a second attempt; a run with no attempt left cancels the round (an
-- unreproducible scoring is never turned into a miner's zero). The round
-- returns to the scoring window at a new stage generation.
CREATE OR REPLACE FUNCTION public.lab_arena_reject_scorings(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_run_ids TEXT[]
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_reject_scorings$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_run_id TEXT;
  v_generation BIGINT;
  v_rejected INTEGER := 0;
  v_exhausted INTEGER := 0;
BEGIN
  IF p_stage NOT IN (1, 2) OR p_run_ids IS NULL OR pg_catalog.array_length(p_run_ids, 1) IS NULL THEN
    RAISE EXCEPTION 'lab_arena_reject_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> ('stage' || p_stage::TEXT || '_judged') THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status);
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOREACH v_run_id IN ARRAY p_run_ids LOOP
    SELECT * INTO v_run FROM public.lab_arena_runs
    WHERE run_id = v_run_id AND round_id = p_round_id AND stage = p_stage AND kind = 'score' AND status = 'accepted'
    FOR UPDATE;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_reject_run_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'replay_rejected',
        terminal_doc = pg_catalog.jsonb_build_object('rejected_at', pg_catalog.clock_timestamp(), 'previous_status', 'accepted',
                                                     'rejected_at_stage_generation', v_round.stage_generation)
    WHERE run_id = v_run.run_id;
    v_rejected := v_rejected + 1;
    IF v_run.attempt < 2 THEN
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, icp_position, icp_hash,
        attempt, status, lease_generation, stage_generation, kind, work_item_id, scored_run_id
      ) VALUES (
        v_run.assignment_id || ':' || (v_run.attempt + 1)::TEXT, v_run.assignment_id, v_run.round_id,
        v_run.submission_id, v_run.miner_hotkey, v_run.stage, v_run.icp_position, v_run.icp_hash,
        v_run.attempt + 1, 'pending', v_run.lease_generation, v_generation,
        v_run.kind, v_run.work_item_id, v_run.scored_run_id
      );
    ELSE
      v_exhausted := v_exhausted + 1;
    END IF;
  END LOOP;
  IF v_exhausted > 0 THEN
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled', status_generation = status_generation + 1, stage_generation = v_generation,
        cancel_reason = 'capacity:replay' || p_stage::TEXT || ':' || v_exhausted::TEXT
    WHERE round_id = p_round_id;
    RETURN pg_catalog.jsonb_build_object('status', 'cancelled', 'round_status', 'cancelled',
      'rejected', v_rejected, 'exhausted', v_exhausted, 'stage_generation', v_generation);
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'stage' || p_stage::TEXT || '_scoring', status_generation = status_generation + 1, stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object('status', 'reopened', 'round_status', 'stage' || p_stage::TEXT || '_scoring',
    'rejected', v_rejected, 'exhausted', 0, 'stage_generation', v_generation);
END;
$lab_arena_reject_scorings$;
ALTER FUNCTION public.lab_arena_reject_scorings(TEXT, SMALLINT, TEXT[]) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_cancel_round(p_round_id TEXT, p_reason TEXT)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_cancel_round$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_generation BIGINT;
BEGIN
  IF COALESCE(pg_catalog.btrim(p_reason), '') = '' OR pg_catalog.length(p_reason) > 200 THEN
    RAISE EXCEPTION 'lab_arena_cancel_reason_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status = 'published' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', 'published');
  END IF;
  IF v_round.status = 'cancelled' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', 'cancelled');
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    IF v_run.status = 'leased' THEN
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'round_cancelled');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object('cancelled_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status)
    WHERE run_id = v_run.run_id;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = 'cancelled', status_generation = status_generation + 1,
      stage_generation = v_generation, cancel_reason = p_reason
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object('status', 'cancelled', 'round_status', 'cancelled',
    'stage_generation', v_generation, 'previous_status', v_round.status);
END;
$lab_arena_cancel_round$;
ALTER FUNCTION public.lab_arena_cancel_round(TEXT, TEXT) OWNER TO lab_arena_owner;

-- Record per-attempt scores produced from a signed scoring plan (write-once).
CREATE OR REPLACE FUNCTION public.lab_arena_record_run_scores(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_scores JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_record_run_scores$
DECLARE
  v_round public.lab_arena_rounds;
  v_score JSONB;
  v_run public.lab_arena_runs;
  v_recorded INTEGER := 0;
  v_existing INTEGER := 0;
  v_value NUMERIC(12, 6);
BEGIN
  IF p_stage NOT IN (1, 2) OR pg_catalog.jsonb_typeof(p_scores) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_scores_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  -- Scores are written while the stage is judged (validators scored it and
  -- the replay stood); the closed status is the older, pre-judging state.
  IF v_round.status NOT IN ('stage' || p_stage::TEXT || '_closed', 'stage' || p_stage::TEXT || '_judged') THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status);
  END IF;
  FOR v_score IN SELECT value FROM pg_catalog.jsonb_array_elements(p_scores) LOOP
    v_value := (v_score ->> 'per_icp_score')::NUMERIC(12, 6);
    IF v_value IS NULL OR v_value < 0 OR v_value > 100 THEN
      RAISE EXCEPTION 'lab_arena_score_value_invalid' USING ERRCODE = '22023';
    END IF;
    SELECT * INTO v_run FROM public.lab_arena_runs
    WHERE run_id = (v_score ->> 'run_id') AND round_id = p_round_id AND stage = p_stage FOR UPDATE;
    IF NOT FOUND OR v_run.status NOT IN ('accepted', 'failed') THEN
      RAISE EXCEPTION 'lab_arena_score_run_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_run.per_icp_score IS NOT NULL THEN
      IF v_run.per_icp_score = v_value THEN
        v_existing := v_existing + 1;
        CONTINUE;
      END IF;
      RAISE EXCEPTION 'lab_arena_score_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_runs
    SET per_icp_score = v_value, score_ref = v_score ->> 'score_ref', score_doc = v_score -> 'score_doc'
    WHERE run_id = v_run.run_id;
    v_recorded := v_recorded + 1;
  END LOOP;
  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'recorded', v_recorded, 'existing', v_existing);
END;
$lab_arena_record_run_scores$;
ALTER FUNCTION public.lab_arena_record_run_scores(TEXT, SMALLINT, JSONB) OWNER TO lab_arena_owner;

-- ---------------------------------------------------------------------------
-- Row level security, grants, policies
-- ---------------------------------------------------------------------------

ALTER TABLE public.lab_arena_rounds ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lab_arena_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lab_arena_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lab_arena_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lab_arena_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lab_arena_ledger ENABLE ROW LEVEL SECURITY;

DO $lab_arena_table_acl$
DECLARE
  relation_name TEXT;
  role_name TEXT;
BEGIN
  FOREACH relation_name IN ARRAY ARRAY[
    'lab_arena_rounds', 'lab_arena_submissions', 'lab_arena_runs',
    'lab_arena_events', 'lab_arena_accounts', 'lab_arena_ledger'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON TABLE public.%I FROM PUBLIC', relation_name);
    EXECUTE pg_catalog.format('REVOKE TRUNCATE ON TABLE public.%I FROM PUBLIC', relation_name);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
        EXECUTE pg_catalog.format('REVOKE ALL ON TABLE public.%I FROM %I', relation_name, role_name);
        EXECUTE pg_catalog.format('REVOKE TRUNCATE ON TABLE public.%I FROM %I', relation_name, role_name);
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format('REVOKE ALL ON TABLE public.%I FROM lab_arena_service', relation_name);
    EXECUTE pg_catalog.format('GRANT SELECT ON TABLE public.%I TO lab_arena_service', relation_name);
    EXECUTE pg_catalog.format('DROP POLICY IF EXISTS %I ON public.%I', relation_name || '_service_read', relation_name);
    EXECUTE pg_catalog.format(
      'CREATE POLICY %I ON public.%I FOR SELECT TO lab_arena_service USING (TRUE)',
      relation_name || '_service_read', relation_name);
  END LOOP;
END;
$lab_arena_table_acl$;

DO $lab_arena_function_acl$
DECLARE
  signature TEXT;
  role_name TEXT;
BEGIN
  -- Service-callable functions.
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_whoami()',
    'public.lab_arena_create_round(TEXT, TEXT, JSONB)',
    'public.lab_arena_transition_round(TEXT, TEXT, TEXT, JSONB)',
    'public.lab_arena_append_journal_entry(TEXT, JSONB)',
    'public.lab_arena_register_submission(TEXT, TEXT, TEXT, JSONB)',
    'public.lab_arena_update_submission(TEXT, TEXT, TEXT, TEXT, JSONB)',
    'public.lab_arena_upsert_account_credential(TEXT, TEXT, TEXT, TEXT, JSONB)',
    'public.lab_arena_record_preflight(TEXT, TEXT, JSONB)',
    'public.lab_arena_open_stage(TEXT, SMALLINT, JSONB, INTEGER[], TEXT[])',
    'public.lab_arena_claim_assignment(TEXT, TEXT, INTEGER, INTEGER, TEXT[], TEXT, TEXT, TEXT, INTEGER)',
    'public.lab_arena_reserve_call(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER)',
    'public.lab_arena_mark_dispatched(TEXT, TEXT, TEXT)',
    'public.lab_arena_settle_call(TEXT, TEXT, TEXT, BIGINT, JSONB, JSONB, INTEGER)',
    'public.lab_arena_mark_uncertain(TEXT, TEXT, TEXT, JSONB, JSONB, INTEGER)',
    'public.lab_arena_append_events(TEXT, TEXT, JSONB, INTEGER)',
    'public.lab_arena_complete_attempt(TEXT, TEXT, JSONB, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT)',
    'public.lab_arena_expire_leases(TEXT)',
    'public.lab_arena_close_stage(TEXT, SMALLINT)',
    'public.lab_arena_open_scoring(TEXT, SMALLINT, JSONB)',
    'public.lab_arena_close_scoring(TEXT, SMALLINT)',
    'public.lab_arena_reject_scorings(TEXT, SMALLINT, TEXT[])',
    'public.lab_arena_cancel_round(TEXT, TEXT)',
    'public.lab_arena_record_run_scores(TEXT, SMALLINT, JSONB)'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
        EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name);
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service', signature);
  END LOOP;
  -- Internal helpers and trigger functions: owner only.
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena__ledger_head(TEXT)',
    'public.lab_arena__run_consumed(TEXT, TEXT)',
    'public.lab_arena__submission_stage_consumed(TEXT, SMALLINT, TEXT, TEXT)',
    'public.lab_arena__aggregate_preflight(JSONB)',
    'public.lab_arena__apply_terminal_funding(public.lab_arena_ledger, TEXT, BIGINT)',
    'public.lab_arena__terminate_open_calls(TEXT, TEXT)',
    'public.lab_arena__lock_current_lease(TEXT, TEXT)',
    'public.lab_arena__append_event_locked(public.lab_arena_runs, JSONB)',
    'public.lab_arena__account_view(public.lab_arena_accounts)',
    'public.lab_arena__call_state_view(public.lab_arena_ledger, public.lab_arena_runs)',
    'public.lab_arena_append_only_v1()',
    'public.lab_arena_rounds_write_once_v1()',
    'public.lab_arena_submissions_frozen_v1()',
    'public.lab_arena_runs_terminal_v1()',
    'public.lab_arena_accounts_touch_v1()'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
        EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name);
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM lab_arena_service', signature);
  END LOOP;
END;
$lab_arena_function_acl$;

COMMENT ON TABLE public.lab_arena_rounds IS
  'Lab Arena V1 rounds: configuration, hash-chained generation journal, benchmark commitment, stage state, scoring plans, finalists, publication and signed reward basis (labarena.md section 11).';
COMMENT ON TABLE public.lab_arena_runs IS
  'Lab Arena V1 per-ICP logical assignments and their attempts; one active attempt per assignment, at most two attempts, preflight_failed records carry attempt 0.';
COMMENT ON TABLE public.lab_arena_ledger IS
  'Lab Arena V1 append-only money ledger: deposits, reservations, dispatch markers, settlements with terminal sanitized responses, uncertain marks, recoveries, refusals.';
COMMENT ON FUNCTION public.lab_arena_whoami() IS
  'SECURITY INVOKER readback of the caller role and its catalog attributes; the service refuses to run unless the role is lab_arena_service without superuser or BYPASSRLS.';

NOTIFY pgrst, 'reload schema';

COMMIT;
