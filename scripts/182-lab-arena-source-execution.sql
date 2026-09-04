-- 182-lab-arena-source-execution.sql
-- Lease the private source archive for execution and remove the retired miner
-- image columns. The scorer image remains round-owned configuration, outside
-- the miner submission record.

BEGIN;

-- Required only for ownership transfers by the hosted migration role.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- A round that already admitted an image-only submission cannot continue in
-- the source competition. Cancel it once, fail its open work through the
-- existing cancellation function, and retire its incomplete submissions.
DO $lab_arena_source_cutover$
DECLARE
  v_round_id TEXT;
BEGIN
  FOR v_round_id IN
    SELECT DISTINCT rounds.round_id
    FROM public.lab_arena_rounds AS rounds
    JOIN public.lab_arena_submissions AS submissions
      ON submissions.round_id = rounds.round_id
    WHERE rounds.status NOT IN ('published', 'cancelled')
      AND submissions.status IN ('uploading', 'accepted', 'frozen')
      AND (
        submissions.source_ref IS NULL
        OR submissions.source_size_bytes IS NULL
      )
    ORDER BY rounds.round_id
  LOOP
    PERFORM public.lab_arena_cancel_round(v_round_id, 'source_bundle_cutover');
  END LOOP;
END;
$lab_arena_source_cutover$;

-- The immutable-frozen trigger correctly protects normal competition data.
-- Disable only that trigger for this one migration transition, after the
-- affected rounds are terminal, then enable it in the same transaction.
ALTER TABLE public.lab_arena_submissions
  DISABLE TRIGGER lab_arena_submissions_frozen;
UPDATE public.lab_arena_submissions AS submissions
SET status = 'rejected',
    rejection_rule = 'source_reupload_required',
    updated_at = pg_catalog.clock_timestamp()
FROM public.lab_arena_rounds AS rounds
WHERE rounds.round_id = submissions.round_id
  AND rounds.status = 'cancelled'
  AND submissions.status IN ('uploading', 'accepted', 'frozen')
  AND (
    submissions.source_ref IS NULL
    OR submissions.source_size_bytes IS NULL
  );
ALTER TABLE public.lab_arena_submissions
  ENABLE TRIGGER lab_arena_submissions_frozen;

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

ALTER TABLE public.lab_arena_submissions
  DROP COLUMN IF EXISTS submitted_reference,
  DROP COLUMN IF EXISTS image_reference,
  DROP COLUMN IF EXISTS image_digest,
  DROP COLUMN IF EXISTS image_size_bytes;

NOTIFY pgrst, 'reload schema';

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
