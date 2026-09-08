-- Retry or replace unfinished source uploads without overwriting source bytes.
-- Name incomplete execution/judging accurately; scoring fairness is unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

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
  v_collision public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_expected_ref TEXT;
  v_checksum TEXT;
  v_existing_checksum TEXT;
BEGIN
  v_expected_ref := 'arena/' || p_round_id || '/sources/' || p_submission_id || '.tar.gz';
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
     OR pg_catalog.jsonb_typeof(p_doc) IS DISTINCT FROM 'object'
     OR p_doc ->> 'source_ref' IS DISTINCT FROM v_expected_ref
     OR COALESCE((p_doc ->> 'source_size_bytes')::BIGINT, 0) NOT BETWEEN 1 AND 10485760
     OR COALESCE((p_doc #>> '{consent,public_rerun}')::BOOLEAN, FALSE) IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_submission_input_invalid' USING ERRCODE = '22023';
  END IF;
  v_checksum := p_doc ->> 'source_content_md5';
  IF v_checksum IS NOT NULL AND (
    pg_catalog.length(pg_catalog.decode(v_checksum, 'base64')) <> 16
    OR pg_catalog.encode(pg_catalog.decode(v_checksum, 'base64'), 'base64') <> v_checksum
  ) THEN
    RAISE EXCEPTION 'lab_arena_source_checksum_invalid' USING ERRCODE = '22023';
  END IF;
  v_is_baseline := COALESCE((p_doc ->> 'is_king')::BOOLEAN, FALSE);

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open'
     OR (
       NOT v_is_baseline
       AND (
         COALESCE(pg_catalog.clock_timestamp() < (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ, TRUE)
         OR COALESCE(pg_catalog.clock_timestamp() >= (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ, TRUE)
       )
     ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'window_closed',
      'round_status', v_round.status
    );
  END IF;

  -- Serialize reservation replacement with finalize and round cutoff. The
  -- existing transport checksum lives in submission_doc; no new model identity.
  SELECT * INTO v_existing
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey
    AND status IN ('uploading', 'accepted', 'frozen')
  ORDER BY created_at LIMIT 1 FOR UPDATE;
  IF FOUND THEN
    v_existing_checksum := v_existing.submission_doc ->> 'source_content_md5';
    IF v_existing.source_size_bytes = (p_doc ->> 'source_size_bytes')::BIGINT
       AND (v_existing_checksum IS NOT DISTINCT FROM v_checksum
         OR (v_existing.status IN ('accepted', 'frozen') AND v_existing_checksum IS NULL)) THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'submission_status', v_existing.status,
        'submission_id', v_existing.submission_id, 'source_ref', v_existing.source_ref
      );
    END IF;
    IF v_existing.status <> 'uploading' OR v_is_baseline OR v_existing.is_king THEN
      RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
    END IF;
    -- Preserve the original bytes and row. A changed unfinalized archive gets
    -- a fresh server-assigned object reference, never an S3 overwrite.
    UPDATE public.lab_arena_submissions
    SET status = 'rejected', rejection_rule = 'source_replaced'
    WHERE submission_id = v_existing.submission_id AND status = 'uploading';
  END IF;

  SELECT * INTO v_collision
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id;
  IF FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
  END IF;

  INSERT INTO public.lab_arena_submissions (
    submission_id, round_id, miner_hotkey, status, is_king,
    source_ref, source_size_bytes, consent, submission_doc
  ) VALUES (
    p_submission_id, p_round_id, p_miner_hotkey, 'uploading', v_is_baseline,
    p_doc ->> 'source_ref', (p_doc ->> 'source_size_bytes')::BIGINT,
    p_doc -> 'consent', p_doc
  );
  RETURN pg_catalog.jsonb_build_object(
    'status', 'registered',
    'submission_status', 'uploading',
    'submission_id', p_submission_id,
    'source_ref', p_doc ->> 'source_ref'
  );
END;
$lab_arena_register_submission$;
ALTER FUNCTION public.lab_arena_register_submission(TEXT, TEXT, TEXT, JSONB)
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
          'execution_incomplete:stage' || p_stage::TEXT || ':' || v_incomplete::TEXT
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
  v_baseline_count INTEGER;
  v_baseline_incomplete INTEGER;
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
    IF (p_stage = 1 AND v_round.status IN ('stage1_judged', 'stage1_scored',
                                           'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published'))
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
  WHERE latest.status <> 'accepted';
  SELECT COUNT(*) INTO v_baseline_count
  FROM pg_catalog.jsonb_array_elements(COALESCE(v_round.participants, '[]'::JSONB)) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  SELECT COUNT(*) INTO v_baseline_incomplete FROM (
    SELECT DISTINCT ON (runs.assignment_id)
      runs.assignment_id, runs.submission_id, runs.status
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.assignment_id, (runs.status = 'accepted') DESC, runs.attempt DESC
  ) AS latest
  WHERE latest.status <> 'accepted'
    AND EXISTS (
      SELECT 1
      FROM pg_catalog.jsonb_array_elements(COALESCE(v_round.participants, '[]'::JSONB)) AS participant
      WHERE participant ->> 'submission_id' = latest.submission_id
        AND COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
    );
  IF v_incomplete > 0 AND (v_baseline_count <> 1 OR v_baseline_incomplete > 0) THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled', status_generation = status_generation + 1, stage_generation = v_generation,
        cancel_reason = 'scoring_incomplete:stage' || p_stage::TEXT || ':' || v_incomplete::TEXT
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

CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_schema_version$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.schema_version.v1', 'version', 193
  );
$lab_arena_schema_version$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1() TO lab_arena_service;
NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
