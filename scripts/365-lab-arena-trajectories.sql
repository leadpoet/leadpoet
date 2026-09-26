-- Store one bounded private event stream for Arena runtime and provider calls.
-- Run identity always comes from the active lease row.  Callers can only
-- append sanitized event documents through the lease-scoped RPC.
BEGIN;

CREATE TABLE IF NOT EXISTS public.lab_arena_trajectory_events (
  trajectory_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES public.lab_arena_runs (run_id),
  event_id UUID NOT NULL,
  round_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  miner_hotkey TEXT NOT NULL,
  runner_hotkey TEXT NOT NULL,
  assignment_id TEXT NOT NULL,
  icp_identifier TEXT NOT NULL,
  stage SMALLINT NOT NULL,
  icp_position SMALLINT NOT NULL,
  attempt SMALLINT NOT NULL,
  run_kind TEXT NOT NULL CHECK (run_kind IN ('execute', 'score')),
  model_role TEXT NOT NULL CHECK (model_role IN ('baseline', 'miner')),
  event_kind TEXT NOT NULL CHECK (
    event_kind ~ '^[a-z][a-z0-9]*([._-][a-z0-9]+){0,7}$'
  ),
  occurred_at TIMESTAMPTZ NOT NULL,
  content JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(content) = 'object'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  UNIQUE (run_id, event_id)
);
ALTER TABLE public.lab_arena_trajectory_events OWNER TO lab_arena_owner;
ALTER SEQUENCE public.lab_arena_trajectory_events_trajectory_id_seq
  OWNER TO lab_arena_owner;

CREATE INDEX IF NOT EXISTS lab_arena_trajectory_run_created_idx
  ON public.lab_arena_trajectory_events (run_id, created_at, trajectory_id);
CREATE INDEX IF NOT EXISTS lab_arena_trajectory_round_created_idx
  ON public.lab_arena_trajectory_events (round_id, created_at, trajectory_id);

ALTER TABLE public.lab_arena_trajectory_events ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.lab_arena_trajectory_events
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON SEQUENCE public.lab_arena_trajectory_events_trajectory_id_seq
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT SELECT ON TABLE public.lab_arena_trajectory_events TO lab_arena_service;

DROP POLICY IF EXISTS lab_arena_trajectory_events_service_read
  ON public.lab_arena_trajectory_events;
CREATE POLICY lab_arena_trajectory_events_service_read
  ON public.lab_arena_trajectory_events
  FOR SELECT TO lab_arena_service USING (TRUE);

CREATE OR REPLACE FUNCTION public.lab_arena_append_trajectory_events_v1(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_events JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_append_trajectory_events_v1$
DECLARE
  v_run public.lab_arena_runs;
  v_event JSONB;
  v_event_id UUID;
  v_kind TEXT;
  v_occurred_at TIMESTAMPTZ;
  v_existing INTEGER := 0;
  v_inserted INTEGER := 0;
  v_provider_count INTEGER;
  v_runtime_count INTEGER;
  v_provider_recent INTEGER;
  v_runtime_recent INTEGER;
  v_model_role TEXT;
  v_round_id TEXT;
  v_late_provider_only BOOLEAN;
BEGIN
  IF COALESCE(p_run_id, '') !~ '^[A-Za-z0-9._:-]{1,200}$'
     OR COALESCE(p_lease_token_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR pg_catalog.jsonb_typeof(p_events) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_events) NOT BETWEEN 1 AND 32
     OR pg_catalog.octet_length(p_events::TEXT) > 65536 THEN
    RAISE EXCEPTION 'lab_arena_trajectory_input_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT COALESCE(
    pg_catalog.bool_and(value ->> 'kind' IN ('provider.response', 'provider.error')),
    FALSE
  ) INTO v_late_provider_only
  FROM pg_catalog.jsonb_array_elements(p_events);

  IF v_late_provider_only THEN
    -- Provider execution can finish after the lease deadline or while a late
    -- billed result moves the run terminal.  The retained exact lease hash is
    -- still required.  Public runtime uploads never enter this branch.
    SELECT round_id INTO v_round_id
    FROM public.lab_arena_runs WHERE run_id = p_run_id;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_run_missing' USING ERRCODE = 'P0002';
    END IF;
    PERFORM 1 FROM public.lab_arena_rounds
    WHERE round_id = v_round_id FOR SHARE;
    SELECT * INTO v_run FROM public.lab_arena_runs
    WHERE run_id = p_run_id FOR UPDATE;
    IF v_run.lease_token_hash IS DISTINCT FROM p_lease_token_hash
       OR v_run.status NOT IN ('leased', 'submitted', 'accepted', 'failed') THEN
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END IF;
  ELSE
    BEGIN
      v_run := public.lab_arena__lock_current_lease(
        p_run_id, p_lease_token_hash
      );
    EXCEPTION WHEN SQLSTATE 'P0003' THEN
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END;
  END IF;
  SELECT CASE WHEN is_king THEN 'baseline' ELSE 'miner' END
  INTO v_model_role
  FROM public.lab_arena_submissions
  WHERE submission_id = v_run.submission_id;
  IF v_model_role IS NULL THEN
    RAISE EXCEPTION 'lab_arena_trajectory_submission_missing' USING ERRCODE = 'P0002';
  END IF;

  SELECT
    COUNT(*) FILTER (WHERE event_kind LIKE 'provider.%'),
    COUNT(*) FILTER (WHERE event_kind LIKE 'runtime.%'),
    COUNT(*) FILTER (
      WHERE event_kind LIKE 'provider.%'
        AND created_at >= pg_catalog.clock_timestamp() - INTERVAL '1 minute'
    ),
    COUNT(*) FILTER (
      WHERE event_kind LIKE 'runtime.%'
        AND created_at >= pg_catalog.clock_timestamp() - INTERVAL '1 minute'
    )
  INTO v_provider_count, v_runtime_count, v_provider_recent, v_runtime_recent
  FROM public.lab_arena_trajectory_events
  WHERE run_id = p_run_id;

  FOR v_event IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_events)
  LOOP
    IF pg_catalog.jsonb_typeof(v_event) IS DISTINCT FROM 'object'
       OR (SELECT COUNT(*) FROM pg_catalog.jsonb_object_keys(v_event)) <> 4
       OR NOT (v_event ?& ARRAY['event_id', 'kind', 'occurred_at', 'content'])
       OR pg_catalog.jsonb_typeof(v_event -> 'content') IS DISTINCT FROM 'object'
       OR pg_catalog.octet_length(v_event::TEXT) > 8192 THEN
      RAISE EXCEPTION 'lab_arena_trajectory_event_invalid' USING ERRCODE = '22023';
    END IF;
    BEGIN
      v_event_id := (v_event ->> 'event_id')::UUID;
      v_occurred_at := (v_event ->> 'occurred_at')::TIMESTAMPTZ;
    EXCEPTION WHEN OTHERS THEN
      RAISE EXCEPTION 'lab_arena_trajectory_event_invalid' USING ERRCODE = '22023';
    END;
    v_kind := v_event ->> 'kind';
    IF COALESCE(v_kind, '') !~ '^[a-z][a-z0-9]*([._-][a-z0-9]+){0,7}$'
       OR v_kind NOT LIKE 'runtime.%' AND v_kind NOT LIKE 'provider.%'
       OR v_occurred_at < pg_catalog.clock_timestamp() - INTERVAL '2 days'
       OR v_occurred_at > pg_catalog.clock_timestamp() + INTERVAL '5 minutes' THEN
      RAISE EXCEPTION 'lab_arena_trajectory_event_invalid' USING ERRCODE = '22023';
    END IF;

    IF EXISTS (
      SELECT 1 FROM public.lab_arena_trajectory_events
      WHERE run_id = p_run_id AND event_id = v_event_id
    ) THEN
      v_existing := v_existing + 1;
      CONTINUE;
    END IF;

    IF v_kind LIKE 'runtime.%' THEN
      IF v_runtime_count >= 512 OR v_runtime_recent >= 256 THEN
        RAISE EXCEPTION 'lab_arena_trajectory_runtime_limit' USING ERRCODE = '54000';
      END IF;
      v_runtime_count := v_runtime_count + 1;
      v_runtime_recent := v_runtime_recent + 1;
    ELSIF v_kind LIKE 'provider.%' THEN
      IF v_provider_count >= 9488 OR v_provider_recent >= 600 THEN
        RAISE EXCEPTION 'lab_arena_trajectory_provider_limit' USING ERRCODE = '54000';
      END IF;
      v_provider_count := v_provider_count + 1;
      v_provider_recent := v_provider_recent + 1;
    ELSE
      IF v_runtime_count + v_provider_count >= 10000 THEN
        RAISE EXCEPTION 'lab_arena_trajectory_event_limit' USING ERRCODE = '54000';
      END IF;
    END IF;

    INSERT INTO public.lab_arena_trajectory_events (
      run_id, event_id, round_id, submission_id, miner_hotkey,
      runner_hotkey, assignment_id, icp_identifier, stage, icp_position,
      attempt, run_kind, model_role, event_kind, occurred_at, content
    ) VALUES (
      v_run.run_id, v_event_id, v_run.round_id, v_run.submission_id,
      v_run.miner_hotkey, v_run.runner_hotkey, v_run.assignment_id,
      v_run.round_id || ':icp:' || v_run.icp_position::TEXT,
      v_run.stage, v_run.icp_position, v_run.attempt, v_run.kind,
      v_model_role, v_kind, v_occurred_at, v_event -> 'content'
    ) ON CONFLICT (run_id, event_id) DO NOTHING;
    IF FOUND THEN
      v_inserted := v_inserted + 1;
    ELSE
      v_existing := v_existing + 1;
    END IF;
  END LOOP;

  RETURN pg_catalog.jsonb_build_object(
    'status', 'accepted',
    'accepted', v_inserted + v_existing,
    'inserted', v_inserted,
    'existing', v_existing
  );
END;
$lab_arena_append_trajectory_events_v1$;

ALTER FUNCTION public.lab_arena_append_trajectory_events_v1(TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_append_trajectory_events_v1(TEXT, TEXT, JSONB)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_append_trajectory_events_v1(TEXT, TEXT, JSONB)
  TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
COMMIT;
