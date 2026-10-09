-- Private self-reported host diagnostics. No job, score, or recovery authority.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
CREATE TABLE IF NOT EXISTS public.lab_arena_validator_events (
  validator_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  validator_hotkey TEXT NOT NULL,
  network TEXT NOT NULL,
  netuid INTEGER NOT NULL,
  event_id UUID NOT NULL,
  event_kind TEXT NOT NULL CHECK (event_kind IN (
    'validator.startup','validator.ready','validator.state',
    'validator.error','validator.recovered','validator.stopping')),
  occurred_at TIMESTAMPTZ NOT NULL,
  content JSONB NOT NULL CHECK (jsonb_typeof(content) = 'object'),
  run_id TEXT,
  round_id TEXT,
  submission_id TEXT,
  icp_position SMALLINT,
  run_kind TEXT,
  gateway_source_commit TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (validator_hotkey, event_id)
);
ALTER TABLE public.lab_arena_validator_events OWNER TO lab_arena_owner;
ALTER SEQUENCE public.lab_arena_validator_events_validator_event_id_seq OWNER TO lab_arena_owner;
CREATE INDEX IF NOT EXISTS lab_arena_validator_events_hotkey_created_idx
  ON public.lab_arena_validator_events (validator_hotkey, created_at);
CREATE INDEX IF NOT EXISTS lab_arena_validator_events_expiry_idx
  ON public.lab_arena_validator_events (created_at, validator_event_id);
ALTER TABLE public.lab_arena_validator_events ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_validator_events FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
REVOKE ALL ON SEQUENCE public.lab_arena_validator_events_validator_event_id_seq FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT SELECT ON public.lab_arena_validator_events TO lab_arena_service;
DROP POLICY IF EXISTS lab_arena_validator_events_service_read ON public.lab_arena_validator_events;
CREATE POLICY lab_arena_validator_events_service_read ON public.lab_arena_validator_events
  FOR SELECT TO lab_arena_service USING (TRUE);

CREATE OR REPLACE FUNCTION public.lab_arena_append_validator_events_v1(
  p_validator_hotkey TEXT, p_network TEXT, p_netuid INTEGER,
  p_events JSONB, p_gateway_source_commit TEXT
) RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $events$
DECLARE
  v_event JSONB;
  v_pair RECORD;
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_id UUID;
  v_time TIMESTAMPTZ;
  v_existing INTEGER := 0;
  v_inserted INTEGER := 0;
  v_minute INTEGER;
  v_day INTEGER;
  v_now TIMESTAMPTZ := clock_timestamp();
BEGIN
  IF COALESCE(p_validator_hotkey, '') !~ '^[1-9A-HJ-NP-Za-km-z]{47,48}$'
     OR COALESCE(p_network, '') !~ '^[a-z][a-z0-9_-]{0,31}$'
     OR p_netuid IS NULL OR p_netuid NOT BETWEEN 0 AND 65535
     OR p_gateway_source_commit IS NULL OR length(p_gateway_source_commit) > 64
     OR jsonb_typeof(p_events) IS DISTINCT FROM 'array'
     OR jsonb_array_length(p_events) NOT BETWEEN 1 AND 16
     OR octet_length(p_events::TEXT) > 32768 THEN
    RAISE EXCEPTION 'lab_arena_validator_events_invalid' USING ERRCODE = '22023';
  END IF;
  -- Serialize one hotkey's bounded rate accounting, including all networks.
  PERFORM pg_advisory_xact_lock(hashtextextended('validator-events:' || p_validator_hotkey, 0));
  DELETE FROM public.lab_arena_validator_events WHERE validator_event_id IN (
    SELECT validator_event_id FROM public.lab_arena_validator_events
    WHERE created_at < v_now - INTERVAL '14 days'
    ORDER BY created_at LIMIT 1000
  );
  SELECT count(*) FILTER (WHERE created_at >= v_now - INTERVAL '1 minute'), count(*)
  INTO v_minute, v_day FROM public.lab_arena_validator_events
  WHERE validator_hotkey = p_validator_hotkey AND created_at >= v_now - INTERVAL '1 day';
  -- Validate the entire batch before any inserts or correlation denials.
  FOR v_event IN SELECT value FROM jsonb_array_elements(p_events) LOOP
    IF jsonb_typeof(v_event) IS DISTINCT FROM 'object'
       OR NOT (v_event ?& ARRAY['event_id','kind','occurred_at','content'])
       OR (v_event - ARRAY['event_id','kind','occurred_at','content','run_id','round_id']) <> '{}'::JSONB
       OR jsonb_typeof(v_event -> 'content') IS DISTINCT FROM 'object'
       OR octet_length(v_event::TEXT) > 4096
       OR COALESCE(v_event ->> 'kind','') NOT IN ('validator.startup','validator.ready','validator.state','validator.error','validator.recovered','validator.stopping') THEN
      RAISE EXCEPTION 'lab_arena_validator_event_invalid' USING ERRCODE = '22023';
    END IF;
    BEGIN
      v_id := (v_event ->> 'event_id')::UUID;
      v_time := (v_event ->> 'occurred_at')::TIMESTAMPTZ;
    EXCEPTION WHEN OTHERS THEN
      RAISE EXCEPTION 'lab_arena_validator_event_invalid' USING ERRCODE = '22023';
    END;
    IF v_id IS NULL OR v_time IS NULL OR NOT isfinite(v_time)
       OR v_time < v_now - INTERVAL '2 days' OR v_time > v_now + INTERVAL '5 minutes' THEN
      RAISE EXCEPTION 'lab_arena_validator_event_time_invalid' USING ERRCODE = '22023';
    END IF;
    FOR v_pair IN SELECT key,value FROM jsonb_each(v_event -> 'content') LOOP
      IF v_pair.key IN ('phase','state','reason','operation','error_class','denial_code','scoring_state','weights_state','last_progress_at','last_poll_at','last_completion_at','launch_stderr','validator_source_commit','validator_source_dirty','validator_source_origin','session_id') THEN
        IF jsonb_typeof(v_pair.value) NOT IN ('string','null') OR octet_length(v_pair.value #>> '{}') > (CASE WHEN v_pair.key = 'launch_stderr' THEN 2048 ELSE 200 END) THEN
          RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
        END IF;
        IF v_pair.key = 'session_id' AND jsonb_typeof(v_pair.value) <> 'null' THEN
          PERFORM (v_pair.value #>> '{}')::UUID;
        END IF;
      ELSIF v_pair.key IN ('retryable','launch_timed_out','launch_stderr_truncated') THEN
        IF jsonb_typeof(v_pair.value) NOT IN ('boolean','null') THEN
          RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
        END IF;
      ELSIF v_pair.key IN ('http_status','attempt','active_runs','ready_slots','proxy_count','launch_exit_code') THEN
        IF jsonb_typeof(v_pair.value) <> 'null' THEN
          IF jsonb_typeof(v_pair.value) <> 'number' OR (v_pair.value #>> '{}') !~ '^-?[0-9]+$' THEN
            RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
          END IF;
          PERFORM (v_pair.value #>> '{}')::INTEGER;
        END IF;
      ELSIF v_pair.key = 'delay_seconds' THEN
        IF jsonb_typeof(v_pair.value) <> 'null' AND (jsonb_typeof(v_pair.value) <> 'number' OR (v_pair.value #>> '{}')::NUMERIC NOT BETWEEN 0 AND 86400) THEN
          RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
        END IF;
      ELSE
        RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
      END IF;
    END LOOP;
    IF (v_event ? 'run_id' AND COALESCE(v_event ->> 'run_id','') !~ '^[A-Za-z0-9._:-]{1,200}$')
       OR (v_event ? 'round_id' AND COALESCE(v_event ->> 'round_id','') !~ '^[A-Za-z0-9._:-]{1,200}$') THEN
      RAISE EXCEPTION 'lab_arena_validator_event_correlation_invalid' USING ERRCODE = '22023';
    END IF;
    v_run := NULL;
    IF v_event ? 'run_id' THEN
      SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = v_event ->> 'run_id';
      IF NOT FOUND OR v_run.runner_hotkey IS DISTINCT FROM p_validator_hotkey THEN
        RETURN jsonb_build_object('status','owner_required');
      END IF;
    END IF;
    v_round := NULL;
    IF v_run.run_id IS NOT NULL OR v_event ? 'round_id' THEN
      SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = COALESCE(v_run.round_id, v_event ->> 'round_id');
      IF NOT FOUND OR v_round.arena_network_name <> p_network OR v_round.arena_netuid <> p_netuid
         OR (v_run.run_id IS NOT NULL AND v_event ? 'round_id' AND v_event ->> 'round_id' <> v_run.round_id) THEN
        RETURN jsonb_build_object('status','correlation_invalid');
      END IF;
    END IF;
  END LOOP;
  -- Count new UUIDs only, so accepted exact retries do not consume allowance.
  SELECT count(*) INTO v_inserted FROM (
    SELECT DISTINCT (value ->> 'event_id')::UUID AS id FROM jsonb_array_elements(p_events)
  ) incoming WHERE NOT EXISTS (
    SELECT 1 FROM public.lab_arena_validator_events old WHERE old.validator_hotkey = p_validator_hotkey AND old.event_id = incoming.id
  );
  IF v_minute + v_inserted > 60 OR v_day + v_inserted > 1000 THEN
    RETURN jsonb_build_object('status','rate_limited');
  END IF;
  v_inserted := 0;
  FOR v_event IN SELECT value FROM jsonb_array_elements(p_events) LOOP
    v_run := NULL;
    IF v_event ? 'run_id' THEN
      SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = v_event ->> 'run_id';
    END IF;
    INSERT INTO public.lab_arena_validator_events (
      validator_hotkey,network,netuid,event_id,event_kind,occurred_at,content,
      run_id,round_id,submission_id,icp_position,run_kind,gateway_source_commit
    ) VALUES (
      p_validator_hotkey,p_network,p_netuid,(v_event ->> 'event_id')::UUID,
      v_event ->> 'kind',(v_event ->> 'occurred_at')::TIMESTAMPTZ,v_event -> 'content',
      v_run.run_id,COALESCE(v_run.round_id,v_event ->> 'round_id'),
      v_run.submission_id,v_run.icp_position,v_run.kind,p_gateway_source_commit
    ) ON CONFLICT (validator_hotkey,event_id) DO NOTHING;
    IF FOUND THEN v_inserted := v_inserted + 1; ELSE v_existing := v_existing + 1; END IF;
  END LOOP;
  RETURN jsonb_build_object('status','accepted','inserted',v_inserted,'existing',v_existing,'accepted',v_inserted + v_existing);
END;
$events$;
ALTER FUNCTION public.lab_arena_append_validator_events_v1(TEXT,TEXT,INTEGER,JSONB,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_append_validator_events_v1(TEXT,TEXT,INTEGER,JSONB,TEXT) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_append_validator_events_v1(TEXT,TEXT,INTEGER,JSONB,TEXT) TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
