-- Retain one sanitized reply for an existing keyed Deepline call. Charges stay
-- in the original append-only ledger; no additional admission or paid dispatch.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_deepline_call_responses (
  call_identity TEXT PRIMARY KEY CHECK (call_identity ~ '^sha256:[0-9a-f]{64}$'),
  reservation_entry_id BIGINT NOT NULL UNIQUE REFERENCES public.lab_arena_ledger(entry_id),
  run_id TEXT NOT NULL REFERENCES public.lab_arena_runs(run_id),
  request_id TEXT NOT NULL CHECK (request_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'),
  terminal_response JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(terminal_response) = 'object'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.lab_arena_deepline_call_responses OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_deepline_call_responses ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.lab_arena_deepline_call_responses FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_deepline_call_responses_append_only ON public.lab_arena_deepline_call_responses;
CREATE TRIGGER lab_arena_deepline_call_responses_append_only
  BEFORE UPDATE OR DELETE ON public.lab_arena_deepline_call_responses
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_recover_deepline_response_v1(
  p_run_id TEXT, p_lease_token_hash TEXT, p_call_identity TEXT,
  p_request_hash TEXT, p_execution_key TEXT, p_credential_fingerprint TEXT,
  p_request_id TEXT, p_operation TEXT, p_actual_microusd BIGINT,
  p_terminal_response JSONB, p_lease_ttl_seconds INTEGER
) RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $recover$
DECLARE
  v_run public.lab_arena_runs;
  v_reservation public.lab_arena_ledger;
  v_head public.lab_arena_ledger;
  v_response public.lab_arena_deepline_call_responses;
  v_actual BIGINT;
BEGIN
  IF COALESCE(p_call_identity, '') !~ '^sha256:[0-9a-f]{64}$'
     OR p_execution_key IS DISTINCT FROM 'arena:' || pg_catalog.substr(p_call_identity, 8)
     OR COALESCE(p_request_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_credential_fingerprint, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_request_id, '') !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
     OR p_request_id ~ '^ctx-tool-[0-9a-f]{32}$'
     OR COALESCE(p_operation, '') = ''
     OR p_actual_microusd < 0
     OR pg_catalog.jsonb_typeof(p_terminal_response) IS DISTINCT FROM 'object'
     OR p_terminal_response -> 'call_succeeded' IS DISTINCT FROM 'true'::JSONB
     OR COALESCE(p_terminal_response ->> 'status', '') !~ '^2[0-9]{2}$'
     OR pg_catalog.jsonb_typeof(p_terminal_response -> 'headers') IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(p_terminal_response -> 'body_b64') IS DISTINCT FROM 'string'
     OR pg_catalog.octet_length(p_terminal_response::TEXT) > 4194304
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 5400 THEN
    RAISE EXCEPTION 'lab_arena_deepline_response_input_invalid' USING ERRCODE = '22023';
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  -- Same round/run/submission locking as admission and delayed settlement.
  PERFORM 1 FROM public.lab_arena_submissions
    WHERE submission_id = v_run.submission_id FOR NO KEY UPDATE;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  SELECT * INTO v_reservation FROM public.lab_arena_ledger
    WHERE call_identity = p_call_identity AND entry_kind = 'reservation';
  IF v_reservation.entry_id IS NULL
     OR v_reservation.run_id IS DISTINCT FROM p_run_id
     OR v_reservation.round_id IS DISTINCT FROM v_run.round_id
     OR v_reservation.submission_id IS DISTINCT FROM v_run.submission_id
     OR v_reservation.miner_hotkey IS DISTINCT FROM v_run.miner_hotkey
     OR v_reservation.stage IS DISTINCT FROM v_run.stage
     OR v_reservation.provider IS DISTINCT FROM 'deepline'
     OR v_reservation.entry_doc ->> 'request_hash' IS DISTINCT FROM p_request_hash
     OR v_reservation.entry_doc ->> 'deepline_execution_key' IS DISTINCT FROM p_execution_key
     OR v_reservation.entry_doc ->> 'credential_fingerprint' IS DISTINCT FROM p_credential_fingerprint
     OR v_reservation.entry_doc ->> 'tool' IS DISTINCT FROM p_operation
     OR v_head.run_id IS DISTINCT FROM p_run_id
     OR v_head.entry_kind NOT IN ('dispatch', 'uncertain', 'settlement')
     OR NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger AS dispatch
       WHERE dispatch.call_identity = p_call_identity AND dispatch.entry_kind = 'dispatch'
         AND dispatch.run_id = p_run_id AND dispatch.provider = 'deepline')
     OR (v_head.entry_kind = 'uncertain'
         AND v_head.entry_doc #>> '{call,deepline_job_id}' IS NOT NULL
         AND v_head.entry_doc #>> '{call,deepline_job_id}' IS DISTINCT FROM p_request_id)
     OR (v_head.entry_kind = 'settlement'
         AND v_head.terminal_response #>> '{provider_cost,request_id}' IS DISTINCT FROM p_request_id) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'conflict');
  END IF;
  IF p_actual_microusd IS NOT NULL AND (
     p_terminal_response #>> '{provider_cost,request_id}' IS DISTINCT FROM p_request_id
     OR p_terminal_response #>> '{provider_cost,operation}' IS DISTINCT FROM p_operation
     OR (v_head.entry_kind = 'settlement' AND v_head.amount_microusd <> p_actual_microusd)
  ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'conflict');
  END IF;
  SELECT * INTO v_response FROM public.lab_arena_deepline_call_responses
    WHERE call_identity = p_call_identity;
  IF v_response.call_identity IS NOT NULL THEN
    IF v_response.request_id IS DISTINCT FROM p_request_id THEN
      RETURN pg_catalog.jsonb_build_object('status', 'conflict');
    END IF;
  END IF;
  IF v_head.entry_kind = 'dispatch' AND p_actual_microusd IS NULL THEN
    RETURN pg_catalog.jsonb_build_object('status', 'conflict');
  END IF;
  IF v_head.entry_kind IN ('dispatch', 'uncertain') AND p_actual_microusd IS NOT NULL THEN
    INSERT INTO public.lab_arena_ledger (
      entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
      call_identity, provider, operation_id, funding_source, amount_microusd,
      entry_doc, terminal_response
    ) VALUES ('settlement', v_reservation.miner_hotkey, v_reservation.round_id,
      v_reservation.submission_id, p_run_id, v_reservation.stage, p_call_identity,
      'deepline', v_reservation.operation_id, v_reservation.funding_source,
      p_actual_microusd, pg_catalog.jsonb_build_object(
        'reserved_microusd', v_reservation.amount_microusd,
        'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
        'variance_microusd', p_actual_microusd - v_reservation.amount_microusd,
        'deepline_response_recovery', TRUE,
        'reconciled_uncertainty_entry_id', v_head.entry_id), p_terminal_response);
    v_head := public.lab_arena__ledger_head(p_call_identity);
  END IF;
  INSERT INTO public.lab_arena_deepline_call_responses (
    call_identity, reservation_entry_id, run_id, request_id, terminal_response
  ) VALUES (p_call_identity, v_reservation.entry_id, p_run_id, p_request_id, p_terminal_response)
    ON CONFLICT (call_identity) DO NOTHING;
  UPDATE public.lab_arena_runs SET lease_expires_at = pg_catalog.clock_timestamp()
    + pg_catalog.make_interval(secs => p_lease_ttl_seconds) WHERE run_id = p_run_id;
  RETURN public.lab_arena__call_state_view(v_head, v_run);
END;
$recover$;
ALTER FUNCTION public.lab_arena_recover_deepline_response_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,JSONB,INTEGER) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_recover_deepline_response_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,JSONB,INTEGER) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_recover_deepline_response_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,JSONB,INTEGER) TO lab_arena_service;

-- Overlay only the useful reply. The ledger head and charge stay authoritative.
DO $response_view$
DECLARE v_definition TEXT; v_old TEXT := $old$'terminal_response', p_head.terminal_response,$old$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena__call_state_view(public.lab_arena_ledger,public.lab_arena_runs)'::pg_catalog.regprocedure) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_call_responses') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'Deepline response view preimage differs';
    END IF;
    v_definition := pg_catalog.replace(v_definition, ' IMMUTABLE', ' STABLE');
    v_definition := pg_catalog.replace(v_definition, $old$'reason', p_head.entry_doc ->> 'reason',$old$,
      $new$'deepline_response_missing', (p_head.provider = 'deepline'
        AND p_head.entry_kind = 'settlement'
        AND p_head.entry_doc ->> 'deepline_delayed_reconciliation' = 'true'
        AND p_head.terminal_response -> 'call_succeeded' = 'false'::JSONB),
      'reason', p_head.entry_doc ->> 'reason',$new$);
    EXECUTE pg_catalog.replace(v_definition, v_old,
      $new$'terminal_response', COALESCE((SELECT response.terminal_response
        FROM public.lab_arena_deepline_call_responses AS response
        WHERE response.call_identity = p_head.call_identity
          AND response.run_id = p_head.run_id AND p_head.provider = 'deepline'),
        p_head.terminal_response),$new$);
  END IF;
END;
$response_view$;

-- Include recovered success once in existing successful-spend calculations.
-- The cancellation branch keeps its existing precedence.
DO $response_cost$
DECLARE v_definition TEXT; v_signature TEXT; v_old TEXT := '        THEN FALSE';
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena__successful_call_cost_state(text,text,text)',
    'public.lab_arena__successful_icp_cost_state(text,text,integer)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(v_signature::pg_catalog.regprocedure) INTO v_definition;
    IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_call_responses') = 0 THEN
      IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
        RAISE EXCEPTION 'Deepline success cost preimage differs';
      END IF;
      v_definition := pg_catalog.overlay(v_definition,
        $new$        THEN FALSE
        WHEN ledger.provider = 'deepline' AND ledger.entry_kind IN ('settlement','uncertain')
          AND EXISTS (SELECT 1 FROM public.lab_arena_deepline_call_responses AS response
            WHERE response.call_identity = ledger.call_identity AND response.run_id = ledger.run_id
              AND response.terminal_response -> 'call_succeeded' = 'true'::JSONB)
        THEN TRUE$new$
         , pg_catalog.strpos(v_definition, v_old), pg_catalog.length(v_old));
      EXECUTE v_definition;
    END IF;
  END LOOP;
END;
$response_cost$;

CREATE OR REPLACE FUNCTION public.lab_arena_deepline_response_schema_v1()
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $response_readiness$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_recover_deepline_response_v1(text,text,text,text,text,text,text,text,bigint,jsonb,integer)') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_deepline_call_responses') IS NULL
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena__call_state_view(public.lab_arena_ledger,public.lab_arena_runs)'::pg_catalog.regprocedure), 'lab_arena_deepline_call_responses') = 0
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena__successful_call_cost_state(text,text,text)'::pg_catalog.regprocedure), 'lab_arena_deepline_call_responses') = 0
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena__successful_icp_cost_state(text,text,integer)'::pg_catalog.regprocedure), 'lab_arena_deepline_call_responses') = 0 THEN
    RAISE EXCEPTION 'lab_arena_deepline_response_schema_incomplete' USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status','ok', 'schema_version',
    'leadpoet.lab_arena.deepline_response_schema.v1', 'version',417);
END;
$response_readiness$;
ALTER FUNCTION public.lab_arena_deepline_response_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_deepline_response_schema_v1() FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_deepline_response_schema_v1() TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
