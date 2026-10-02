-- Reuse an exact, completed score-provider request through the existing ledger.
-- The reservation remains the durable claim; no provider call spans this lock.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE INDEX IF NOT EXISTS lab_arena_ledger_judgment_cache_reservation_idx
  ON public.lab_arena_ledger ((entry_doc ->> 'judgment_cache_key'), entry_id)
  WHERE entry_kind = 'reservation' AND provider = 'openrouter'
    AND entry_doc ? 'judgment_cache_key';

CREATE OR REPLACE FUNCTION public.lab_arena_reserve_judgment_call(
  p_run_id TEXT, p_lease_token_hash TEXT, p_call_identity TEXT,
  p_operation_id TEXT, p_provider TEXT, p_funding_source TEXT,
  p_amount_microusd BIGINT, p_call_doc JSONB, p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_reserve_judgment_call$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_head public.lab_arena_ledger;
  v_source RECORD;
  v_scope JSONB;
  v_body JSONB;
  v_canonical TEXT;
  v_key TEXT;
  v_reserved JSONB;
  v_terminal JSONB;
  v_source_terminal JSONB;
  v_prior_call_doc JSONB;
  v_busy BOOLEAN := FALSE;
  v_expires TIMESTAMPTZ;
BEGIN
  v_key := p_call_doc ->> 'judgment_cache_key';
  v_canonical := p_call_doc ->> 'judgment_cache_canonical';
  IF COALESCE(p_call_identity, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(v_key, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_operation_id, '') NOT IN
       ('openrouter.chat', 'openrouter.responses')
     OR p_provider IS DISTINCT FROM 'openrouter'
     OR p_funding_source NOT IN ('host', 'miner_key')
     OR COALESCE(p_amount_microusd, -1) < 0
     OR pg_catalog.jsonb_typeof(p_call_doc) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_call_doc::TEXT) > 65536
     OR pg_catalog.jsonb_typeof(p_call_doc -> 'judgment_cache_scope')
       IS DISTINCT FROM 'object'
     OR COALESCE(pg_catalog.octet_length(v_canonical), 0) NOT BETWEEN 2 AND 65536
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_judgment_cache_input_invalid'
      USING ERRCODE = '22023';
  END IF;
  BEGIN
    v_scope := v_canonical::JSONB;
  EXCEPTION WHEN invalid_text_representation THEN
    RAISE EXCEPTION 'lab_arena_judgment_cache_input_invalid'
      USING ERRCODE = '22023';
  END;
  IF v_scope IS DISTINCT FROM p_call_doc -> 'judgment_cache_scope'
     OR v_key IS DISTINCT FROM
       'sha256:' || pg_catalog.encode(extensions.digest(v_canonical, 'sha256'), 'hex')
     OR v_scope ->> 'schema_version' IS DISTINCT FROM
       'leadpoet.lab_arena.verifier_provider_request.v1'
     OR v_scope ->> 'requested_operation_id' IS DISTINCT FROM p_operation_id
     OR COALESCE(v_scope ->> 'effective_operation_id', '') NOT IN
       ('openrouter.chat', 'openrouter.responses')
     OR v_scope ->> 'effective_operation_id' IS DISTINCT FROM p_operation_id
     OR COALESCE(v_scope ->> 'request_hash', '') !~ '^sha256:[0-9a-f]{64}$'
     OR v_scope ->> 'request_hash' IS DISTINCT FROM p_call_doc ->> 'request_hash'
     OR COALESCE(v_scope ->> 'outbound_body_hash', '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(v_scope ->> 'evaluation_date', '') !~
       '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
     OR COALESCE(v_scope ->> 'scorer_image_digest', '') !~
       '^sha256:[0-9a-f]{64}$'
     OR COALESCE(v_scope ->> 'scorer_image_reference', '') = ''
     OR pg_catalog.jsonb_typeof(v_scope -> 'scorer_policy')
       IS DISTINCT FROM 'object' THEN
    RAISE EXCEPTION 'lab_arena_judgment_cache_input_invalid'
      USING ERRCODE = '22023';
  END IF;

  -- The key lock serializes a short ledger lookup and reservation. The
  -- provider request occurs only after this transaction has committed.
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena_judgment:' || v_key, 0)
  );
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = v_run.round_id;
  IF v_run.kind IS DISTINCT FROM 'score'
     OR v_scope ->> 'round_id' IS DISTINCT FROM v_run.round_id
     OR v_scope ->> 'evaluation_date' IS DISTINCT FROM v_round.evaluation_date
     OR v_scope ->> 'scorer_image_digest' IS DISTINCT FROM
       v_round.configuration_doc ->> 'scorer_image_digest'
     OR v_scope ->> 'scorer_image_reference' IS DISTINCT FROM
       v_round.configuration_doc ->> 'scorer_image_reference'
     OR v_scope -> 'scorer_policy' IS DISTINCT FROM
       v_round.configuration_doc -> 'scorer_policy' THEN
    RAISE EXCEPTION 'lab_arena_judgment_cache_scope_mismatch'
      USING ERRCODE = '22023';
  END IF;

  -- A retry must keep its own ledger identity, even if another source has
  -- become eligible since the first reservation.
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NOT NULL THEN
    SELECT CASE WHEN entry_kind = 'refusal' THEN entry_doc -> 'call'
      ELSE entry_doc END
    INTO v_prior_call_doc
    FROM public.lab_arena_ledger
    WHERE call_identity = p_call_identity
      AND entry_kind IN ('reservation', 'refusal')
    ORDER BY entry_id LIMIT 1;
    IF v_head.run_id IS DISTINCT FROM p_run_id
       OR v_prior_call_doc ->> 'judgment_cache_key' IS DISTINCT FROM v_key
       OR v_prior_call_doc -> 'judgment_cache_scope' IS DISTINCT FROM v_scope
       OR v_prior_call_doc ->> 'request_hash' IS DISTINCT FROM
         p_call_doc ->> 'request_hash' THEN
      RAISE EXCEPTION 'lab_arena_judgment_cache_identity_conflict'
        USING ERRCODE = '23505';
    END IF;
    v_reserved := public.lab_arena_reserve_call(
      p_run_id, p_lease_token_hash, p_call_identity, p_operation_id,
      p_provider, p_funding_source, p_amount_microusd, p_call_doc,
      p_lease_ttl_seconds
    );
    IF v_reserved ->> 'status' IN ('uncertain', 'settled') THEN
      SELECT entry_doc #> '{call,judgment_cache_response}'
      INTO v_source_terminal
      FROM public.lab_arena_ledger
      WHERE call_identity = p_call_identity
        AND entry_kind = 'uncertain'
        AND entry_doc #> '{call,judgment_cache_response,judgment_cache_eligible}'
          = 'true'::JSONB
      ORDER BY entry_id LIMIT 1;
      IF v_source_terminal IS NOT NULL THEN
        RETURN v_reserved || pg_catalog.jsonb_build_object(
          'terminal_response', v_source_terminal,
          'judgment_cache_pending_reply', TRUE
        );
      END IF;
    END IF;
    RETURN v_reserved;
  END IF;

  -- Consider only direct paid-provider reservations. A cached destination
  -- may replay a source response but must never become a new source.
  FOR v_source IN
    SELECT reservation.call_identity, reservation.run_id,
      head.entry_kind, head.terminal_response,
      pending.judgment_cache_response,
      source_run.status AS run_status,
      source_run.lease_expires_at
    FROM public.lab_arena_ledger AS reservation
    JOIN public.lab_arena_runs AS source_run
      ON source_run.run_id = reservation.run_id
    JOIN LATERAL (
      SELECT entry_kind, terminal_response
      FROM public.lab_arena_ledger
      WHERE call_identity = reservation.call_identity
      ORDER BY entry_id DESC LIMIT 1
    ) AS head ON TRUE
    LEFT JOIN LATERAL (
      SELECT entry_doc #> '{call,judgment_cache_response}'
        AS judgment_cache_response
      FROM public.lab_arena_ledger
      WHERE call_identity = reservation.call_identity
        AND entry_kind = 'uncertain'
        AND entry_doc #> '{call,judgment_cache_response}' IS NOT NULL
      ORDER BY entry_id LIMIT 1
    ) AS pending ON TRUE
    WHERE reservation.entry_kind = 'reservation'
      AND reservation.provider = 'openrouter'
      AND reservation.entry_doc ? 'judgment_cache_key'
      AND reservation.entry_doc ->> 'judgment_cache_key' = v_key
      AND reservation.entry_doc -> 'judgment_cache_scope' = v_scope
      AND NOT reservation.entry_doc ? 'judgment_cache_source_call_identity'
      AND source_run.kind = 'score'
      AND reservation.call_identity <> p_call_identity
    ORDER BY reservation.entry_id
  LOOP
    -- The first complete reply remains the canonical response if its
    -- uncertain cost is reconciled later with a different ledger head.
    v_source_terminal := COALESCE(
      v_source.judgment_cache_response,
      CASE WHEN v_source.entry_kind = 'settlement'
        THEN v_source.terminal_response ELSE NULL END
    );
    IF v_source_terminal -> 'judgment_cache_eligible' = 'true'::JSONB
       AND v_source_terminal -> 'call_succeeded' = 'true'::JSONB
       AND COALESCE(v_source_terminal ->> 'status', '') ~ '^[0-9]{3}$'
       AND (v_source_terminal ->> 'status')::INTEGER BETWEEN 200 AND 299
       AND pg_catalog.jsonb_typeof(v_source_terminal -> 'headers') = 'object'
       AND pg_catalog.jsonb_typeof(v_source_terminal -> 'body_b64') = 'string'
       AND COALESCE(v_source_terminal ->> 'body_b64', '') <> '' THEN
      -- A marker alone cannot make an empty or malformed body reusable.
      BEGIN
        v_body := pg_catalog.convert_from(
          pg_catalog.decode(v_source_terminal ->> 'body_b64', 'base64'),
          'UTF8'
        )::JSONB;
      EXCEPTION WHEN data_exception THEN
        v_body := NULL;
      END;
      IF pg_catalog.jsonb_typeof(v_body) = 'object' THEN
        v_reserved := public.lab_arena_reserve_call(
          p_run_id, p_lease_token_hash, p_call_identity, p_operation_id,
          p_provider, p_funding_source, 0,
          (p_call_doc - 'reserve_remaining_budget')
            || pg_catalog.jsonb_build_object(
            'judgment_cache_source_call_identity', v_source.call_identity,
            'judgment_cache_source_run_id', v_source.run_id
          ), p_lease_ttl_seconds
        );
        IF v_reserved ->> 'status' IS DISTINCT FROM 'reserved' THEN
          RETURN v_reserved;
        END IF;
        v_terminal := (v_source_terminal - 'provider_cost'
          - 'account_failure_evidence') || pg_catalog.jsonb_build_object(
            'judgment_cache_key', v_key,
            'judgment_cache_source_call_identity', v_source.call_identity,
            'judgment_cache_source_run_id', v_source.run_id,
            'judgment_cache_eligible', TRUE
          );
        INSERT INTO public.lab_arena_ledger (
          entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
          call_identity, provider, operation_id, funding_source,
          amount_microusd, entry_doc, terminal_response
        ) VALUES (
          'settlement', v_run.miner_hotkey, v_run.round_id,
          v_run.submission_id, p_run_id, v_run.stage, p_call_identity,
          p_provider, p_operation_id, p_funding_source, 0,
          pg_catalog.jsonb_build_object(
            'reserved_microusd', 0, 'released_microusd', 0,
            'variance_microusd', 0,
            'judgment_cache_key', v_key,
            'judgment_cache_source_call_identity', v_source.call_identity,
            'judgment_cache_source_run_id', v_source.run_id
          ), v_terminal
        );
        RETURN pg_catalog.jsonb_build_object(
          'status', 'cache_hit', 'idempotent', FALSE,
          'call_identity', p_call_identity,
          'judgment_cache_key', v_key,
          'source_call_identity', v_source.call_identity,
          'source_run_id', v_source.run_id,
          'actual_microusd', 0, 'amount_microusd', 0,
          'terminal_response', v_terminal,
          'lease_expires_at', v_reserved -> 'lease_expires_at'
        );
      END IF;
    END IF;
    IF v_source.entry_kind IN ('reservation', 'dispatch')
       AND v_source.run_status = 'leased'
       AND v_source.lease_expires_at > pg_catalog.clock_timestamp() THEN
      v_busy := TRUE;
    END IF;
  END LOOP;

  IF v_busy THEN
    v_expires := pg_catalog.clock_timestamp()
      + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
    UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
    WHERE run_id = p_run_id;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'cache_busy', 'idempotent', FALSE,
      'call_identity', p_call_identity,
      'judgment_cache_key', v_key, 'lease_expires_at', v_expires
    );
  END IF;
  RETURN public.lab_arena_reserve_call(
    p_run_id, p_lease_token_hash, p_call_identity, p_operation_id,
    p_provider, p_funding_source, p_amount_microusd, p_call_doc,
    p_lease_ttl_seconds
  );
END;
$lab_arena_reserve_judgment_call$;
ALTER FUNCTION public.lab_arena_reserve_judgment_call(
  TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_reserve_judgment_call(
  TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_reserve_judgment_call(
  TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
) TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
COMMIT;
