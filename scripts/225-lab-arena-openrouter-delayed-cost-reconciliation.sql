-- Reconcile one exact OpenRouter generation after its immediate billing
-- readback was not ready. The paid request is never sent again.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_225_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__ledger_head(text)'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_ledger_settlement_uq'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_ledger_nonsettlement_terminal_uq'
     ) IS NULL THEN
    RAISE EXCEPTION
      'apply 223-lab-arena-cancelled-call-late-settlement.sql first';
  END IF;
END;
$lab_arena_225_prerequisites$;

CREATE OR REPLACE FUNCTION public.lab_arena_list_openrouter_cost_reconciliations_v1(
  p_round_id TEXT,
  p_run_id TEXT,
  p_after_entry_id BIGINT,
  p_limit INTEGER
)
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_list_openrouter_cost_reconciliations_v1$
  SELECT pg_catalog.jsonb_build_object(
    'status', 'ok',
    'items', COALESCE(
      (
        SELECT pg_catalog.jsonb_agg(
          pg_catalog.jsonb_build_object(
            'uncertain_entry_id', candidate.uncertain_entry_id,
            'round_id', candidate.round_id,
            'run_id', candidate.run_id,
            'submission_id', candidate.submission_id,
            'miner_hotkey', candidate.miner_hotkey,
            'assignment_id', candidate.assignment_id,
            'stage', candidate.stage,
            'icp_position', candidate.icp_position,
            'attempt', candidate.attempt,
            'kind', candidate.kind,
            'call_identity', candidate.call_identity,
            'generation_id', candidate.generation_id,
            'credential_fingerprint', candidate.credential_fingerprint,
            'funding_source', candidate.funding_source,
            'run_status', candidate.run_status,
            'lease_expires_at', candidate.lease_expires_at
          )
          ORDER BY candidate.uncertain_entry_id
        )
        FROM (
          SELECT
            uncertainty.entry_id AS uncertain_entry_id,
            uncertainty.round_id,
            uncertainty.run_id,
            uncertainty.submission_id,
            uncertainty.miner_hotkey,
            runs.assignment_id,
            uncertainty.stage,
            runs.icp_position,
            runs.attempt,
            runs.kind,
            uncertainty.call_identity,
            uncertainty.entry_doc #>>
              '{call,openrouter_generation_id}' AS generation_id,
            uncertainty.entry_doc #>>
              '{call,credential_fingerprint}' AS credential_fingerprint,
            uncertainty.funding_source,
            runs.status AS run_status,
            pg_catalog.to_char(
              runs.lease_expires_at AT TIME ZONE 'UTC',
              'YYYY-MM-DD"T"HH24:MI:SS"Z"'
            ) AS lease_expires_at
          FROM public.lab_arena_ledger AS uncertainty
          JOIN public.lab_arena_runs AS runs
            ON runs.run_id = uncertainty.run_id
           AND runs.round_id = uncertainty.round_id
           AND runs.submission_id = uncertainty.submission_id
           AND runs.miner_hotkey = uncertainty.miner_hotkey
           AND runs.stage = uncertainty.stage
          JOIN public.lab_arena_rounds AS rounds
            ON rounds.round_id = uncertainty.round_id
          WHERE uncertainty.round_id = p_round_id
            AND (COALESCE(p_run_id, '') = '' OR uncertainty.run_id = p_run_id)
            AND rounds.status NOT IN ('open', 'published', 'cancelled')
            AND uncertainty.entry_kind = 'uncertain'
            AND uncertainty.provider = 'openrouter'
            AND uncertainty.operation_id = 'openrouter.chat'
            AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
            AND uncertainty.entry_doc #>> '{call,reason}' =
                'missing_provider_cost'
            AND uncertainty.entry_doc #>>
                '{call,openrouter_generation_id}' ~
                '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
            AND uncertainty.entry_doc #>>
                '{call,credential_fingerprint}' ~ '^sha256:[0-9a-f]{64}$'
            AND NOT EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS later
              WHERE later.call_identity = uncertainty.call_identity
                AND later.entry_id > uncertainty.entry_id
            )
            AND EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS reservation
              WHERE reservation.call_identity = uncertainty.call_identity
                AND reservation.entry_kind = 'reservation'
                AND reservation.run_id = uncertainty.run_id
                AND reservation.round_id = uncertainty.round_id
                AND reservation.submission_id = uncertainty.submission_id
                AND reservation.miner_hotkey = uncertainty.miner_hotkey
                AND reservation.stage = uncertainty.stage
                AND reservation.provider = uncertainty.provider
                AND reservation.operation_id = uncertainty.operation_id
                AND reservation.funding_source = uncertainty.funding_source
            )
            AND EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS dispatch
              WHERE dispatch.call_identity = uncertainty.call_identity
                AND dispatch.entry_kind = 'dispatch'
                AND dispatch.run_id = uncertainty.run_id
            )
          ORDER BY
            uncertainty.entry_id <= GREATEST(COALESCE(p_after_entry_id, 0), 0),
            uncertainty.entry_id
          LIMIT CASE
            WHEN p_limit BETWEEN 1 AND 20 THEN p_limit
            ELSE 0
          END
        ) AS candidate
      ),
      '[]'::JSONB
    )
  );
$lab_arena_list_openrouter_cost_reconciliations_v1$;
ALTER FUNCTION public.lab_arena_list_openrouter_cost_reconciliations_v1(
  TEXT, TEXT, BIGINT, INTEGER
) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_reconcile_openrouter_cost_v1(
  p_round_id TEXT,
  p_run_id TEXT,
  p_call_identity TEXT,
  p_uncertain_entry_id BIGINT,
  p_generation_id TEXT,
  p_credential_fingerprint TEXT,
  p_actual_microusd BIGINT,
  p_cost_units TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_reconcile_openrouter_cost_v1$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_submission public.lab_arena_submissions;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_terminal JSONB;
BEGIN
  IF COALESCE(p_round_id, '') = ''
     OR COALESCE(p_run_id, '') = ''
     OR COALESCE(p_call_identity, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_uncertain_entry_id, 0) < 1
     OR COALESCE(p_generation_id, '') !~
        '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
     OR COALESCE(p_credential_fingerprint, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_actual_microusd, -1) < 0
     OR p_actual_microusd > 9007199254740991
     OR COALESCE(p_cost_units, '') !~
        '^(0|[1-9][0-9]{0,9})([.][0-9]{1,18})?$'
     OR pg_catalog.ceil(p_cost_units::NUMERIC * 1000000)::BIGINT
        IS DISTINCT FROM p_actual_microusd THEN
    RAISE EXCEPTION 'lab_arena_openrouter_reconciliation_input_invalid'
      USING ERRCODE = '22023';
  END IF;

  -- Use the same round, run, submission lock order as reservation and
  -- settlement. No lock is held while the gateway performs the provider GET.
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status IN ('open', 'published', 'cancelled') THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  SELECT * INTO v_run
  FROM public.lab_arena_runs
  WHERE run_id = p_run_id
    AND round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  SELECT * INTO v_submission
  FROM public.lab_arena_submissions
  WHERE submission_id = v_run.submission_id
  FOR NO KEY UPDATE;
  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;

  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NULL
     OR v_head.round_id IS DISTINCT FROM p_round_id
     OR v_head.run_id IS DISTINCT FROM p_run_id
     OR v_head.submission_id IS DISTINCT FROM v_run.submission_id
     OR v_head.miner_hotkey IS DISTINCT FROM v_run.miner_hotkey
     OR v_head.stage IS DISTINCT FROM v_run.stage THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  IF v_head.entry_kind = 'settlement'
     AND v_head.entry_doc ->> 'openrouter_delayed_reconciliation' = 'true' THEN
    IF (v_head.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
          = p_uncertain_entry_id
       AND v_head.entry_doc ->> 'openrouter_generation_id' = p_generation_id
       AND v_head.entry_doc ->> 'credential_fingerprint' =
           p_credential_fingerprint
       AND v_head.amount_microusd = p_actual_microusd
       AND v_head.terminal_response #>> '{provider_cost,units}' = p_cost_units THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'settled',
        'idempotent', TRUE,
        'actual_microusd', v_head.amount_microusd,
        'released_microusd',
          (v_head.entry_doc ->> 'released_microusd')::BIGINT,
        'variance_microusd',
          (v_head.entry_doc ->> 'variance_microusd')::BIGINT
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'conflict');
  END IF;
  IF v_head.entry_id IS DISTINCT FROM p_uncertain_entry_id
     OR v_head.entry_kind IS DISTINCT FROM 'uncertain'
     OR v_head.round_id IS DISTINCT FROM p_round_id
     OR v_head.run_id IS DISTINCT FROM p_run_id
     OR v_head.submission_id IS DISTINCT FROM v_run.submission_id
     OR v_head.miner_hotkey IS DISTINCT FROM v_run.miner_hotkey
     OR v_head.stage IS DISTINCT FROM v_run.stage
     OR v_head.provider IS DISTINCT FROM 'openrouter'
     OR v_head.operation_id IS DISTINCT FROM 'openrouter.chat'
     OR v_head.entry_doc ->> 'reason' IS DISTINCT FROM 'worker_reported'
     OR v_head.entry_doc #>> '{call,reason}' IS DISTINCT FROM
        'missing_provider_cost'
     OR v_head.entry_doc #>> '{call,openrouter_generation_id}'
        IS DISTINCT FROM p_generation_id
     OR v_head.entry_doc #>> '{call,credential_fingerprint}'
        IS DISTINCT FROM p_credential_fingerprint THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  SELECT * INTO v_reservation
  FROM public.lab_arena_ledger
  WHERE call_identity = p_call_identity
    AND entry_kind = 'reservation';
  IF v_reservation.entry_id IS NULL
     OR v_reservation.run_id IS DISTINCT FROM p_run_id
     OR v_reservation.round_id IS DISTINCT FROM p_round_id
     OR v_reservation.submission_id IS DISTINCT FROM v_run.submission_id
     OR v_reservation.miner_hotkey IS DISTINCT FROM v_run.miner_hotkey
     OR v_reservation.stage IS DISTINCT FROM v_run.stage
     OR v_reservation.provider IS DISTINCT FROM 'openrouter'
     OR v_reservation.operation_id IS DISTINCT FROM 'openrouter.chat'
     OR v_reservation.funding_source IS DISTINCT FROM v_head.funding_source
     OR NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_ledger AS dispatch
       WHERE dispatch.call_identity = p_call_identity
         AND dispatch.entry_kind = 'dispatch'
         AND dispatch.run_id = p_run_id
     ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;

  v_terminal := pg_catalog.jsonb_build_object(
    'status', 502,
    'headers', pg_catalog.jsonb_build_object(
      'content-type', 'application/json',
      'content-length', '41'
    ),
    'body_b64',
      'eyJlcnJvciI6eyJjb2RlIjoicHJvdmlkZXJfdW5hdmFpbGFibGUifX0=',
    'provider_cost', pg_catalog.jsonb_build_object(
      'basis', 'openrouter_generation_cost',
      'units', p_cost_units,
      'unit_name', 'usd',
      'operation', 'openrouter.chat',
      'request_id', p_generation_id
    )
  );
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc, terminal_response
  ) VALUES (
    'settlement', v_reservation.miner_hotkey, v_reservation.round_id,
    v_reservation.submission_id, v_reservation.run_id, v_reservation.stage,
    v_reservation.call_identity, v_reservation.provider,
    v_reservation.operation_id, v_reservation.funding_source,
    p_actual_microusd,
    pg_catalog.jsonb_build_object(
      'reserved_microusd', v_reservation.amount_microusd,
      'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
      'variance_microusd', p_actual_microusd - v_reservation.amount_microusd,
      'late_reconciliation', TRUE,
      'openrouter_delayed_reconciliation', TRUE,
      'reconciled_uncertainty_entry_id', p_uncertain_entry_id,
      'reconciled_uncertainty_reason', 'worker_reported',
      'openrouter_generation_id', p_generation_id,
      'credential_fingerprint', p_credential_fingerprint
    ),
    v_terminal
  );
  RETURN pg_catalog.jsonb_build_object(
    'status', 'settled',
    'idempotent', FALSE,
    'actual_microusd', p_actual_microusd,
    'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
    'variance_microusd', p_actual_microusd - v_reservation.amount_microusd
  );
END;
$lab_arena_reconcile_openrouter_cost_v1$;
ALTER FUNCTION public.lab_arena_reconcile_openrouter_cost_v1(
  TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, BIGINT, TEXT
) OWNER TO lab_arena_owner;

DO $lab_arena_225_function_acl$
DECLARE
  v_role TEXT;
BEGIN
  EXECUTE 'REVOKE ALL ON FUNCTION public.lab_arena_list_openrouter_cost_reconciliations_v1(text,text,bigint,integer) FROM PUBLIC';
  EXECUTE 'REVOKE ALL ON FUNCTION public.lab_arena_reconcile_openrouter_cost_v1(text,text,text,bigint,text,text,bigint,text) FROM PUBLIC';
  FOREACH v_role IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = v_role) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON FUNCTION public.lab_arena_list_openrouter_cost_reconciliations_v1(text,text,bigint,integer) FROM %I',
        v_role
      );
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON FUNCTION public.lab_arena_reconcile_openrouter_cost_v1(text,text,text,bigint,text,text,bigint,text) FROM %I',
        v_role
      );
    END IF;
  END LOOP;
  GRANT EXECUTE ON FUNCTION public.lab_arena_list_openrouter_cost_reconciliations_v1(
    TEXT, TEXT, BIGINT, INTEGER
  ) TO lab_arena_service;
  GRANT EXECUTE ON FUNCTION public.lab_arena_reconcile_openrouter_cost_v1(
    TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, BIGINT, TEXT
  ) TO lab_arena_service;
END;
$lab_arena_225_function_acl$;

COMMENT ON FUNCTION public.lab_arena_list_openrouter_cost_reconciliations_v1(
  TEXT, TEXT, BIGINT, INTEGER
) IS 'Service-only bounded exact-generation billing work list; credential fingerprints are never public.';
COMMENT ON FUNCTION public.lab_arena_reconcile_openrouter_cost_v1(
  TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, BIGINT, TEXT
) IS 'Service-only append-only settlement of one retained OpenRouter generation identity.';

NOTIFY pgrst, 'reload schema';
COMMIT;
