-- Reconcile a completed provider reply that reaches the authenticated gateway
-- after cancellation closed its exact leased run. The cancellation uncertainty
-- remains immutable; one later settlement becomes the authoritative ledger head.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_223_prerequisites$
DECLARE
  v_definition TEXT;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__lock_current_lease(text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__ledger_head(text)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply 206-lab-arena-combined-provider-budget.sql first';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_settle_call'
    AND procedure.pronargs = 6;
  IF v_definition IS NULL
     OR pg_catalog.strpos(v_definition, 'v_submission public.lab_arena_submissions') = 0
     OR pg_catalog.strpos(v_definition, '''variance_microusd''') = 0 THEN
    RAISE EXCEPTION 'lab_arena_settle_call_shape_unexpected';
  END IF;
  IF pg_catalog.to_regclass('public.lab_arena_ledger_terminal_uq') IS NULL
     AND (
       pg_catalog.to_regclass('public.lab_arena_ledger_settlement_uq') IS NULL
       OR pg_catalog.to_regclass('public.lab_arena_ledger_nonsettlement_terminal_uq') IS NULL
     ) THEN
    RAISE EXCEPTION 'lab_arena_ledger_terminal_indexes_missing';
  END IF;
END;
$lab_arena_223_prerequisites$;

-- Keep one immutable uncertainty/refusal/recovery, while allowing exactly one
-- settlement to follow an uncertainty. The RPC below is the only writer that
-- admits that transition.
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_ledger_settlement_uq
  ON public.lab_arena_ledger (call_identity)
  WHERE entry_kind = 'settlement';
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_ledger_nonsettlement_terminal_uq
  ON public.lab_arena_ledger (call_identity)
  WHERE entry_kind IN ('uncertain', 'recovery', 'refusal');
DROP INDEX IF EXISTS public.lab_arena_ledger_terminal_uq;

CREATE OR REPLACE FUNCTION public.lab_arena_settle_call(
  p_run_id TEXT, p_lease_token_hash TEXT, p_call_identity TEXT,
  p_actual_microusd BIGINT, p_terminal_response JSONB,
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
  v_round public.lab_arena_rounds;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_submission public.lab_arena_submissions;
  v_round_id TEXT;
  v_expires TIMESTAMPTZ;
  v_late_cancel_settlement BOOLEAN := FALSE;
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
    -- Cancellation locks the round before it terminates calls and fails runs.
    -- Reacquire the same round-then-run lock order and accept only that exact
    -- cancellation outcome. A resumed run has a different/cleared lease hash.
    SELECT round_id INTO v_round_id
    FROM public.lab_arena_runs
    WHERE run_id = p_run_id;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_run_missing' USING ERRCODE = 'P0002';
    END IF;
    SELECT * INTO v_round
    FROM public.lab_arena_rounds
    WHERE round_id = v_round_id
    FOR SHARE;
    SELECT * INTO v_run
    FROM public.lab_arena_runs
    WHERE run_id = p_run_id
    FOR UPDATE;
    v_head := public.lab_arena__ledger_head(p_call_identity);

    IF v_round.status IS DISTINCT FROM 'cancelled'
       OR v_run.status IS DISTINCT FROM 'failed'
       OR v_run.terminal_cause IS DISTINCT FROM 'stage_closed'
       OR v_run.stage_generation + 1 IS DISTINCT FROM v_round.stage_generation
       OR v_run.terminal_doc ->> 'previous_status' IS DISTINCT FROM 'leased'
       OR COALESCE(v_run.terminal_doc ->> 'cancelled_at', '') = ''
       OR v_run.lease_token_hash IS NULL
       OR v_run.lease_token_hash IS DISTINCT FROM p_lease_token_hash
       OR v_head.entry_id IS NULL
       OR v_head.run_id IS DISTINCT FROM p_run_id
       OR v_head.round_id IS DISTINCT FROM v_run.round_id
       OR v_head.submission_id IS DISTINCT FROM v_run.submission_id THEN
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END IF;

    -- An exact replay of a late settlement is safe. Conflicting cost or
    -- terminal evidence is rejected and cannot change the stored charge.
    IF v_head.entry_kind = 'settlement'
       AND v_head.entry_doc ->> 'late_reconciliation' = 'true' THEN
      IF v_head.amount_microusd = p_actual_microusd
         AND v_head.terminal_response = p_terminal_response THEN
        RETURN pg_catalog.jsonb_build_object(
          'status', 'settled', 'idempotent', TRUE,
          'late_reconciliation', TRUE,
          'call_identity', p_call_identity,
          'actual_microusd', v_head.amount_microusd,
          'released_microusd',
            (v_head.entry_doc ->> 'released_microusd')::BIGINT,
          'variance_microusd',
            (v_head.entry_doc ->> 'variance_microusd')::BIGINT,
          'terminal_response', v_head.terminal_response,
          'lease_expires_at', v_run.lease_expires_at
        );
      END IF;
      RETURN pg_catalog.jsonb_build_object(
        'status', 'conflict', 'call_identity', p_call_identity
      );
    END IF;

    IF v_head.entry_kind IS DISTINCT FROM 'uncertain'
       OR v_head.entry_doc ->> 'reason' IS DISTINCT FROM 'round_cancelled'
       OR COALESCE(p_terminal_response ->> 'status', '') !~ '^[0-9]{3}$'
       OR (
         (p_terminal_response ->> 'status')::INTEGER NOT BETWEEN 200 AND 299
         AND (
           pg_catalog.jsonb_typeof(p_terminal_response -> 'provider_cost')
             IS DISTINCT FROM 'object'
           OR COALESCE(p_terminal_response #>> '{provider_cost,basis}', '') = ''
           OR COALESCE(p_terminal_response #>> '{provider_cost,units}', '') = ''
           OR COALESCE(p_terminal_response #>> '{provider_cost,unit_name}', '') = ''
           OR COALESCE(p_terminal_response #>> '{provider_cost,operation}', '') = ''
         )
       ) THEN
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END IF;
    v_late_cancel_settlement := TRUE;
  END;

  -- Match reserve's round, run, submission lock order. A settlement that
  -- exceeds its estimate is visible before another reservation can pass.
  SELECT * INTO v_submission FROM public.lab_arena_submissions
  WHERE submission_id = v_run.submission_id FOR NO KEY UPDATE;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NULL OR v_head.run_id <> p_run_id THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_reserved');
  END IF;
  IF NOT v_late_cancel_settlement AND v_head.entry_kind <> 'dispatch' THEN
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  IF v_late_cancel_settlement AND (
       v_head.entry_kind IS DISTINCT FROM 'uncertain'
       OR v_head.entry_doc ->> 'reason' IS DISTINCT FROM 'round_cancelled'
     ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  SELECT * INTO v_reservation FROM public.lab_arena_ledger
  WHERE call_identity = p_call_identity AND entry_kind = 'reservation';
  IF v_late_cancel_settlement AND (
       v_reservation.entry_id IS NULL
       OR v_reservation.run_id IS DISTINCT FROM p_run_id
       OR v_reservation.round_id IS DISTINCT FROM v_run.round_id
       OR v_reservation.submission_id IS DISTINCT FROM v_run.submission_id
       OR v_head.miner_hotkey IS DISTINCT FROM v_reservation.miner_hotkey
       OR v_head.round_id IS DISTINCT FROM v_reservation.round_id
       OR v_head.submission_id IS DISTINCT FROM v_reservation.submission_id
       OR v_head.stage IS DISTINCT FROM v_reservation.stage
       OR v_head.provider IS DISTINCT FROM v_reservation.provider
       OR v_head.operation_id IS DISTINCT FROM v_reservation.operation_id
       OR v_head.funding_source IS DISTINCT FROM v_reservation.funding_source
       OR NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_ledger AS dispatch
         WHERE dispatch.call_identity = p_call_identity
           AND dispatch.entry_kind = 'dispatch'
           AND dispatch.run_id = p_run_id
       )
     ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc, terminal_response
  ) VALUES (
    'settlement', v_reservation.miner_hotkey, v_reservation.round_id,
    v_reservation.submission_id, p_run_id, v_reservation.stage,
    p_call_identity, v_reservation.provider, v_reservation.operation_id,
    v_reservation.funding_source, p_actual_microusd,
    CASE WHEN v_late_cancel_settlement THEN
      pg_catalog.jsonb_build_object(
        'reserved_microusd', v_reservation.amount_microusd,
        'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
        'variance_microusd', p_actual_microusd - v_reservation.amount_microusd,
        'late_reconciliation', TRUE,
        'reconciled_uncertainty_entry_id', v_head.entry_id,
        'reconciled_uncertainty_reason', v_head.entry_doc ->> 'reason'
      )
    ELSE
      pg_catalog.jsonb_build_object(
        'reserved_microusd', v_reservation.amount_microusd,
        'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
        'variance_microusd', p_actual_microusd - v_reservation.amount_microusd
      )
    END,
    p_terminal_response
  );
  IF NOT v_late_cancel_settlement THEN
    v_expires := pg_catalog.clock_timestamp()
      + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
    UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
    WHERE run_id = p_run_id;
  ELSE
    v_expires := v_run.lease_expires_at;
  END IF;
  IF v_late_cancel_settlement THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'settled', 'idempotent', FALSE,
      'late_reconciliation', TRUE,
      'call_identity', p_call_identity, 'actual_microusd', p_actual_microusd,
      'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
      'variance_microusd', p_actual_microusd - v_reservation.amount_microusd,
      'terminal_response', p_terminal_response,
      'lease_expires_at', v_expires
    );
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'settled', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'actual_microusd', p_actual_microusd,
    'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
    'variance_microusd', p_actual_microusd - v_reservation.amount_microusd,
    'terminal_response', p_terminal_response,
    'lease_expires_at', v_expires
  );
END;
$lab_arena_settle_call$;
ALTER FUNCTION public.lab_arena_settle_call(
  TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
) OWNER TO lab_arena_owner;

DO $lab_arena_223_function_acl$
DECLARE
  v_role TEXT;
BEGIN
  EXECUTE 'REVOKE ALL ON FUNCTION public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer) FROM PUBLIC';
  FOREACH v_role IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = v_role) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON FUNCTION public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer) FROM %I',
        v_role
      );
    END IF;
  END LOOP;
  GRANT EXECUTE ON FUNCTION public.lab_arena_settle_call(
    TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
  ) TO lab_arena_service;
END;
$lab_arena_223_function_acl$;

NOTIFY pgrst, 'reload schema';
COMMIT;
