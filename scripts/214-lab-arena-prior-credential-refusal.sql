-- Distinguish a real budget cap from a retry blocked by an unknown provider
-- charge. A proven miner credential refusal has precedence. Every uncertain
-- charge remains locked at its full reservation; this changes classification
-- evidence only.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_214_prior_credential_refusal$
DECLARE
  v_definition TEXT;
  v_declaration_old TEXT := $old$
  v_reason TEXT := NULL;
  v_expires TIMESTAMPTZ;
$old$;
  v_declaration_new TEXT := $new$
  v_reason TEXT := NULL;
  v_prior_miner_credential_refusal BOOLEAN := FALSE;
  v_has_uncertain_provider_cost BOOLEAN := FALSE;
  v_settled_spend BIGINT := 0;
  v_expires TIMESTAMPTZ;
$new$;
  v_refusal_old TEXT := $old$
  IF v_reason IS NOT NULL THEN
    INSERT INTO public.lab_arena_ledger (
$old$;
  v_refusal_new TEXT := $new$
  IF v_reason IS NOT NULL THEN
    IF p_funding_source = 'miner_key' THEN
      SELECT EXISTS (
        SELECT 1
        FROM (
          SELECT DISTINCT ON (ledger.call_identity)
            ledger.entry_kind, ledger.entry_doc, ledger.run_id
          FROM public.lab_arena_ledger AS ledger
          WHERE ledger.submission_id = v_run.submission_id
            AND ledger.provider = p_provider
            AND ledger.funding_source = 'miner_key'
            AND ledger.call_identity IS NOT NULL
          ORDER BY ledger.call_identity, ledger.entry_id DESC
        ) AS head
        JOIN public.lab_arena_runs AS prior_run
          ON prior_run.run_id = head.run_id
        WHERE head.entry_kind = 'uncertain'
          AND prior_run.kind = v_run.kind
          AND head.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
          AND head.entry_doc #>> '{call,provider_status}'
            IN ('401', '402', '403')
      ) INTO v_prior_miner_credential_refusal;
    END IF;
    IF v_reason = 'money_cap'
       AND NOT v_prior_miner_credential_refusal THEN
      SELECT
        COUNT(*) FILTER (
          WHERE head.entry_kind = 'uncertain'
        ) > 0,
        COALESCE(SUM(head.amount_microusd) FILTER (
          WHERE head.entry_kind = 'settlement'
        ), 0)::BIGINT
      INTO v_has_uncertain_provider_cost, v_settled_spend
      FROM (
        SELECT DISTINCT ON (ledger.call_identity)
          ledger.entry_kind, ledger.amount_microusd
        FROM public.lab_arena_ledger AS ledger
        JOIN public.lab_arena_runs AS prior_run
          ON prior_run.run_id = ledger.run_id
        WHERE ledger.submission_id = v_run.submission_id
          AND prior_run.kind = v_run.kind
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      ) AS head;
      IF v_has_uncertain_provider_cost
         AND (
           (v_dynamic AND v_settled_spend < v_money_cap)
           OR (
             NOT v_dynamic
             AND v_settled_spend <= v_money_cap - p_amount_microusd
           )
         ) THEN
        v_reason := 'provider_cost_uncertain';
      END IF;
    END IF;
    INSERT INTO public.lab_arena_ledger (
$new$;
  v_return_old TEXT := $old$
    RETURN pg_catalog.jsonb_build_object(
      'status', 'refused', 'idempotent', FALSE, 'reason', v_reason,
      'call_identity', p_call_identity, 'lease_expires_at', v_expires
    );
$old$;
  v_return_new TEXT := $new$
    RETURN pg_catalog.jsonb_build_object(
      'status', 'refused', 'idempotent', FALSE, 'reason', v_reason,
      'prior_miner_credential_refusal', v_prior_miner_credential_refusal,
      'call_identity', p_call_identity, 'lease_expires_at', v_expires
    );
$new$;
  v_entry_old TEXT := $old$
      pg_catalog.jsonb_build_object(
        'reason', v_reason, 'requested_microusd', p_amount_microusd,
        'call', p_call_doc
      )
$old$;
  v_entry_new TEXT := $new$
      pg_catalog.jsonb_build_object(
        'reason', v_reason, 'requested_microusd', p_amount_microusd,
        'prior_miner_credential_refusal', v_prior_miner_credential_refusal,
        'call', p_call_doc
      )
$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply 206-lab-arena-combined-provider-budget.sql first';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_reserve_call'
    AND procedure.pronargs = 9;
  IF pg_catalog.strpos(
       v_definition, 'v_prior_miner_credential_refusal'
     ) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_declaration_old) = 0
     OR pg_catalog.strpos(v_definition, v_refusal_old) = 0
     OR pg_catalog.strpos(v_definition, v_return_old) = 0
     OR pg_catalog.strpos(v_definition, v_entry_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_reserve_call_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(
    v_definition, v_declaration_old, v_declaration_new
  );
  v_definition := pg_catalog.replace(
    v_definition, v_refusal_old, v_refusal_new
  );
  v_definition := pg_catalog.replace(
    v_definition, v_return_old, v_return_new
  );
  v_definition := pg_catalog.replace(
    v_definition, v_entry_old, v_entry_new
  );
  EXECUTE v_definition;
END;
$lab_arena_214_prior_credential_refusal$;

DO $lab_arena_214_replay_credential_refusal$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$
    'reason', p_head.entry_doc ->> 'reason',
    'lease_expires_at', p_run.lease_expires_at
$old$;
  v_new TEXT := $new$
    'reason', p_head.entry_doc ->> 'reason',
    'prior_miner_credential_refusal', COALESCE(
      (p_head.entry_doc ->> 'prior_miner_credential_refusal')::BOOLEAN,
      FALSE
    ),
    'lease_expires_at', p_run.lease_expires_at
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena__call_state_view'
    AND procedure.pronargs = 2;
  IF pg_catalog.strpos(
       v_definition, 'prior_miner_credential_refusal'
     ) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_call_state_view_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$lab_arena_214_replay_credential_refusal$;

NOTIFY pgrst, 'reload schema';
COMMIT;
