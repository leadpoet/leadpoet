-- Recover only exact successful miner-key Deepline score bills after a run closes.
-- The shared failed-call admission helper, score, and publication stay intact.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $miner_billing_prerequisites$
DECLARE
  v_existing TEXT;
  v_identity JSONB;
  v_host_identity JSONB;
BEGIN
  IF pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE') THEN
    RAISE EXCEPTION 'closed miner billing schema authority changed';
  END IF;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)'::REGPROCEDURE))
       <> '7312e9ca80018f634cd3799cd4a90093'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__submission_kind_admission_spend_v1(text,text,boolean)'::REGPROCEDURE))
       <> 'e063c9c4b8537fa63478a75ce4e66392'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__submission_kind_has_admission_uncertainty_v1(text,text)'::REGPROCEDURE))
       <> '01aa529f7e13e89d98f0a911e9cdbbb7'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)'::REGPROCEDURE))
       <> 'b096b9440dd66037b9f286632c5b27a9' THEN
    RAISE EXCEPTION 'closed miner billing prerequisite changed';
  END IF;
  SELECT pg_catalog.jsonb_build_array(
           owner.rolname, p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig)
    INTO v_host_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = 'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)'::REGPROCEDURE;
  IF v_host_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner', '{lab_arena_owner=X/lab_arena_owner}',
      TRUE, 's', ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'closed miner billing host security changed';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(
           owner.rolname, p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig)
    INTO v_existing, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure(
    'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)');
  IF v_existing IS NOT NULL AND (
      pg_catalog.md5(v_existing) <> '2827d938ea977794ef3427aa9d7517dc'
      OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
        'lab_arena_owner', '{lab_arena_owner=X/lab_arena_owner}',
        TRUE, 's', ARRAY['search_path=pg_catalog, public'])) THEN
    RAISE EXCEPTION 'closed miner billing helper already differs';
  END IF;
END;
$miner_billing_prerequisites$;

DO $miner_billing_create_authority$
BEGIN
  GRANT CREATE ON SCHEMA public TO lab_arena_owner;
END;
$miner_billing_create_authority$;

CREATE OR REPLACE FUNCTION public.lab_arena__closed_miner_score_success_uncertainty_v1(
  p_uncertain_entry_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $closed_miner_score_success_uncertainty$
  SELECT COALESCE((
    SELECT
      uncertainty.entry_kind = 'uncertain'
      AND uncertainty.provider = 'deepline'
      AND uncertainty.funding_source = 'miner_key'
      AND runs.kind = 'score'
      AND (
        (runs.status = 'failed'
         AND runs.terminal_cause IN (
           'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
           'stage_closed', 'judge_error', 'judge_timeout'
         ))
        OR (runs.status = 'accepted' AND runs.terminal_cause = 'accepted')
      )
      AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
      AND uncertainty.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
      AND uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
      AND uncertainty.entry_doc #> '{call,provider_status}' = '200'::JSONB
      AND reservation.entry_doc ->> 'deepline_execution_key' =
          'arena:' || pg_catalog.substr(uncertainty.call_identity, 8)
      AND uncertainty.entry_doc #> '{call,deepline_execution_key}' =
          reservation.entry_doc -> 'deepline_execution_key'
      -- This existing binding proves deterministic request, operation and
      -- credential identity. It never supplies or guesses a charge amount.
      AND CASE WHEN uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
        THEN public.lab_arena__deepline_cost_binding_v1(
          uncertainty.entry_doc, reservation.entry_doc, uncertainty.call_identity
        ) ELSE FALSE END
      AND EXISTS (
        SELECT 1 FROM public.lab_arena_ledger AS dispatch
        WHERE dispatch.call_identity = uncertainty.call_identity
          AND dispatch.entry_kind = 'dispatch'
          AND dispatch.run_id = uncertainty.run_id
          AND dispatch.round_id = uncertainty.round_id
          AND dispatch.submission_id = uncertainty.submission_id
          AND dispatch.miner_hotkey = uncertainty.miner_hotkey
          AND dispatch.stage = uncertainty.stage
          AND dispatch.provider = uncertainty.provider
          AND dispatch.operation_id = uncertainty.operation_id
          AND dispatch.funding_source = uncertainty.funding_source
      )
    FROM public.lab_arena_ledger AS uncertainty
    JOIN public.lab_arena_ledger AS reservation
      ON reservation.call_identity = uncertainty.call_identity
     AND reservation.entry_kind = 'reservation'
     AND reservation.run_id = uncertainty.run_id
     AND reservation.round_id = uncertainty.round_id
     AND reservation.submission_id = uncertainty.submission_id
     AND reservation.miner_hotkey = uncertainty.miner_hotkey
     AND reservation.stage = uncertainty.stage
     AND reservation.provider = uncertainty.provider
     AND reservation.operation_id = uncertainty.operation_id
     AND reservation.funding_source = uncertainty.funding_source
    JOIN public.lab_arena_runs AS runs
      ON runs.run_id = uncertainty.run_id
     AND runs.round_id = uncertainty.round_id
     AND runs.submission_id = uncertainty.submission_id
     AND runs.miner_hotkey = uncertainty.miner_hotkey
     AND runs.stage = uncertainty.stage
    WHERE uncertainty.entry_id = p_uncertain_entry_id
  ), FALSE);
$closed_miner_score_success_uncertainty$;
ALTER FUNCTION public.lab_arena__closed_miner_score_success_uncertainty_v1(BIGINT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__closed_miner_score_success_uncertainty_v1(BIGINT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $miner_billing_closed_functions$
DECLARE
  v_signatures CONSTANT TEXT[] := ARRAY[
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)',
    'public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)',
    'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)',
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)',
    'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)'
  ];
  v_before CONSTANT TEXT[] := ARRAY[
    'caaae6224f604eb9213e73af2b893529',
    '48e83d10b78deb039d8b1fefd21ca089',
    'd90ae5ac9aa286d9daa00788b183013b',
    '94d6c32274ed57106a12899f40b710a9',
    '7eadc08cbbe29619cec802ce63489a03'
  ];
  v_after CONSTANT TEXT[] := ARRAY[
    '029aad69dc9bc570bed8322c3b069d8d',
    '82531576cffccfea2937eb215d47c89a',
    'efa38335a0c082e73c2d45d27b3c22a4',
    'a37f1d7cae52a3fa7f0c44cc3b52ec40',
    'b8e84308a78ecb7adeec52d5c4ca13a9'
  ];
  v_arg TEXT;
  v_old TEXT;
  v_new TEXT;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_expected_identity JSONB;
  v_helper_identity JSONB;
  v_index INTEGER;
BEGIN
  FOR v_index IN 1..5 LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),
           pg_catalog.jsonb_build_array(
             owner.rolname, p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig)
      INTO v_definition, v_identity
    FROM pg_catalog.pg_proc AS p
    JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
    WHERE p.oid = pg_catalog.to_regprocedure(v_signatures[v_index]);
    v_expected_identity := pg_catalog.jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE, CASE WHEN v_index <= 3 THEN 's' ELSE 'v' END,
      ARRAY['search_path=pg_catalog, public']);
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM v_expected_identity THEN
      RAISE EXCEPTION 'closed miner billing security shape changed';
    END IF;
    IF pg_catalog.md5(v_definition) = v_after[v_index] THEN
      CONTINUE;
    ELSIF pg_catalog.md5(v_definition) <> v_before[v_index] THEN
      RAISE EXCEPTION 'closed miner billing preimage differs';
    END IF;
    IF v_index = 3 THEN
      v_old := E'      OR public.lab_arena__closed_host_score_success_uncertainty_v1(uncertainty.entry_id)\n    )';
      v_new := E'      OR public.lab_arena__closed_host_score_success_uncertainty_v1(uncertainty.entry_id)\n      OR public.lab_arena__closed_miner_score_success_uncertainty_v1(uncertainty.entry_id)\n    )';
    ELSE
      IF v_index <= 2 THEN
        v_arg := E'\n                      uncertainty.entry_id\n                    ';
      ELSE
        v_arg := $argument$
         CASE
           WHEN v_head.entry_kind = 'uncertain' THEN v_head.entry_id
           WHEN v_head.entry_kind = 'settlement'
             AND v_head.entry_doc ->> 'deepline_delayed_reconciliation' = 'true'
             AND v_head.entry_doc ->> 'reconciled_uncertainty_entry_id' ~ '^[0-9]{1,18}$'
           THEN (v_head.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
           ELSE NULL
         END
       $argument$;
      END IF;
      v_old := 'public.lab_arena__closed_host_score_success_uncertainty_v1(' || v_arg || '))';
      v_new := 'public.lab_arena__closed_host_score_success_uncertainty_v1(' || v_arg
        || ') OR public.lab_arena__closed_miner_score_success_uncertainty_v1(' || v_arg || '))';
    END IF;
    IF (pg_catalog.length(v_definition) - pg_catalog.length(
          pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.length(v_old) <> 1 THEN
      RAISE EXCEPTION 'closed miner billing replacement scope differs';
    END IF;
    v_updated := pg_catalog.replace(v_definition, v_old, v_new);
    IF pg_catalog.md5(v_updated) <> v_after[v_index] THEN
      RAISE EXCEPTION 'closed miner billing postimage differs';
    END IF;
    EXECUTE v_updated;
    SELECT pg_catalog.pg_get_functiondef(p.oid),
           pg_catalog.jsonb_build_array(
             owner.rolname, p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig)
      INTO v_definition, v_identity
    FROM pg_catalog.pg_proc AS p
    JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
    WHERE p.oid = pg_catalog.to_regprocedure(v_signatures[v_index]);
    IF pg_catalog.md5(v_definition) <> v_after[v_index]
       OR v_identity IS DISTINCT FROM v_expected_identity THEN
      RAISE EXCEPTION 'closed miner billing readback differs';
    END IF;
  END LOOP;
  SELECT pg_catalog.jsonb_build_array(
           owner.rolname, p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig)
    INTO v_helper_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = 'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)'::REGPROCEDURE;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)'::REGPROCEDURE))
       <> '2827d938ea977794ef3427aa9d7517dc'
     OR v_helper_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
        'lab_arena_owner', '{lab_arena_owner=X/lab_arena_owner}',
        TRUE, 's', ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'closed miner billing helper readback differs';
  END IF;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)'::REGPROCEDURE))
       <> '7312e9ca80018f634cd3799cd4a90093'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__submission_kind_admission_spend_v1(text,text,boolean)'::REGPROCEDURE))
       <> 'e063c9c4b8537fa63478a75ce4e66392'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__submission_kind_has_admission_uncertainty_v1(text,text)'::REGPROCEDURE))
       <> '01aa529f7e13e89d98f0a911e9cdbbb7'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
      'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)'::REGPROCEDURE))
       <> 'b096b9440dd66037b9f286632c5b27a9' THEN
    RAISE EXCEPTION 'closed miner billing preserved functions changed';
  END IF;
END;
$miner_billing_closed_functions$;

-- Keep the closed helper owner-only; restore any temporary CREATE privilege.
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
