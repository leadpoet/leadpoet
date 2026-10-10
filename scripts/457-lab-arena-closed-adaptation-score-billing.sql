-- Keep exact paid adaptation-error judge calls billable after publication.
-- This changes no score, accepted result, admission amount, or provider request.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $closed_adaptation_prerequisite$
DECLARE
  v_existing TEXT;
  v_security JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_existing,v_security
    FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
    WHERE p.oid=pg_catalog.to_regprocedure(
      'public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)');
  IF v_existing IS NOT NULL AND (
       pg_catalog.md5(v_existing)<>'4552bb19ed4834b2de11e3da523e9028'
       OR v_security IS DISTINCT FROM pg_catalog.jsonb_build_array(
         'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',
         TRUE,'s',ARRAY['search_path=pg_catalog, public'])) THEN
    RAISE EXCEPTION 'closed adaptation billing helper already differs';
  END IF;
END;
$closed_adaptation_prerequisite$;

CREATE OR REPLACE FUNCTION public.lab_arena__closed_score_adaptation_uncertainty_v1(
  p_uncertain_entry_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $closed_score_adaptation$
  -- lab_arena_closed_score_adaptation_billing_457
  SELECT COALESCE((
    SELECT
      uncertainty.entry_kind = 'uncertain'
      AND uncertainty.provider = 'deepline'
      AND uncertainty.funding_source IN ('host', 'miner_key')
      AND runs.kind = 'score'
      AND runs.status = 'accepted'
      AND runs.terminal_cause = 'accepted'
      AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
      AND uncertainty.entry_doc #>> '{call,reason}' = 'settle_failure'
      AND uncertainty.entry_doc #>> '{call,failure_stage}' = 'response_adaptation'
      AND uncertainty.entry_doc #>> '{call,error_class}' = 'CompatibilityResponseError'
      AND uncertainty.entry_doc #> '{call,call_succeeded}' = 'false'::JSONB
      AND reservation.entry_doc ->> 'deepline_execution_key' =
          'arena:' || pg_catalog.substr(uncertainty.call_identity, 8)
      AND public.lab_arena__deepline_cost_binding_v1(
        uncertainty.entry_doc, reservation.entry_doc, uncertainty.call_identity
      )
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
$closed_score_adaptation$;
ALTER FUNCTION public.lab_arena__closed_score_adaptation_uncertainty_v1(BIGINT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__closed_score_adaptation_uncertainty_v1(BIGINT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $closed_adaptation_helper_readback$
BEGIN
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       'public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)'::REGPROCEDURE))
       <> '4552bb19ed4834b2de11e3da523e9028' THEN
    RAISE EXCEPTION 'closed adaptation billing helper readback differs';
  END IF;
END;
$closed_adaptation_helper_readback$;

DO $closed_adaptation_functions$
DECLARE
  v_signatures CONSTANT TEXT[] := ARRAY[
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)',
    'public.lab_arena_list_deepline_cost_reconciliations_v3(text,text,bigint,integer)',
    'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)',
    'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)'
  ];
  v_before CONSTANT TEXT[] := ARRAY[
    '029aad69dc9bc570bed8322c3b069d8d',
    'e680ed9de2bfe11571fae621b3e77f4d',
    'efa38335a0c082e73c2d45d27b3c22a4',
    '36e64ac1babd07f2459f4d5de35680a6'
  ];
  v_after CONSTANT TEXT[] := ARRAY[
    'b1b891991482cd99c8c06a22c71178f3', '7feabbe2544e2e3418a611ed865bacd3', '8a12f86bc9de2584a9bb9d481108cfb4', '83c43a96f555d2b70ca54eb614d27aa7'
  ];
  v_definition TEXT;
  v_updated TEXT;
  v_old TEXT;
  v_new TEXT;
  v_security JSONB;
  v_security_after JSONB;
  v_index INTEGER;
BEGIN
  FOR v_index IN 1..4 LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),
      pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
      INTO v_definition,v_security
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
      WHERE p.oid=pg_catalog.to_regprocedure(v_signatures[v_index]);
    IF v_security IS DISTINCT FROM pg_catalog.jsonb_build_array(
        'lab_arena_owner',
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
        TRUE,CASE WHEN v_index < 4 THEN 's' ELSE 'v' END,
        ARRAY['search_path=pg_catalog, public']) THEN
      RAISE EXCEPTION 'closed adaptation billing security shape differs';
    END IF;
    IF pg_catalog.md5(v_definition)=v_after[v_index] THEN
      CONTINUE;
    ELSIF pg_catalog.md5(v_definition)<>v_before[v_index] THEN
      RAISE EXCEPTION 'closed adaptation billing preimage differs: %',v_signatures[v_index];
    END IF;
    IF v_index IN (1,2) THEN
      v_old := E'                    ) OR public.lab_arena__closed_miner_score_success_uncertainty_v1(\n                      uncertainty.entry_id\n                    ))';
      v_new := E'                    ) OR public.lab_arena__closed_miner_score_success_uncertainty_v1(\n                      uncertainty.entry_id\n                    ) OR public.lab_arena__closed_score_adaptation_uncertainty_v1(\n                      uncertainty.entry_id\n                    ))';
    ELSIF v_index=3 THEN
      v_old := E'      OR public.lab_arena__closed_miner_score_success_uncertainty_v1(uncertainty.entry_id)\n    )';
      v_new := E'      OR public.lab_arena__closed_miner_score_success_uncertainty_v1(uncertainty.entry_id)\n      OR public.lab_arena__closed_score_adaptation_uncertainty_v1(uncertainty.entry_id)\n    )';
    ELSE
      v_old := E'       ))\n     ), FALSE) THEN';
      v_new := E'       ) OR public.lab_arena__closed_score_adaptation_uncertainty_v1(\n         CASE\n           WHEN v_head.entry_kind = ''uncertain'' THEN v_head.entry_id\n           WHEN v_head.entry_kind = ''settlement''\n             AND v_head.entry_doc ->> ''deepline_delayed_reconciliation'' = ''true''\n             AND v_head.entry_doc ->> ''reconciled_uncertainty_entry_id'' ~ ''^[0-9]{1,18}$''\n           THEN (v_head.entry_doc ->> ''reconciled_uncertainty_entry_id'')::BIGINT\n           ELSE NULL\n         END\n       ))\n     ), FALSE) THEN';
    END IF;
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
          pg_catalog.replace(v_definition,v_old,'')))<>pg_catalog.length(v_old) THEN
      RAISE EXCEPTION 'closed adaptation billing replacement scope differs: %',v_signatures[v_index];
    END IF;
    v_updated:=pg_catalog.replace(v_definition,v_old,v_new);
    IF pg_catalog.md5(v_updated)<>v_after[v_index] THEN
      RAISE EXCEPTION 'closed adaptation billing target differs: %',v_signatures[v_index];
    END IF;
    EXECUTE v_updated;
    SELECT pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
      INTO v_security_after
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
      WHERE p.oid=pg_catalog.to_regprocedure(v_signatures[v_index]);
    IF pg_catalog.md5(pg_catalog.pg_get_functiondef(v_signatures[v_index]::REGPROCEDURE))<>v_after[v_index]
       OR v_security_after IS DISTINCT FROM v_security THEN
      RAISE EXCEPTION 'closed adaptation billing readback differs: %',v_signatures[v_index];
    END IF;
  END LOOP;
END;
$closed_adaptation_functions$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
