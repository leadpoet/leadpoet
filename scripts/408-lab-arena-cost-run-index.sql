-- Scope run metadata through a compact submission/run index. Final ledger
-- round, provider, head, and cost filters remain exactly as in migration 407.
BEGIN;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '45s';

DO $index_408$
DECLARE
  v_definition TEXT;
  v_valid BOOLEAN;
  v_ready BOOLEAN;
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_408_ledger_missing';
  END IF;
  IF pg_catalog.to_regclass('public.lab_arena_ledger_cost_run_idx') IS NULL THEN
    CREATE INDEX lab_arena_ledger_cost_run_idx
      ON public.lab_arena_ledger (submission_id, run_id)
      WHERE call_identity IS NOT NULL;
  END IF;
  SELECT pg_catalog.pg_get_indexdef(i.indexrelid), i.indisvalid, i.indisready
    INTO v_definition, v_valid, v_ready
  FROM pg_catalog.pg_index AS i
  WHERE i.indexrelid = pg_catalog.to_regclass('public.lab_arena_ledger_cost_run_idx')
    AND i.indrelid = 'public.lab_arena_ledger'::pg_catalog.regclass;
  IF v_definition IS DISTINCT FROM
       'CREATE INDEX lab_arena_ledger_cost_run_idx ON public.lab_arena_ledger USING btree (submission_id, run_id) WHERE (call_identity IS NOT NULL)'
     OR v_valid IS DISTINCT FROM TRUE
     OR v_ready IS DISTINCT FROM TRUE THEN
    RAISE EXCEPTION 'lab_arena_cost_408_index_shape_invalid';
  END IF;
END;
$index_408$;

DO $scope_408_call$
DECLARE
  v_definition TEXT;
  v_updated TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure('public.lab_arena__successful_call_cost_state(text,text,text)'))
    INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_408_call_missing';
  END IF;
  IF pg_catalog.md5(v_definition) = '82e3bc799dc185d7034771ff46542985' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) <> '0b7654d353830fe59d29816706b1af27' THEN
    RAISE EXCEPTION 'lab_arena_cost_408_call_preimage_changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition,
    $scope_408_call_old$      WHERE ledger.submission_id = p_submission_id
        AND (p_provider IS NULL OR ledger.provider = p_provider)
        AND ledger.call_identity IS NOT NULL$scope_408_call_old$,
    $scope_408_call_new$      WHERE ledger.submission_id = p_submission_id
        AND ledger.call_identity IS NOT NULL$scope_408_call_new$);
  IF pg_catalog.md5(v_updated) <> '82e3bc799dc185d7034771ff46542985' THEN
    RAISE EXCEPTION 'lab_arena_cost_408_call_postimage_invalid';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure('public.lab_arena__successful_call_cost_state(text,text,text)'))) <> '82e3bc799dc185d7034771ff46542985' THEN
    RAISE EXCEPTION 'lab_arena_cost_408_call_readback_mismatch';
  END IF;
END;
$scope_408_call$;

DO $scope_408_icp$
DECLARE
  v_definition TEXT;
  v_updated TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)'))
    INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_408_icp_missing';
  END IF;
  IF pg_catalog.md5(v_definition) = 'f220f32be89411ddb8af6deda805c21e' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) <> '21027e8dd1401901c05e72887d4300e6' THEN
    RAISE EXCEPTION 'lab_arena_cost_408_icp_preimage_changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition,
    $scope_408_icp_old$      WHERE ledger.round_id = p_round_id
        AND ledger.submission_id = p_submission_id
        AND ledger.call_identity IS NOT NULL$scope_408_icp_old$,
    $scope_408_icp_new$      WHERE ledger.submission_id = p_submission_id
        AND ledger.call_identity IS NOT NULL$scope_408_icp_new$);
  IF pg_catalog.md5(v_updated) <> 'f220f32be89411ddb8af6deda805c21e' THEN
    RAISE EXCEPTION 'lab_arena_cost_408_icp_postimage_invalid';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)'))) <> 'f220f32be89411ddb8af6deda805c21e' THEN
    RAISE EXCEPTION 'lab_arena_cost_408_icp_readback_mismatch';
  END IF;
END;
$scope_408_icp$;

ALTER FUNCTION public.lab_arena__successful_call_cost_state(TEXT,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_call_cost_state(TEXT,TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
ALTER FUNCTION public.lab_arena__successful_icp_cost_state(TEXT,TEXT,INTEGER) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_icp_cost_state(TEXT,TEXT,INTEGER)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
NOTIFY pgrst, 'reload schema';
COMMIT;
