-- Reduce repeated run metadata probes in Arena cost aggregates.
-- Guard each exact live definition and preserve all billing and head rules.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $cost_407_call$
DECLARE
  v_definition TEXT;
  v_updated TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure('public.lab_arena__successful_call_cost_state(text,text,text)'))
    INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_407_call_missing';
  END IF;
  IF pg_catalog.md5(v_definition) = '0b7654d353830fe59d29816706b1af27' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) <> '4aa8c4b2d7e5c63a61f7168b7d1beacf' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_call_preimage_changed';
  END IF;
  v_updated := pg_catalog.replace(
    v_definition,
    $cost_407_call_0_old$  WITH heads AS ($cost_407_call_0_old$,
    $cost_407_call_0_new$  WITH matching_runs AS MATERIALIZED (
    SELECT runs.run_id
    FROM (
      SELECT DISTINCT ledger.run_id
      FROM public.lab_arena_ledger AS ledger
      WHERE ledger.submission_id = p_submission_id
        AND (p_provider IS NULL OR ledger.provider = p_provider)
        AND ledger.call_identity IS NOT NULL
    ) AS scoped
    JOIN public.lab_arena_runs AS runs ON runs.run_id = scoped.run_id
    WHERE runs.kind = p_kind
  ), heads AS ($cost_407_call_0_new$);
  v_updated := pg_catalog.replace(
    v_updated,
    $cost_407_call_1_old$    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND runs.kind = p_kind$cost_407_call_1_old$,
    $cost_407_call_1_new$    JOIN matching_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id$cost_407_call_1_new$);
  IF pg_catalog.md5(v_updated) <> '0b7654d353830fe59d29816706b1af27' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_call_postimage_invalid';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure('public.lab_arena__successful_call_cost_state(text,text,text)'))) <> '0b7654d353830fe59d29816706b1af27' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_call_readback_mismatch';
  END IF;
END;
$cost_407_call$;

DO $cost_407_icp$
DECLARE
  v_definition TEXT;
  v_updated TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)'))
    INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_407_icp_missing';
  END IF;
  IF pg_catalog.md5(v_definition) = '21027e8dd1401901c05e72887d4300e6' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) <> 'ec6781f2d8f96c67b8f611f78545bbd6' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_icp_preimage_changed';
  END IF;
  v_updated := pg_catalog.replace(
    v_definition,
    $cost_407_icp_0_old$  WITH heads AS ($cost_407_icp_0_old$,
    $cost_407_icp_0_new$  WITH matching_runs AS MATERIALIZED (
    SELECT runs.run_id
    FROM (
      SELECT DISTINCT ledger.run_id
      FROM public.lab_arena_ledger AS ledger
      WHERE ledger.round_id = p_round_id
        AND ledger.submission_id = p_submission_id
        AND ledger.call_identity IS NOT NULL
    ) AS scoped
    JOIN public.lab_arena_runs AS runs ON runs.run_id = scoped.run_id
    WHERE runs.kind = 'execute'
      AND runs.icp_position = p_icp_position
  ), heads AS ($cost_407_icp_0_new$);
  v_updated := pg_catalog.replace(
    v_updated,
    $cost_407_icp_1_old$    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.round_id = p_round_id
      AND ledger.submission_id = p_submission_id
      AND runs.kind = 'execute'
      AND runs.icp_position = p_icp_position$cost_407_icp_1_old$,
    $cost_407_icp_1_new$    JOIN matching_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.round_id = p_round_id
      AND ledger.submission_id = p_submission_id$cost_407_icp_1_new$);
  IF pg_catalog.md5(v_updated) <> '21027e8dd1401901c05e72887d4300e6' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_icp_postimage_invalid';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)'))) <> '21027e8dd1401901c05e72887d4300e6' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_icp_readback_mismatch';
  END IF;
END;
$cost_407_icp$;

DO $cost_407_submission$
DECLARE
  v_definition TEXT;
  v_updated TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure('public.lab_arena_submission_costs(text)'))
    INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_407_submission_missing';
  END IF;
  IF pg_catalog.md5(v_definition) = '31fbeff7e4f975e5496f98205dd1f7df' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) <> '4610064cc9dfdc2b2b6fc8a288845c28' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_submission_preimage_changed';
  END IF;
  v_updated := pg_catalog.replace(
    v_definition,
    $cost_407_submission_0_old$    SELECT DISTINCT runs.kind, ledger.provider
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND ledger.call_identity IS NOT NULL$cost_407_submission_0_old$,
    $cost_407_submission_0_new$    WITH run_kinds AS MATERIALIZED (
      SELECT runs.run_id, runs.kind
      FROM (
        SELECT DISTINCT ledger.run_id
        FROM public.lab_arena_ledger AS ledger
        WHERE ledger.submission_id = p_submission_id
          AND ledger.call_identity IS NOT NULL
      ) AS scoped
      JOIN public.lab_arena_runs AS runs ON runs.run_id = scoped.run_id
    )
    SELECT DISTINCT runs.kind, ledger.provider
    FROM public.lab_arena_ledger AS ledger
    JOIN run_kinds AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND ledger.call_identity IS NOT NULL$cost_407_submission_0_new$);
  IF pg_catalog.md5(v_updated) <> '31fbeff7e4f975e5496f98205dd1f7df' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_submission_postimage_invalid';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure('public.lab_arena_submission_costs(text)'))) <> '31fbeff7e4f975e5496f98205dd1f7df' THEN
    RAISE EXCEPTION 'lab_arena_cost_407_submission_readback_mismatch';
  END IF;
END;
$cost_407_submission$;

-- Keep the existing security boundary explicit after CREATE OR REPLACE.
ALTER FUNCTION public.lab_arena__successful_call_cost_state(TEXT,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_call_cost_state(TEXT,TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
ALTER FUNCTION public.lab_arena__successful_icp_cost_state(TEXT,TEXT,INTEGER) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_icp_cost_state(TEXT,TEXT,INTEGER)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
ALTER FUNCTION public.lab_arena_submission_costs(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_submission_costs(TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_submission_costs(TEXT)
  TO lab_arena_service;
NOTIFY pgrst, 'reload schema';
COMMIT;
