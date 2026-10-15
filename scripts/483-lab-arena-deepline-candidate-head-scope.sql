-- Filter scoped/latest uncertainty heads before the existing full V1 validation.
-- All four pinned published-round helpers require a score run. Exclude execute
-- rows only for published rounds; cancelled and active liabilities stay eligible.
-- Keep every original final guard, helper, order and limit. No other RPC changes.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $deepline_candidate_head_scope$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)';
  v_helper TEXT;
  v_hash TEXT;
  v_definition TEXT;
  v_updated TEXT;
  v_security JSONB;
  v_security_after JSONB;
BEGIN
  -- The cheap score-only implication is valid only for these exact helpers.
  FOR v_helper, v_hash IN SELECT * FROM (VALUES
    ('public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)', '7312e9ca80018f634cd3799cd4a90093'),
    ('public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)', 'b096b9440dd66037b9f286632c5b27a9'),
    ('public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)', '2827d938ea977794ef3427aa9d7517dc'),
    ('public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)', '4552bb19ed4834b2de11e3da523e9028')
  ) AS expected(signature, definition_hash) LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),
      pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
      INTO v_definition,v_security
    FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
    WHERE p.oid=pg_catalog.to_regprocedure(v_helper);
    IF pg_catalog.md5(v_definition) IS DISTINCT FROM v_hash
       OR v_security IS DISTINCT FROM pg_catalog.jsonb_build_array(
         'lab_arena_owner', '{lab_arena_owner=X/lab_arena_owner}',
         TRUE, 's', ARRAY['search_path=pg_catalog, public']) THEN
      RAISE EXCEPTION 'Deepline candidate head helper prerequisite changed: %',v_helper;
    END IF;
  END LOOP;

  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_security
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_security IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE, 's', ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'Deepline candidate head security shape changed';
  END IF;
  IF pg_catalog.md5(v_definition) = 'fb95dc8af00d4ca233ce74d31be5e484' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) IS DISTINCT FROM 'b1b891991482cd99c8c06a22c71178f3' THEN
    RAISE EXCEPTION 'Deepline candidate head preimage changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition, $before$          -- lab_arena_deepline_candidate_query
          FROM (
            SELECT uncertainty.*,
              uncertainty.entry_id <= GREATEST(COALESCE(p_after_entry_id, 0), 0)
                AS cursor_wrapped
            FROM public.lab_arena_runs AS scope
            CROSS JOIN LATERAL (
              SELECT entry.*
              FROM public.lab_arena_ledger AS entry
              WHERE entry.run_id = scope.run_id
              OFFSET 0
            ) AS uncertainty
            WHERE scope.round_id = p_round_id
              AND uncertainty.round_id = p_round_id
              AND uncertainty.entry_kind = 'uncertain'
              AND uncertainty.provider = 'deepline'
            ORDER BY cursor_wrapped, uncertainty.entry_id
            OFFSET 0
          ) AS uncertainty
$before$, $after$          -- lab_arena_deepline_candidate_head_scope_483
          -- lab_arena_deepline_candidate_query
          FROM (
            SELECT uncertainty.*,
              uncertainty.entry_id <= GREATEST(COALESCE(p_after_entry_id, 0), 0)
                AS cursor_wrapped
            FROM public.lab_arena_runs AS scope
            JOIN public.lab_arena_rounds AS scope_round
              ON scope_round.round_id = scope.round_id
            CROSS JOIN LATERAL (
              SELECT entry.*
              FROM public.lab_arena_ledger AS entry
              WHERE entry.run_id = scope.run_id
              OFFSET 0
            ) AS uncertainty
            WHERE scope.round_id = p_round_id
              AND (COALESCE(p_run_id, '') = '' OR scope.run_id = p_run_id)
              AND (scope_round.status <> 'published' OR scope.kind = 'score')
              AND uncertainty.round_id = p_round_id
              AND uncertainty.entry_kind = 'uncertain'
              AND uncertainty.provider = 'deepline'
              AND NOT EXISTS (
                SELECT 1
                FROM public.lab_arena_ledger AS later
                WHERE later.call_identity = uncertainty.call_identity
                  AND later.entry_id > uncertainty.entry_id
              )
            ORDER BY cursor_wrapped, uncertainty.entry_id
            OFFSET 0
          ) AS uncertainty
$after$);
  IF pg_catalog.md5(v_updated) IS DISTINCT FROM 'fb95dc8af00d4ca233ce74d31be5e484' THEN
    RAISE EXCEPTION 'Deepline candidate head target changed';
  END IF;
  -- CREATE OR REPLACE retains owner, grants, settings and the function OID.
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_security_after
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE))
       IS DISTINCT FROM 'fb95dc8af00d4ca233ce74d31be5e484'
     OR v_security_after IS DISTINCT FROM v_security THEN
    RAISE EXCEPTION 'Deepline candidate head readback changed';
  END IF;
END;
$deepline_candidate_head_scope$;

NOTIFY pgrst, 'reload schema';
COMMIT;
