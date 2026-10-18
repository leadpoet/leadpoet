-- A published round admits only score-call billing through its three pinned
-- closed-score helpers. The V2 successful-execute lane cannot match them.
-- Skip its ledger scan while preserving every final eligibility guard.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $successful_deepline_published_fast_exit$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)';
  v_helper TEXT;
  v_hash TEXT;
  v_definition TEXT;
  v_updated TEXT;
  v_security JSONB;
  v_security_after JSONB;
BEGIN
  -- These exact published-round helpers are score-only. A helper change must
  -- review this implication before V2 can retain the fast exit.
  FOR v_helper, v_hash IN SELECT * FROM (VALUES
    ('public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)', '7312e9ca80018f634cd3799cd4a90093'),
    ('public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)', 'b096b9440dd66037b9f286632c5b27a9'),
    ('public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)', '2827d938ea977794ef3427aa9d7517dc')
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
      RAISE EXCEPTION 'successful Deepline published helper prerequisite changed: %', v_helper;
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
    RAISE EXCEPTION 'successful Deepline published security shape changed';
  END IF;
  IF pg_catalog.md5(v_definition) = '6cc6a84e068ce445dc25769a070075ae' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) IS DISTINCT FROM 'ea4cd7bc3fd18d3861c6a05addfcb69e' THEN
    RAISE EXCEPTION 'successful Deepline published preimage changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition,
    $before$WHERE early.round_id = p_round_id
              AND (COALESCE(p_run_id, '') = '' OR early.run_id = p_run_id)$before$,
    $after$WHERE early.round_id = p_round_id
              AND (p_successful_execute_only IS DISTINCT FROM TRUE
                OR EXISTS (
                  SELECT 1 FROM public.lab_arena_rounds AS early_round
                  WHERE early_round.round_id = p_round_id
                    AND early_round.status <> 'published'
                ))
              AND (COALESCE(p_run_id, '') = '' OR early.run_id = p_run_id)$after$);
  IF pg_catalog.md5(v_updated) IS DISTINCT FROM '6cc6a84e068ce445dc25769a070075ae' THEN
    RAISE EXCEPTION 'successful Deepline published target changed';
  END IF;
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_security_after
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE))
       IS DISTINCT FROM '6cc6a84e068ce445dc25769a070075ae'
     OR v_security_after IS DISTINCT FROM v_security THEN
    RAISE EXCEPTION 'successful Deepline published readback changed';
  END IF;
END;
$successful_deepline_published_fast_exit$;

NOTIFY pgrst, 'reload schema';
COMMIT;
