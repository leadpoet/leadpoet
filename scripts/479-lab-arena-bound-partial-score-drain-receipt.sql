-- A failed score attempt may durably retain its validated legacy partial
-- artifact. Reuse the completion/retry binding for its restart receipt.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $bound_partial_score_drain479$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__restart_drain_state_v1()';
  v_helper_signature CONSTANT TEXT :=
    'public.lab_arena__legacy_partial_bound(public.lab_arena_runs,jsonb,text)';
  v_helper_image CONSTANT TEXT := '5e25500320b1d60a8b26bccf5964ada24630dad639917d5cd72d77d58544b3df';
  v_preimage CONSTANT TEXT := '5c1d6335fd6160b4240b0cd60b003f2892505850df6b510a4e275057ccf5980c';
  v_postimage CONSTANT TEXT := '65384dfe99fbbac1f484139cfcf2a213079601b8d3af458a7153ec0ef7b2438a';
  v_old_gate CONSTANT TEXT := $old$          AND COALESCE(runs.output_ref, '') = ''$old$;
  v_new_gate CONSTANT TEXT := $new$          AND (COALESCE(runs.output_ref, '') = ''
            OR (runs.kind = 'score'
              AND EXISTS (SELECT 1 FROM public.lab_arena_rounds AS partial_round
                WHERE partial_round.round_id = runs.round_id
                  AND partial_round.evaluation_date::DATE >= DATE '2026-10-14')
              AND public.lab_arena__legacy_partial_bound(
                runs, runs.result_doc -> 'legacy_partial_checkpoint', runs.output_ref)))$new$;
  v_definition TEXT;
  v_helper_definition TEXT;
  v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig)
  INTO v_helper_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_helper_signature::REGPROCEDURE;
  IF pg_catalog.encode(extensions.digest(v_helper_definition,'sha256'),'hex')
       IS DISTINCT FROM v_helper_image
     OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
       '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
       ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena legacy partial binding differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_signature::REGPROCEDURE;
  IF v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
    ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena partial score drain security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_old_gate,'')))<>pg_catalog.length(v_old_gate) THEN
    RAISE EXCEPTION 'Arena partial score drain preimage differs' USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old_gate,v_new_gate);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena partial score drain postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena partial score drain readback differs' USING ERRCODE='55000';
  END IF;
END;
$bound_partial_score_drain479$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
