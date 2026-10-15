-- Bound the existing original-uncertainty proof to its nonnull call and run.
-- Keep every null-safe identity check and the text receipt comparison intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $indexed_infrastructure_cost_proof$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__execute_infrastructure_cost_exempt_v1(public.lab_arena_runs)';
  v_preimage CONSTANT TEXT := '0f862bf2fdccab3c15a8d863120d7c4e290391f03691910c7487d92aadd456a8';
  v_postimage CONSTANT TEXT := '295bace433f093c67e2bc28b76b0c2bf5ed1f2fe1d39b609446e4a19bcd07bdb';
  v_definition TEXT;
  v_identity JSONB;
  v_after_identity JSONB;
  v_old CONSTANT TEXT := $old$WHERE original.entry_id::TEXT=head.entry_doc$old$;
  v_new CONSTANT TEXT := $new$WHERE original.call_identity=head.call_identity
                   AND original.run_id=head.run_id
                   AND original.entry_id::TEXT=head.entry_doc$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig,language.lanname)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  JOIN pg_catalog.pg_language language ON language.oid=p.prolang
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
      ARRAY['search_path=pg_catalog, public']::TEXT[],'sql') THEN
    RAISE EXCEPTION 'Arena infrastructure cost lookup security shape differs';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
        pg_catalog.replace(v_definition,v_new,'')))<>pg_catalog.length(v_new)
      OR pg_catalog.strpos(v_definition,v_old)>0 THEN
      RAISE EXCEPTION 'Arena infrastructure cost lookup replay differs';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
    OR (pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_old,'')))<>pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena infrastructure cost lookup preimage differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(
      pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE),'sha256'),'hex')
      <>v_postimage THEN
    RAISE EXCEPTION 'Arena infrastructure cost lookup readback differs';
  END IF;
  SELECT pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig,language.lanname)
  INTO v_after_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  JOIN pg_catalog.pg_language language ON language.oid=p.prolang
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_after_identity IS DISTINCT FROM v_identity THEN
    RAISE EXCEPTION 'Arena infrastructure cost lookup security readback differs';
  END IF;
END;
$indexed_infrastructure_cost_proof$;

COMMIT;
