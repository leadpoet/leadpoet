-- Finish an already-eligible baseline recovery before ordinary miner work.
-- All admission, retry, host-fault, deadline and frozen-input guards stay intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $baseline_recovery_priority$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  v_preimage CONSTANT TEXT := '8ea400f297a8a1da73536098a9e216d1017255d1a963c24cc9129ccb1445736c';
  v_postimage CONSTANT TEXT := '6432e422112cb17f0e487a18d355f881cc8d4e6a17889f93b89ad55a21017de1';
  v_definition TEXT;
  v_identity JSONB;
  v_old CONSTANT TEXT := $old$  ORDER BY
$old$;
  v_new CONSTANT TEXT := $new$  ORDER BY
    -- lab_arena_baseline_recovery_priority_v1: finish the existing baseline
    -- recovery before ordinary miner work; every admission guard stays above.
    CASE WHEN runs.kind = 'execute' AND runs.stage = 1
              AND public.lab_arena__baseline_recovery_phase_v1(v_round,runs)
      THEN 0 ELSE 1 END,
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena baseline recovery priority security shape differs';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
    OR (pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_old,'')))<>pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena baseline recovery priority preimage differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(
      pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE),'sha256'),'hex')
      <>v_postimage THEN
    RAISE EXCEPTION 'Arena baseline recovery priority readback differs';
  END IF;
END;
$baseline_recovery_priority$;

COMMIT;
